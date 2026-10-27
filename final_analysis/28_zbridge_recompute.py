#!/usr/bin/env python3
"""
Real Z-bridge of nnU-Net predictions, mirroring the manual obstruction-hallucination.

For each predicted mask:
  - 26-conn label; keep the largest component as the main tube.
  - For every OTHER component whose z-range is DISJOINT from the main (i.e. a separate
    craniocaudal tube part above/below the main) and whose contact end is within MAXLAT
    voxels of the main in-plane (so we don't bridge to a far-off blob), include it and
    BRIDGE the zero-prediction gap with a SINGLE-VOXEL COLUMN (linear (x,y) interpolation
    between the centroids of the bounding present slices, one voxel per empty slice).
  - Components that overlap the main's z-range (lateral side-blobs / FP) are NOT bridged.

Tabs kept: how many cases bridged, gap length (slices) per case, max lateral wander.
Compares manual-vs-predicted morphometry ICC: largest_CC | union | zbridge (gross excl).
Outputs: 28_zbridge_report.txt, 28_zbridge_percase.csv
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from scipy.ndimage import binary_erosion
import pingouin as pg

S26 = ndimage.generate_binary_structure(3, 3)
BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
LAB = BASE / "nnunet_setup" / "Dataset005_NeckAirway" / "labelsTr"
PRED = BASE / "final_analysis" / "predicted_masks"
OUT = BASE / "final_analysis"
FEATS = ["volume_ml", "min_area_mm2", "compactness", "sphericity",
         "rl_dev_max_abs_mm", "len_below_50mm2_mm"]
MAXLAT = 20          # max in-plane voxel distance between part ends to accept as one tube
GROSS_RATIO = 1.5    # union/manual volume above this = gross over-prediction (excluded)


def feats(b, sp):
    vx, vy, vz = sp
    sd = []
    for z in range(b.shape[2]):
        sl = b[:, :, z]; nv = sl.sum()
        if nv > 0:
            c = np.where(sl > 0)
            sd.append((nv * vx * vy, c[0].mean()))
    if not sd:
        return None
    areas = [a for a, _ in sd]
    V = b.sum() * vx * vy * vz
    cxi = b.shape[0] / 2
    rl = [(cx - cxi) * vx for _, cx in sd]
    surf = b & ~binary_erosion(b)
    SA = surf.sum() * (vx * vy + vx * vz + vy * vz) / 3
    sph = (np.pi ** (1 / 3) * (6 * V) ** (2 / 3)) / SA if SA > 0 else 0
    c = np.where(b > 0)
    bbox = ((np.ptp(c[0]) + 1) * vx) * ((np.ptp(c[1]) + 1) * vy) * ((np.ptp(c[2]) + 1) * vz)

    def run(bl):
        mx = cur = 0
        for v in bl:
            cur = cur + 1 if v else 0; mx = max(mx, cur)
        return mx
    return {"volume_ml": V / 1000, "min_area_mm2": min(areas),
            "compactness": V / bbox if bbox > 0 else 0, "sphericity": sph,
            "rl_dev_max_abs_mm": max(abs(d) for d in rl),
            "len_below_50mm2_mm": run([a < 50 for a in areas]) * vz}


def centroid(mask2d):
    c = np.where(mask2d)
    return np.array([c[0].mean(), c[1].mean()])


def make_variants(P):
    """returns largest_cc, union, zbridge masks + bridge stats dict"""
    lab, n = ndimage.label(P, structure=S26)
    if n <= 1:
        return P, P, P, dict(ncomp=1, bridged=False, n_gaps=0, n_bridge_slices=0,
                             max_gap=0, max_lat=0.0)
    sizes = ndimage.sum(P, lab, range(1, n + 1))
    main_id = int(np.argmax(sizes)) + 1
    main = lab == main_id
    mz = np.where(main.any(axis=(0, 1)))[0]; mzlo, mzhi = int(mz.min()), int(mz.max())
    incl = main.copy()
    for k in range(1, n + 1):
        if k == main_id:
            continue
        comp = lab == k
        cz = np.where(comp.any(axis=(0, 1)))[0]; fzlo, fzhi = int(cz.min()), int(cz.max())
        if fzlo > mzhi:        # above main
            cm = centroid(main[:, :, mzhi]); cf = centroid(comp[:, :, fzlo])
        elif fzhi < mzlo:      # below main
            cm = centroid(main[:, :, mzlo]); cf = centroid(comp[:, :, fzhi])
        else:
            continue           # z-overlapping side-blob -> not a craniocaudal part
        if np.hypot(*(cm - cf)) <= MAXLAT:
            incl |= comp
    union = P
    # fill internal empty slices of incl with single-voxel columns
    zp = [z for z in range(incl.shape[2]) if incl[:, :, z].any()]
    zmin, zmax = min(zp), max(zp)
    empties = [z for z in range(zmin, zmax + 1) if not incl[:, :, z].any()]
    bridged = incl.copy()
    n_gaps = 0; max_gap = 0; max_lat = 0.0
    if empties:
        gaps = []; run = [empties[0]]
        for z in empties[1:]:
            if z == run[-1] + 1:
                run.append(z)
            else:
                gaps.append(run); run = [z]
        gaps.append(run)
        n_gaps = len(gaps)
        for g in gaps:
            a, b = g[0], g[-1]; zb, za = a - 1, b + 1
            p0, p1 = centroid(bridged[:, :, zb]), centroid(bridged[:, :, za])
            max_gap = max(max_gap, len(g)); max_lat = max(max_lat, float(np.hypot(*(p1 - p0))))
            for z in g:
                t = (z - zb) / (za - zb)
                x = int(round(p0[0] + t * (p1[0] - p0[0])))
                y = int(round(p0[1] + t * (p1[1] - p0[1])))
                bridged[min(max(x, 0), bridged.shape[0] - 1),
                        min(max(y, 0), bridged.shape[1] - 1), z] = True
    stats = dict(ncomp=n, bridged=len(empties) > 0, n_gaps=n_gaps,
                 n_bridge_slices=len(empties), max_gap=max_gap, max_lat=round(max_lat, 1))
    return main, union, bridged, stats


def icc(frame, mcol, pcol):
    sub = frame[[mcol, pcol]].dropna(); m = len(sub)
    long = pd.DataFrame({"t": list(range(m)) * 2, "r": ["m"] * m + ["p"] * m,
                         "s": list(sub[mcol]) + list(sub[pcol])})
    r = pg.intraclass_corr(data=long, targets="t", raters="r", ratings="s")
    row = r[r.Type == "ICC2"].iloc[0]
    return row["ICC"], row["CI95%"][0], row["CI95%"][1]


def main():
    rows = []
    for pf in sorted(PRED.glob("case*.nii.gz")):
        cid = pf.stem.replace(".nii", ""); lf = LAB / pf.name
        if not lf.exists():
            continue
        mn = nib.load(str(lf)); pn = nib.load(str(pf))
        sp = tuple(float(z) for z in mn.header.get_zooms()[:3])
        M = mn.get_fdata() > 0; P = pn.get_fdata() > 0
        if M.sum() == 0 or P.sum() == 0:
            continue
        ml, _, _, _ = make_variants(M)
        fm = feats(ml, sp)
        plcc, puni, pbr, st = make_variants(P)
        fl, fu, fb = feats(plcc, sp), feats(puni, sp), feats(pbr, sp)
        gross = (fu["volume_ml"] / fm["volume_ml"]) > GROSS_RATIO if fm["volume_ml"] > 0 else False
        row = {"case_id": cid, "gross": gross, **{f"st_{k}": v for k, v in st.items()}}
        for f in FEATS:
            row[f"man_{f}"] = fm[f]; row[f"lcc_{f}"] = fl[f]
            row[f"uni_{f}"] = fu[f]; row[f"br_{f}"] = fb[f]
        rows.append(row)
    df = pd.DataFrame(rows)
    dfng = df[~df.gross]

    out = []
    def pr(s=""):
        print(s); out.append(str(s))

    br = df[df.st_bridged]
    pr("=" * 90)
    pr(f"REAL Z-BRIDGE RECOMPUTE (single-voxel column)   n={len(df)} cases")
    pr("=" * 90)
    pr(f"\nCases bridged: {len(br)} / {len(df)}  ({len(df[df.st_ncomp>1])} were multi-component)")
    pr(f"Cases excluded as gross over-pred: {df.gross.sum()}  (bridged among them: {br.gross.sum()})")
    if len(br):
        bs = br.st_n_bridge_slices
        pr(f"\nBridge length (total empty slices filled per case):")
        pr(f"  min={bs.min()}  median={bs.median():.0f}  mean={bs.mean():.1f}  max={bs.max()}")
        pr(f"  distribution (#slices: #cases): "
           + ", ".join(f"{k}:{v}" for k, v in bs.value_counts().sort_index().items()))
        pr(f"  gaps per case: max={br.st_n_gaps.max()}  | max single-gap slices={br.st_max_gap.max()}")
        pr(f"  lateral wander of bridge (voxels): median={br.st_max_lat.median():.1f}  max={br.st_max_lat.max():.1f}")

    pr(f"\nManual-vs-predicted ICC(2,1)  (n={len(dfng)}, gross over-pred excluded):")
    pr(f"{'feature':<20}{'largest_CC':>22}{'union':>22}{'Z-bridge':>22}")
    for f in FEATS:
        a = icc(dfng, f"man_{f}", f"lcc_{f}")
        b = icc(dfng, f"man_{f}", f"uni_{f}")
        c = icc(dfng, f"man_{f}", f"br_{f}")
        pr(f"{f:<20}{a[0]:>8.3f} [{a[1]:.2f},{a[2]:.2f}]{b[0]:>8.3f} [{b[1]:.2f},{b[2]:.2f}]"
           f"{c[0]:>8.3f} [{c[1]:.2f},{c[2]:.2f}]")

    pr(f"\nBridged cases (sorted by bridge length):")
    for _, r in br.sort_values("st_n_bridge_slices", ascending=False).head(20).iterrows():
        tag = " [GROSS-excl]" if r.gross else ""
        pr(f"  {r.case_id}: {int(r.st_n_bridge_slices)} slice(s), {int(r.st_n_gaps)} gap(s), "
           f"lat {r.st_max_lat:.0f}vox{tag}")

    (OUT / "28_zbridge_report.txt").write_text("\n".join(out))
    df.to_csv(OUT / "28_zbridge_percase.csv", index=False)
    pr(f"\nSaved 28_zbridge_report.txt / 28_zbridge_percase.csv")


if __name__ == "__main__":
    main()
