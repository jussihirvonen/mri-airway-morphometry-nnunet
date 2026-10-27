#!/usr/bin/env python3
"""
Paper-2 reanalysis: how nnU-Net predicted-mask postprocessing affects
manual-vs-predicted morphometric agreement.

Motivation: manual masks are OBSTRUCTION-BRIDGED (raters hallucinated voxels across
total obstructions to keep the lumen continuous); nnU-Net predicts only patent air,
so it leaves Z-gaps at obstruction sites. The current pipeline applies largest-CC to
predictions (01_run_full_morphology.analyze_case), which DELETES real tube beyond the
obstruction -> undercounts predicted volume/length and may fake a "loses fine-caliber"
result.

Three predicted-mask variants compared (manual side fixed, computed identically):
  1. largest_cc  : main 26-conn component only (current behaviour)
  2. union       : all components (expert reader: "include as a whole")
  3. bridged     : main + craniocaudally column-aligned fragments only
                   (drop laterally-separate FP); mirrors the human obstruction bridge

Also classifies every multi-component prediction and flags gross over-predictions.
Outputs: 27_pred_postproc_report.txt, 27_pred_classification.csv
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
ALIGN_THRESH = 0.30      # fragment footprint fraction inside main column to count as aligned
GROSS_RATIO = 1.5        # pred(union)/manual volume above this = gross over-prediction


def feats(binary, sp):
    vx, vy, vz = sp
    sd = []
    for z in range(binary.shape[2]):
        sl = binary[:, :, z]; nv = sl.sum()
        if nv > 0:
            c = np.where(sl > 0)
            sd.append((nv * vx * vy, c[0].mean()))
    if not sd:
        return None
    areas = [a for a, _ in sd]
    V = binary.sum() * vx * vy * vz
    cxi = binary.shape[0] / 2
    rl = [(cx - cxi) * vx for _, cx in sd]
    surf = binary & ~binary_erosion(binary)
    SA = surf.sum() * (vx * vy + vx * vz + vy * vz) / 3
    sph = (np.pi ** (1 / 3) * (6 * V) ** (2 / 3)) / SA if SA > 0 else 0
    c = np.where(binary > 0)
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


def variants(P):
    lab, n = ndimage.label(P, structure=S26)
    if n <= 1:
        return P, P, P, n, []          # lcc, union, bridged identical
    sizes = ndimage.sum(P, lab, range(1, n + 1))
    main_id = int(np.argmax(sizes)) + 1
    main = lab == main_id
    main_fp = main.any(axis=2)
    bridged = main.copy()
    frag_info = []
    for k in range(1, n + 1):
        if k == main_id:
            continue
        comp = lab == k
        fp = comp.any(axis=2)
        ov = (main_fp & fp).sum() / fp.sum() if fp.sum() else 0
        cz = np.where(comp.any(axis=(0, 1)))[0]
        aligned = ov >= ALIGN_THRESH
        if aligned:
            bridged |= comp
        frag_info.append({"vox": int(sizes[k - 1]), "z": (int(cz.min()), int(cz.max())),
                          "footprint_overlap": round(float(ov), 2), "aligned": aligned})
    return main, P, bridged, n, frag_info


def icc(frame, mcol, pcol):
    sub = frame[[mcol, pcol]].dropna(); m = len(sub)
    long = pd.DataFrame({"t": list(range(m)) * 2, "r": ["m"] * m + ["p"] * m,
                         "s": list(sub[mcol]) + list(sub[pcol])})
    r = pg.intraclass_corr(data=long, targets="t", raters="r", ratings="s")
    row = r[r.Type == "ICC2"].iloc[0]
    return float(row["ICC"]), row["CI95%"][0], row["CI95%"][1]


def main():
    rows, cls = [], []
    for pf in sorted(PRED.glob("case*.nii.gz")):
        cid = pf.stem.replace(".nii", "")
        lf = LAB / pf.name
        if not lf.exists():
            continue
        mn = nib.load(str(lf)); pn = nib.load(str(pf))
        sp = tuple(float(z) for z in mn.header.get_zooms()[:3])
        M = mn.get_fdata() > 0
        P = pn.get_fdata() > 0
        if M.sum() == 0 or P.sum() == 0:
            continue
        # manual: bridged already; use largest CC for parity with published manual side
        ml, _, _, _, _ = variants(M)
        fm = feats(ml, sp)
        plcc, puni, pbr, npred, frag = variants(P)
        fl, fu, fb = feats(plcc, sp), feats(puni, sp), feats(pbr, sp)
        man_vol = fm["volume_ml"]; uni_vol = fu["volume_ml"]
        gross = (uni_vol / man_vol) > GROSS_RATIO if man_vol > 0 else False
        row = {"case_id": cid, "ncomp": npred, "gross": gross}
        for f in FEATS:
            row[f"man_{f}"] = fm[f]; row[f"lcc_{f}"] = fl[f]
            row[f"uni_{f}"] = fu[f]; row[f"br_{f}"] = fb[f]
        rows.append(row)
        if npred > 1:
            dropped_uni = uni_vol - fl["volume_ml"]
            dropped_br = uni_vol - fb["volume_ml"]
            aligned_vox = sum(fr["vox"] for fr in frag if fr["aligned"])
            fp_vox = sum(fr["vox"] for fr in frag if not fr["aligned"])
            cat = ("GROSS_OVERPRED" if gross else
                   "Z_DISCONTINUITY" if aligned_vox >= fp_vox else "FP_LATERAL")
            cls.append({"case_id": cid, "ncomp": npred, "category": cat, "gross": gross,
                        "man_vol_ml": round(man_vol, 2), "pred_union_vol_ml": round(uni_vol, 2),
                        "aligned_vox": aligned_vox, "fp_vox": fp_vox,
                        "frags": frag})
    df = pd.DataFrame(rows)
    dfng = df[~df.gross]                                   # excluding gross over-preds

    out = []
    def pr(s=""):
        print(s); out.append(str(s))

    pr("=" * 92)
    pr(f"PAPER-2 PREDICTED-MASK POSTPROCESSING RECOMPUTE  (n={len(df)} cases; "
       f"{(df.ncomp>1).sum()} multi-component; {df.gross.sum()} gross over-pred)")
    pr("=" * 92)
    pr(f"\nManual-vs-predicted ICC(2,1) by postprocessing variant:")
    pr(f"{'feature':<20}{'largest_CC':>22}{'union':>22}{'bridged':>22}")
    for f in FEATS:
        a = icc(df, f"man_{f}", f"lcc_{f}")
        b = icc(df, f"man_{f}", f"uni_{f}")
        c = icc(df, f"man_{f}", f"br_{f}")
        pr(f"{f:<20}{a[0]:>8.3f} [{a[1]:.2f},{a[2]:.2f}]{b[0]:>8.3f} [{b[1]:.2f},{b[2]:.2f}]"
           f"{c[0]:>8.3f} [{c[1]:.2f},{c[2]:.2f}]")
    pr(f"\nSame, EXCLUDING {df.gross.sum()} gross over-predictions (n={len(dfng)}):")
    pr(f"{'feature':<20}{'largest_CC':>22}{'union':>22}{'bridged':>22}")
    for f in FEATS:
        a = icc(dfng, f"man_{f}", f"lcc_{f}")
        b = icc(dfng, f"man_{f}", f"uni_{f}")
        c = icc(dfng, f"man_{f}", f"br_{f}")
        pr(f"{f:<20}{a[0]:>8.3f} [{a[1]:.2f},{a[2]:.2f}]{b[0]:>8.3f} [{b[1]:.2f},{b[2]:.2f}]"
           f"{c[0]:>8.3f} [{c[1]:.2f},{c[2]:.2f}]")

    cdf = pd.DataFrame(cls)
    pr(f"\nMulti-component prediction classification ({len(cdf)} cases):")
    pr(cdf["category"].value_counts().to_string())
    pr(f"\nTop cases by dropped real tube (union - largest_CC volume):")
    cdf["dropped_ml"] = [round((r.pred_union_vol_ml - df[df.case_id==r.case_id]['lcc_volume_ml'].iloc[0]),2) for r in cdf.itertuples()]
    for _, r in cdf.sort_values("dropped_ml", ascending=False).head(15).iterrows():
        pr(f"  {r.case_id}: {r.category:<16} ncomp={r.ncomp} man={r.man_vol_ml} "
           f"union={r.pred_union_vol_ml} dropped={r.dropped_ml}mL aligned/fp={r.aligned_vox}/{r.fp_vox}")

    (OUT / "27_pred_postproc_report.txt").write_text("\n".join(out))
    cdf.drop(columns=["frags"]).to_csv(OUT / "27_pred_classification.csv", index=False)
    df.to_csv(OUT / "27_pred_postproc_percase.csv", index=False)
    pr(f"\nSaved 27_pred_postproc_report.txt / 27_pred_classification.csv / 27_pred_postproc_percase.csv")


if __name__ == "__main__":
    main()
