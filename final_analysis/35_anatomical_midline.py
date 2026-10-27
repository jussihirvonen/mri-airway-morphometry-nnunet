#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 10): maximum midline shift
referenced to the patient's own midsagittal plane instead of the image centre.

For every patient the midsagittal plane is estimated from the T1C image by
bilateral-symmetry maximisation: on each axial slice spanned by the airway the
slice is rotated by a candidate yaw angle and the mirror position that
maximises the normalised correlation between the image and its left-right
reflection is found (FFT-based, all mirror positions at once). The yaw angle is
estimated once per patient (median over five slices, -8..8 deg); the per-slice
mirror positions are then fitted by a straight line over z (= the trace of the
midsagittal plane; its slope is the in-plane roll). The airway centroid is
measured in the same rotated frame, and

    maxMLS_anat = max_z | centroid_x(z) - midplane_x(z) |   (mm)

Also reported: head yaw, roll, lateral offset of the head midplane from the
image centre at mid-airway, and a symmetry score for QC. Associations with
ICU admission are then recomputed (Mann-Whitney U, rank-biserial r, AUC) for
the original (image-centre) and the anatomical-midline maxMLS, full cohort and
abscess-positive subset. Montages of the estimated midline on 16 cases are
written to 35_midline_montages/ for visual QC.

Inputs: nnunet_setup/Dataset005_NeckAirway/imagesTr/caseXXX_0000.nii.gz (T1C),
        the reference-mask folder, XXXXa.nii.gz (manual masks), airway_morphology_merged.csv
Output: 35_anatomical_midline.csv (per case), 35_anatomical_midline.txt,
        35_anatomical_midline_tests.csv, 35_midline_montages/*.png
"""
import os, re, sys
import json
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from skimage.filters import threshold_otsu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
FA = BASE / "final_analysis"; REF_MASKS = BASE / _CFG.get("reference_masks", "reference_masks")
IMG = BASE / "nnunet_setup" / "Dataset005_NeckAirway" / "imagesTr"
MONT = FA / "35_midline_montages"; MONT.mkdir(exist_ok=True)
PIX = 1.2            # working in-plane resolution (mm)
ANGLES = np.arange(-8, 9, 2)
N_ANGLE_SLICES = 5


def prep_slice(sl, mask_sl, zoom):
    """Return (image, airway mask) zoomed to PIX mm and TRANSPOSED so that axis 0 = A->P and
    axis 1 = R->L (NIfTI axcodes L,P,S: original axis 0 is left-right). The mirror search in
    sym_scores runs along axis 1, i.e. across the left-right direction."""
    s = ndimage.zoom(sl, zoom, order=1); m = ndimage.zoom(mask_sl.astype(float), zoom, order=0) > 0.5
    return s.T, m.T


def body_mask(s):
    t = threshold_otsu(s[s > 0]) if (s > 0).any() else 0
    b = s > 0.5 * t
    b = ndimage.binary_fill_holes(ndimage.binary_closing(b, iterations=2))
    lab, n = ndimage.label(b)
    if n > 1:
        sizes = ndimage.sum(b, lab, range(1, n + 1)); b = lab == (np.argmax(sizes) + 1)
    return b


def sym_scores(R, M, centre_col=None, window_px=None):
    """Normalised mirror-symmetry score for every mirror position (columns) of a 2D slice.
    R: zero-mean-within-body image (0 outside body); M: body mask (float).
    The mirror axis is searched within +/- window_px of centre_col (default: body-mask centroid
    column, +/- 15 mm) and only where the mirrored body overlap is at least 60 % of the body."""
    H, W = R.shape; n = 2 * W
    F = R[:, ::-1]; MF = M[:, ::-1]
    def xcorr(A, B):   # c[L] = sum_y sum_x A[y, x+L] B[y, x]
        return np.fft.irfft((np.fft.rfft(A, n, axis=1) * np.conj(np.fft.rfft(B, n, axis=1))).sum(0), n)
    C = xcorr(R, F); N1 = xcorr(R * R, MF); N2 = xcorr(M, F * F); OV = xcorr(M, MF)
    L = np.arange(n); L[L >= W] -= n
    score = C / np.sqrt(np.clip(N1 * N2, 1e-9, None))
    axis = (W - 1 + L) / 2.0                     # mirror axis column (pixel index)
    if centre_col is None:
        cols = np.where(M.sum(0) > 0)[0]; centre_col = cols.mean() if len(cols) else (W - 1) / 2
    if window_px is None: window_px = 15.0 / PIX
    ok = (np.abs(L) <= W // 2) & (np.abs(axis - centre_col) <= window_px) & (OV >= 0.6 * M.sum())
    score[~ok] = -np.inf
    if not np.isfinite(score).any():
        return float(centre_col), float("nan")
    i = int(np.argmax(score))
    return axis[i], score[i]


def analyse(cid):
    num = int(cid.replace("case", ""))
    im = nib.load(str(IMG / f"{cid}_0000.nii.gz")); mk = nib.load(str(REF_MASKS / f"{num:04d}a.nii.gz"))
    I = im.get_fdata(); Mk = mk.get_fdata() > 0
    vx, vy, vz = [float(z) for z in im.header.get_zooms()]
    assert abs(vx - vy) < 1e-3
    zoom = vx / PIX
    # largest component of mask (no-op on corrected masks)
    lab, n = ndimage.label(Mk, structure=ndimage.generate_binary_structure(3, 3))
    if n > 1:
        sizes = ndimage.sum(Mk, lab, range(1, n + 1)); Mk = lab == (np.argmax(sizes) + 1)
    zs = np.where(Mk.any(axis=(0, 1)))[0]
    hi = np.percentile(I[I > 0], 99.5)
    # --- yaw estimation on 5 slices
    pick = np.unique(np.linspace(zs.min(), zs.max(), N_ANGLE_SLICES).round().astype(int))
    best_angles = []
    cache = {}
    for z in pick:
        s, _ = prep_slice(np.clip(I[:, :, z], 0, hi), Mk[:, :, z], zoom)
        b = body_mask(s); R0 = np.where(b, s - s[b].mean(), 0.0)
        res = []
        for th in ANGLES:
            Rr = ndimage.rotate(R0, th, reshape=False, order=1); Mr = ndimage.rotate(b.astype(float), th, reshape=False, order=0)
            Rr = Rr * Mr
            ax, sc = sym_scores(Rr, Mr); res.append((sc, th))
        best_angles.append(max(res)[1])
    yaw = float(np.median(best_angles))
    # --- per-slice mirror axis and airway centroid at the fixed yaw
    rows = []
    for z in zs:
        s, am = prep_slice(np.clip(I[:, :, z], 0, hi), Mk[:, :, z], zoom)
        b = body_mask(s); R0 = np.where(b, s - s[b].mean(), 0.0)
        Rr = ndimage.rotate(R0, yaw, reshape=False, order=1); Mr = ndimage.rotate(b.astype(float), yaw, reshape=False, order=0)
        Ar = ndimage.rotate(am.astype(float), yaw, reshape=False, order=0) > 0.5
        if Ar.sum() == 0:   # tiny pinpoint lost by zoom -> fall back to unrotated centroid scaled
            cx = np.mean(np.where(Mk[:, :, z])[0]) * zoom
        else:
            cx = np.mean(np.where(Ar)[1])          # column index = left-right position
        ax, sc = sym_scores(Rr * Mr, Mr)
        rows.append((z, ax, cx, sc, Rr.shape[1]))
    T = np.array(rows, dtype=float)
    z_mm = T[:, 0] * vz; ax_px = T[:, 1]; cx_px = T[:, 2]; sc = T[:, 3]; W = T[0, 4]
    # robust straight-line fit of the midplane trace over z
    b1, b0 = np.polyfit(z_mm, ax_px, 1); resid = ax_px - (b0 + b1 * z_mm)
    keep = np.abs(resid) <= max(2.5 * resid.std(), 1.0)
    if keep.sum() >= 3: b1, b0 = np.polyfit(z_mm[keep], ax_px[keep], 1)
    mid_fit = b0 + b1 * z_mm
    # guard: a roll beyond 15 deg or a fit RMSE above 5 mm is implausible for a coil-positioned head;
    # fall back to a constant midplane at the median per-slice axis (translation-only correction)
    if abs(np.degrees(np.arctan(b1 * PIX))) > 15 or np.sqrt(np.mean((resid[keep] * PIX) ** 2)) > 5:
        b1 = 0.0; b0 = float(np.median(ax_px)); mid_fit = np.full_like(z_mm, b0); resid = ax_px - b0; keep = np.ones_like(keep)
    dev_anat = (cx_px - mid_fit) * PIX                      # mm, signed (rotated frame, axis0 = L)
    dev_anat_raw = (cx_px - ax_px) * PIX
    centre = (W - 1) / 2.0
    dev_img = (cx_px - centre) * PIX                        # image-centre reference in the same rotated frame
    roll_deg = float(np.degrees(np.arctan(b1 * PIX)))       # slope (mm per mm)
    head_offset = float((mid_fit[len(mid_fit) // 2] - centre) * PIX)
    return dict(case_id=cid, yaw_deg=yaw, roll_deg=roll_deg, head_offset_mm=head_offset,
                sym_score_mean=float(sc.mean()), sym_score_min=float(sc.min()), n_slices_fit=int(keep.sum()), n_slices=len(zs),
                maxMLS_img=float(np.max(np.abs(dev_img))), maxMLS_anat=float(np.max(np.abs(dev_anat))), maxMLS_anat_raw=float(np.max(np.abs(dev_anat_raw))),
                meanMLS_anat=float(np.mean(np.abs(dev_anat))), midline_fit_rmse_mm=float(np.sqrt(np.mean((resid[keep] * PIX) ** 2)))), \
           dict(I=I, Mk=Mk, zs=zs, yaw=yaw, zoom=zoom, hi=hi, mid_fit=mid_fit, cx_px=cx_px, W=W)


def montage(cid, aux, out):
    I, Mk, zs, yaw, zoom, hi, mid_fit, cx_px, W = [aux[k] for k in ("I", "Mk", "zs", "yaw", "zoom", "hi", "mid_fit", "cx_px", "W")]
    idx = [0, len(zs) // 2, len(zs) - 1]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    for a, i in zip(ax, idx):
        z = zs[i]
        s, am = prep_slice(np.clip(I[:, :, z], 0, hi), Mk[:, :, z], zoom)
        Rr = ndimage.rotate(s, yaw, reshape=False, order=1); Ar = ndimage.rotate(am.astype(float), yaw, reshape=False, order=0) > 0.5
        a.imshow(Rr, cmap="gray")       # rows = A->P (anterior at top), columns = R->L (patient left on image right)
        a.axvline(mid_fit[i], color="cyan", lw=1); a.axvline((W - 1) / 2, color="yellow", lw=0.8, ls=":")
        a.contour(Ar, levels=[.5], colors="lime", linewidths=0.8); a.axvline(cx_px[i], color="red", lw=0.8, ls="--")
        a.set_title(f"z={z}  anat {abs(cx_px[i]-mid_fit[i])*PIX:.1f} mm | img {abs(cx_px[i]-(W-1)/2)*PIX:.1f} mm", fontsize=9); a.axis("off")
    fig.suptitle(f"{cid}: yaw {yaw:+.0f} deg; cyan = anatomical midline (symmetry), yellow = image centre, red = airway centroid", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def main():
    df = pd.read_csv(FA / "airway_morphology_merged.csv"); df["case_id"] = df.case_id.astype(str)
    cases = sorted(df.case_id)
    out = []; auxs = {}
    for k, cid in enumerate(cases):
        try:
            r, aux = analyse(cid)
        except Exception as e:
            print("FAIL", cid, e, file=sys.stderr); continue
        out.append(r); auxs[cid] = aux if cid in _CFG.get("examples", {}).get("midline_debug", []) else None
        if k % 25 == 0: print(f"{k}/{len(cases)} {cid} yaw={r['yaw_deg']:+.0f} off={r['head_offset_mm']:+.1f} img={r['maxMLS_img']:.1f} anat={r['maxMLS_anat']:.1f}", flush=True)
    P = pd.DataFrame(out)
    d = df.merge(P, on="case_id")
    P.to_csv(FA / "35_anatomical_midline.csv", index=False, float_format="%.3f")
    # montages: fixed set + extremes of head offset / yaw / disagreement
    sel = set(c for c in auxs if auxs[c] is not None)
    sel |= set(d.nlargest(3, "head_offset_mm").case_id) | set(d.nsmallest(2, "head_offset_mm").case_id) | set(d.nlargest(2, "yaw_deg").case_id)
    d["disagree"] = (d.maxMLS_img - d.maxMLS_anat).abs(); sel |= set(d.nlargest(3, "disagree").case_id)
    for cid in sorted(sel):
        if auxs.get(cid) is None:
            _, auxs[cid] = analyse(cid)
        montage(cid, auxs[cid], MONT / f"{cid}.png")
    # --- report
    L = [f"Anatomical-midline maxMLS (symmetry-based midsagittal plane), n={len(d)}", ""]
    L.append("Check: image-centre maxMLS recomputed in the rotated/zoomed frame vs canonical rl_dev_max_abs_mm: "
             f"Spearman rho={spearmanr(d.rl_dev_max_abs_mm, d.maxMLS_img).statistic:.3f}, median |diff| {np.median(np.abs(d.rl_dev_max_abs_mm-d.maxMLS_img)):.2f} mm")
    L.append("Head positioning (median [IQR], range):")
    for c in ["yaw_deg", "roll_deg", "head_offset_mm"]:
        q = d[c].quantile([.25, .5, .75]); L.append(f"  {c:<16s} {q[.5]:+.1f} [{q[.25]:+.1f}, {q[.75]:+.1f}]  range {d[c].min():+.1f} to {d[c].max():+.1f};  |{c}| median {d[c].abs().median():.1f}, >5: {(d[c].abs()>5).sum()} ({100*(d[c].abs()>5).mean():.0f}%)")
    L.append(f"  symmetry score mean {d.sym_score_mean.median():.2f} (IQR {d.sym_score_mean.quantile(.25):.2f}-{d.sym_score_mean.quantile(.75):.2f}); midline fit RMSE median {d.midline_fit_rmse_mm.median():.1f} mm")
    L.append("")
    L.append(f"maxMLS image-centre vs anatomical: median {d.maxMLS_img.median():.1f} vs {d.maxMLS_anat.median():.1f} mm; Spearman rho={spearmanr(d.maxMLS_img, d.maxMLS_anat).statistic:.3f}")
    r1 = spearmanr(d.head_offset_mm.abs(), d.icu_binary); L.append(f"|head offset| vs ICU: Spearman rho={r1.statistic:+.3f} p={r1.pvalue:.3f}; |yaw| vs ICU rho={spearmanr(d.yaw_deg.abs(), d.icu_binary).statistic:+.3f}; |roll| vs ICU rho={spearmanr(d.roll_deg.abs(), d.icu_binary).statistic:+.3f}")
    L.append("")
    tests = []
    for label, sub in [("Full cohort", d), ("Abscess-positive", d[d.abs_positive == 1])]:
        y = sub.icu_binary.astype(int).values
        L.append(f"{label} (n={len(sub)}, ICU={int(y.sum())}):"); L.append(f"  {'variable':<18s} {'med noICU':>10s} {'med ICU':>8s} {'p':>9s} {'r_rb':>7s} {'AUC':>6s}")
        for c in ["rl_dev_max_abs_mm", "maxMLS_img", "maxMLS_anat", "maxMLS_anat_raw", "meanMLS_anat"]:
            x = sub[c].values; a, b = x[y == 0], x[y == 1]
            u, p = mannwhitneyu(b, a, alternative="two-sided"); rb = 2 * u / (len(a) * len(b)) - 1; auc = roc_auc_score(y, x)
            L.append(f"  {c:<18s} {np.median(a):>10.1f} {np.median(b):>8.1f} {p:>9.4f} {rb:>+7.3f} {auc:>6.3f}")
            tests.append(dict(cohort=label, variable=c, n=len(sub), icu=int(y.sum()), med_noicu=np.median(a), med_icu=np.median(b), p=p, r_rb=rb, auc=auc))
        for ax_ in ["retroinf"]:
            a = sub[sub[ax_] == 0].maxMLS_anat; b = sub[sub[ax_] == 1].maxMLS_anat
            L.append(f"  maxMLS_anat RPE- vs RPE+: {a.median():.1f} vs {b.median():.1f} mm, p={mannwhitneyu(a, b).pvalue:.2e}")
        s2 = sub.dropna(subset=["abs_diamax_mri"]); L.append(f"  maxMLS_anat ~ abscess diameter Spearman rho={spearmanr(s2.maxMLS_anat, s2.abs_diamax_mri).statistic:+.3f}; ~CRP rho={spearmanr(sub.dropna(subset=['crp']).maxMLS_anat, sub.dropna(subset=['crp']).crp).statistic:+.3f}")
        L.append("")
    pd.DataFrame(tests).to_csv(FA / "35_anatomical_midline_tests.csv", index=False, float_format="%.5f")
    (FA / "35_anatomical_midline.txt").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
