#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 10): head positioning and
the maximum midline shift (maxMLS).

maxMLS in the manuscript = max over slices of |centroid_x - image centre| (mm).
A lateral offset of the patient in the field of view, or an in-plane head tilt
(roll), adds a slice-wise linear component to the centroid trajectory that is
not caused by disease. This script separates the two:

  (1) global offset  = mean signed lateral deviation (whole-airway translation)
  (2) tilt angle     = arctan(slope of centroid_x on z) in degrees (in-plane roll)
  (3) detrended maxMLS = max |residual| after least-squares straight-line fit of
      centroid_x on z over all airway slices — the local bowing of the air
      column relative to the patient's own airway axis, invariant to translation
      and tilt.

The script reports the distribution of offset and tilt, their association with
ICU admission (are ICU patients positioned differently?), the agreement between
original and detrended maxMLS, and the ICU association of the detrended feature
(Mann-Whitney U, rank-biserial r, AUC) in the full cohort and abscess-positive
subset, mirroring Table 3.

Output: 31_midline_positioning.csv (per case) / 31_midline_positioning.txt
"""
import os, re
import json
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
import pingouin as pg

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
FA = BASE / "final_analysis"
REF_MASKS = BASE / _CFG.get("reference_masks", "reference_masks")
MERGED = FA / "airway_morphology_merged.csv"


def per_case(mask_file):
    nii = nib.load(str(mask_file))
    m = nii.get_fdata()
    vx, vy, vz = [float(z) for z in nii.header.get_zooms()]
    b = (m > 0).astype(int)
    lab, n = ndimage.label(b, structure=ndimage.generate_binary_structure(3, 3))
    if n > 1:
        sizes = ndimage.sum(b, lab, range(1, n + 1))
        b = (lab == (np.argmax(sizes) + 1)).astype(int)
    cx_img = b.shape[0] / 2
    zs, cxs = [], []
    for z in range(b.shape[2]):
        sl = b[:, :, z]
        if sl.sum() > 0:
            coords = np.where(sl > 0)
            zs.append(z * vz); cxs.append((np.mean(coords[0]) - cx_img) * vx)
    zs, cxs = np.array(zs), np.array(cxs)
    orig_max = np.max(np.abs(cxs))
    offset = float(np.mean(cxs))
    if len(zs) > 2:
        slope, intercept = np.polyfit(zs, cxs, 1)
        resid = cxs - (slope * zs + intercept)
        tilt_deg = float(np.degrees(np.arctan(slope)))
    else:
        resid = cxs - offset; tilt_deg = 0.0
    return dict(maxMLS_orig=orig_max, offset_mm=offset, tilt_deg=tilt_deg,
                maxMLS_detrended=float(np.max(np.abs(resid))),
                maxMLS_centered=float(np.max(np.abs(cxs - offset))), n_slices=len(zs))


def mwu_block(sub, col, y):
    x = sub[col].values
    a, b = x[y == 0], x[y == 1]
    u, p = mannwhitneyu(b, a, alternative="two-sided")
    rb = 2 * u / (len(a) * len(b)) - 1
    return np.median(a), np.median(b), p, rb, roc_auc_score(y, x)


def main():
    df = pd.read_csv(MERGED); df["case_id"] = df.case_id.astype(str)
    rows = []
    for f in sorted(REF_MASKS.glob("*a.nii.gz")):
        num = int(re.match(r"(\d+)a", f.name).group(1))
        cid = f"case{num:03d}"
        d = per_case(f); d["case_id"] = cid; rows.append(d)
    P = pd.DataFrame(rows).rename(columns={"n_slices": "n_slices_air"})
    d = df.merge(P, on="case_id")
    assert np.allclose(d.rl_dev_max_abs_mm, d.maxMLS_orig, atol=0.06), "orig maxMLS mismatch vs canonical"
    L = ["Head positioning and maximum midline shift (n=%d)" % len(d), ""]
    L.append("Distribution of positioning parameters (median [IQR], range):")
    for c in ["offset_mm", "tilt_deg"]:
        q = d[c].quantile([.25, .5, .75]); L.append(f"  {c:<12s} {q[.5]:+.1f} [{q[.25]:+.1f}, {q[.75]:+.1f}]  range {d[c].min():+.1f} to {d[c].max():+.1f}")
        a = d[c].abs().quantile([.25, .5, .75]); L.append(f"  |{c}|{'':<9s} {a[.5]:.1f} [{a[.25]:.1f}, {a[.75]:.1f}]  max {d[c].abs().max():.1f}")
    L.append(f"  |offset| > 5 mm: {(d.offset_mm.abs()>5).sum()} ({100*(d.offset_mm.abs()>5).mean():.1f}%);  |tilt| > 5 deg: {(d.tilt_deg.abs()>5).sum()} ({100*(d.tilt_deg.abs()>5).mean():.1f}%)")
    L.append("")
    L.append("Agreement original vs detrended maxMLS:")
    icc = pg.intraclass_corr(data=pd.melt(d[["case_id", "maxMLS_orig", "maxMLS_detrended"]], id_vars="case_id"), targets="case_id", raters="variable", ratings="value")
    icc2 = icc[icc.Type == "ICC2"].iloc[0]
    rho = spearmanr(d.maxMLS_orig, d.maxMLS_detrended)
    L.append(f"  ICC(2,1) = {icc2.ICC:.3f} [{icc2['CI95%'][0]:.2f}, {icc2['CI95%'][1]:.2f}];  Spearman rho = {rho.statistic:.3f}")
    L.append(f"  median original {d.maxMLS_orig.median():.1f} mm vs detrended {d.maxMLS_detrended.median():.1f} mm (detrended is smaller by construction)")
    L.append("")
    out = []
    for label, sub in [("Full cohort", d), ("Abscess-positive", d[d.abs_positive == 1])]:
        y = sub.icu_binary.astype(int).values
        L.append(f"{label} (n={len(sub)}, ICU={int(y.sum())}):")
        L.append(f"  {'variable':<22s} {'med noICU':>10s} {'med ICU':>8s} {'p':>9s} {'r_rb':>7s} {'AUC':>6s}")
        for c in ["maxMLS_orig", "maxMLS_detrended", "maxMLS_centered", "offset_mm", "tilt_deg"]:
            col = c
            if c in ("offset_mm", "tilt_deg"):
                sub = sub.assign(**{c + "_abs": sub[c].abs()}); col = c + "_abs"
            ma, mb, p, rb, auc = mwu_block(sub, col, y)
            L.append(f"  {col:<22s} {ma:>10.1f} {mb:>8.1f} {p:>9.4f} {rb:>+7.3f} {auc:>6.3f}")
            out.append(dict(cohort=label, variable=col, n=len(sub), icu=int(y.sum()), med_noicu=ma, med_icu=mb, p=p, r_rb=rb, auc=auc))
        # Spearman of positioning with severity axes
        for ax in ["crp", "abs_diamax_mri"]:
            s = sub.dropna(subset=[ax])
            r1 = spearmanr(s.offset_mm.abs(), s[ax]); r2 = spearmanr(s.tilt_deg.abs(), s[ax])
            L.append(f"  Spearman |offset| ~ {ax}: rho={r1.statistic:+.3f} p={r1.pvalue:.3f};  |tilt| ~ {ax}: rho={r2.statistic:+.3f} p={r2.pvalue:.3f}")
        for ax in ["retroinf"]:
            a = sub[sub[ax] == 0]; b = sub[sub[ax] == 1]
            L.append(f"  |offset| RPE- vs RPE+: {a.offset_mm.abs().median():.1f} vs {b.offset_mm.abs().median():.1f} mm, p={mannwhitneyu(a.offset_mm.abs(), b.offset_mm.abs()).pvalue:.3f}")
        L.append("")
    pd.DataFrame(out).to_csv(FA / "31_midline_positioning_tests.csv", index=False, float_format="%.5f")
    d[["case_id", "rl_dev_max_abs_mm", "maxMLS_orig", "maxMLS_centered", "maxMLS_detrended", "offset_mm", "tilt_deg", "n_slices_air", "icu_binary", "abs_positive"]].to_csv(FA / "31_midline_positioning.csv", index=False, float_format="%.3f")
    (FA / "31_midline_positioning.txt").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
