#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 6): sensitivity of the
"length of airway below threshold" feature to the caliber threshold.

LCSA50 (len_below_50mm2_mm) is the pre-specified feature. Here the same
longest-contiguous-run length is recomputed for absolute thresholds of
25/40/50/60/75/100 mm2 and for percentile/relative thresholds (25/33/50 % of
the patient's own maximum CSA), and each variant is tested against ICU
admission (Mann-Whitney U, rank-biserial r, univariate AUC) in the full cohort
and the abscess-positive subset. Absolute 40 and 60 mm2 are recomputed from the
corrected manual masks (same slice loop as 01_run_full_morphology.analyze_case,
largest 26-connected component, no-op on the corrected single-component masks).

Output: 30_lcsa_threshold_sensitivity.csv / .txt
"""
import os, re
import json
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
FA = BASE / "final_analysis"
REF_MASKS = BASE / _CFG.get("reference_masks", "reference_masks")
MERGED = FA / "airway_morphology_merged.csv"
THRESH_ABS = [25, 40, 50, 60, 75, 100]
THRESH_REL = [0.25, 0.33, 0.50]


def longest_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def per_case_lengths(mask_file):
    nii = nib.load(str(mask_file))
    m = nii.get_fdata()
    vx, vy, vz = [float(z) for z in nii.header.get_zooms()]
    b = (m > 0).astype(int)
    lab, n = ndimage.label(b, structure=ndimage.generate_binary_structure(3, 3))
    if n > 1:
        sizes = ndimage.sum(b, lab, range(1, n + 1))
        b = (lab == (np.argmax(sizes) + 1)).astype(int)
    areas = []
    for z in range(b.shape[2]):
        nv = b[:, :, z].sum()
        if nv > 0:
            areas.append(nv * vx * vy)
    areas = np.array(areas)
    out = {}
    for t in THRESH_ABS:
        out[f"len_below_{t}mm2_mm"] = longest_run(areas < t) * vz
    mx = areas.max()
    for r in THRESH_REL:
        out[f"len_below_{int(r*100)}pct_max_mm"] = longest_run(areas <= r * mx) * vz
    return out


def main():
    df = pd.read_csv(MERGED)
    df["case_id"] = df["case_id"].astype(str)
    rows = []
    for f in sorted(REF_MASKS.glob("*a.nii.gz")):
        mobj = re.match(r"(\d+)a\.nii\.gz", f.name)
        cid = f"case{int(mobj.group(1)):03d}"
        d = per_case_lengths(f); d["case_id"] = cid
        rows.append(d)
    L = pd.DataFrame(rows)
    L = L.rename(columns={c: c + "_re" for c in L.columns if c != "case_id"})
    # Consistency check against the canonical table for the thresholds it already holds
    chk = df.merge(L, on="case_id")
    for c in ["len_below_25mm2_mm", "len_below_50mm2_mm", "len_below_75mm2_mm", "len_below_100mm2_mm",
              "len_below_25pct_max_mm", "len_below_33pct_max_mm", "len_below_50pct_max_mm"]:
        diff = (chk[c] - chk[c + "_re"]).abs().max()
        print(f"consistency {c}: max |diff| = {diff:.3f} mm")
        assert diff < 0.05, c
    # Use recomputed columns for everything (identical to canonical where overlapping)
    feats = [f"len_below_{t}mm2_mm" for t in THRESH_ABS] + [f"len_below_{int(r*100)}pct_max_mm" for r in THRESH_REL]
    out = []
    lines = ["LCSA threshold sensitivity — ICU admission (Mann-Whitney U, rank-biserial r, univariate AUC)", ""]
    for label, sub in [("Full cohort", chk), ("Abscess-positive", chk[chk.abs_positive == 1])]:
        y = sub.icu_binary.astype(int).values
        lines.append(f"{label}: n={len(sub)}, ICU={int(y.sum())}")
        lines.append(f"  {'feature':<28s} {'median noICU':>12s} {'median ICU':>10s} {'p':>10s} {'r_rb':>7s} {'AUC':>6s} {'%zero':>6s}")
        for c in feats:
            x = sub[c + "_re"].values
            a, b = x[y == 0], x[y == 1]
            u, p = mannwhitneyu(b, a, alternative="two-sided")
            rb = 2 * u / (len(a) * len(b)) - 1
            auc = roc_auc_score(y, x)
            pz = 100 * (x == 0).mean()
            lines.append(f"  {c:<28s} {np.median(a):>12.1f} {np.median(b):>10.1f} {p:>10.2e} {rb:>+7.3f} {auc:>6.3f} {pz:>6.1f}")
            out.append(dict(cohort=label, feature=c, n=len(sub), icu=int(y.sum()), med_noicu=np.median(a), med_icu=np.median(b),
                            p=p, r_rb=rb, auc=auc, pct_zero=pz))
        lines.append("")
    # Spearman correlation among threshold variants
    cm = chk[[c + "_re" for c in feats]].corr(method="spearman").round(2)
    cm.index = feats; cm.columns = feats
    lines.append("Spearman correlation between threshold variants (full cohort):")
    lines.append(cm.to_string())
    pd.DataFrame(out).to_csv(FA / "30_lcsa_threshold_sensitivity.csv", index=False, float_format="%.5f")
    (FA / "30_lcsa_threshold_sensitivity.txt").write_text("\n".join(lines))
    L.to_csv(FA / "30_lcsa_threshold_percase.csv", index=False, float_format="%.2f")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
