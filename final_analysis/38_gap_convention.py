#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 8): how empty slices at
complete obstructions enter the caliber features.

Manual protocol: a complete obstruction is bridged with a pinpoint segment
(area of 1-2 voxels), so minimum CSA is ~0 and the bridged slices count as
"below 50 mm2" in LCSA50. Predictions: the network leaves the slice empty. In
the reported pipeline (01_run_full_morphology.analyze_case) empty slices are
skipped, so for predictions (i) minimum CSA is the smallest PATENT slice and
(ii) the longest run below 50 mm2 jumps across the gap without counting it.
Here the alternative "gap-as-zero" convention is computed for the predictions:
every slice between the first and last predicted slice counts, with empty
slices as area 0 (so a gap sets minimum CSA to 0 and adds to LCSA50). Agreement
with the manual features (ICC(2,1)) and ICU associations are compared between
the reported (gap-skipped) and gap-as-zero conventions, n = 322.
Output: 38_gap_convention.csv / .txt
"""
import os, re
import json
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import pingouin as pg

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
FA = BASE / "final_analysis"; PRED = FA / "predicted_masks"; REF_MASKS = BASE / _CFG.get("reference_masks", "reference_masks")


def longest_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0; best = max(best, cur)
    return best


def feats(mask_file, gap_as_zero):
    nii = nib.load(str(mask_file)); m = nii.get_fdata() > 0
    vx, vy, vz = [float(z) for z in nii.header.get_zooms()]
    counts = m.sum(axis=(0, 1)); zs = np.where(counts > 0)[0]
    if gap_as_zero:
        areas = counts[zs.min():zs.max() + 1] * vx * vy
    else:
        areas = counts[zs] * vx * vy
    return dict(min_area=float(areas.min()), len50=float(longest_run(areas < 50) * vz), n_gap=int((counts[zs.min():zs.max() + 1] == 0).sum()),
                n_pinpoint=int(((counts > 0) & (counts <= 2)).sum()))


pc = pd.read_csv(FA / "09_seg_agreement_percase.csv")
mrg = pd.read_csv(FA / "airway_morphology_merged.csv"); mrg["case_id"] = mrg.case_id.astype(str)
rows = []
for cid in pc.case_id:
    num = int(cid.replace("case", ""))
    fm = feats(REF_MASKS / f"{num:04d}a.nii.gz", gap_as_zero=True)      # manual: no gaps, pinpoints counted
    fp_skip = feats(PRED / f"{cid}.nii.gz", gap_as_zero=False); fp_zero = feats(PRED / f"{cid}.nii.gz", gap_as_zero=True)
    rows.append(dict(case_id=cid, man_min=fm["min_area"], man_len50=fm["len50"], man_pinpoint=fm["n_pinpoint"],
                     pred_min_skip=fp_skip["min_area"], pred_len50_skip=fp_skip["len50"], pred_min_zero=fp_zero["min_area"], pred_len50_zero=fp_zero["len50"], pred_gap=fp_zero["n_gap"]))
d = pd.DataFrame(rows).merge(pc[["case_id", "pred_min_area_mm2", "pred_len_below_50mm2_mm", "manual_min_area_mm2", "manual_len_below_50mm2_mm"]], on="case_id").merge(mrg[["case_id", "icu_binary"]], on="case_id")
L = [f"n = {len(d)}"]
L.append(f"consistency vs 09 (reported, gap-skipped): max|diff| minCSA {np.abs(d.pred_min_skip - d.pred_min_area_mm2).max():.2f} mm2, LCSA50 {np.abs(d.pred_len50_skip - d.pred_len_below_50mm2_mm).max():.2f} mm; manual minCSA {np.abs(d.man_min - d.manual_min_area_mm2).max():.2f}, LCSA50 {np.abs(d.man_len50 - d.manual_len_below_50mm2_mm).max():.2f}")
L.append(f"manual masks with a pinpoint (<=2 voxel) slice: {(d.man_pinpoint>0).sum()} cases, {int(d.man_pinpoint.sum())} slices; manual minCSA in those cases median {d[d.man_pinpoint>0].man_min.median():.2f} mm2")
L.append(f"predictions with an interior empty slice: {(d.pred_gap>0).sum()} cases; gap length median {d[d.pred_gap>0].pred_gap.median():.0f} slices, max {d.pred_gap.max()}")
L.append(f"  of these, manual mask has a pinpoint slice in {((d.pred_gap>0)&(d.man_pinpoint>0)).sum()} cases (gap coincides with a bridged obstruction)")
L.append("")


def icc(a, b):
    df = pd.DataFrame({"case": list(range(len(a))) * 2, "rater": ["m"] * len(a) + ["p"] * len(a), "v": list(a) + list(b)})
    r = pg.intraclass_corr(data=df, targets="case", raters="rater", ratings="v"); r = r[r.Type == "ICC2"].iloc[0]
    return r.ICC, r["CI95%"]


L.append("ICC(2,1) manual vs predicted (n=322):")
res = []
for lab, col in [("minimum CSA, gap-skipped (reported)", "pred_min_skip"), ("minimum CSA, gap-as-zero", "pred_min_zero"), ("LCSA50, gap-skipped (reported)", "pred_len50_skip"), ("LCSA50, gap-as-zero", "pred_len50_zero")]:
    man = d.man_min if "CSA" in lab and "LCSA" not in lab else d.man_len50
    v, ci = icc(man, d[col]); L.append(f"  {lab:<38s} ICC {v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    y = d.icu_binary.astype(int).values; x = d[col].values; a, b = x[y == 0], x[y == 1]
    u, p = mannwhitneyu(b, a, alternative="two-sided"); rb = 2 * u / (len(a) * len(b)) - 1; auc = roc_auc_score(y, x); auc = max(auc, 1 - auc)
    L.append(f"  {'':<38s} ICU association: p={p:.4f} |r|={abs(rb):.2f} AUC={auc:.3f}")
    res.append(dict(feature=lab, ICC=v, CI_lo=ci[0], CI_hi=ci[1], p_ICU=p, r_abs=abs(rb), AUC=auc))
sub = d[d.pred_gap > 0]
L.append(f"\nIn the {len(sub)} gap cases: median predicted minCSA gap-skipped {sub.pred_min_skip.median():.1f} vs manual {sub.man_min.median():.1f} mm2; LCSA50 gap-skipped {sub.pred_len50_skip.median():.1f} vs gap-as-zero {sub.pred_len50_zero.median():.1f} vs manual {sub.man_len50.median():.1f} mm")
pd.DataFrame(res).to_csv(FA / "38_gap_convention.csv", index=False, float_format="%.4f"); d.to_csv(FA / "38_gap_convention_percase.csv", index=False, float_format="%.2f")
(FA / "38_gap_convention.txt").write_text("\n".join(L)); print("\n".join(L))


# ---------------------------------------------------------------------------
# Part 2: all five pre-specified features under the three conventions for the
# predictions (largest component / union as reported / union with gap-as-zero
# for the two caliber features), ICC(2,1) against the corrected manual masks.
# Replaces the pre-correction numbers of 27_pred_postproc_report.txt.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(FA))
import importlib
mod = importlib.import_module("01_run_full_morphology")
man = pd.read_csv(FA / "reference_full_morphology.csv"); man["case_id"] = man.case_id.astype(str)
FEAT = [("volume_ml", "Airway volume"), ("min_area_mm2", "Minimum CSA"), ("compactness", "Compactness"), ("rl_dev_max_abs_mm", "Maximum midline shift"), ("len_below_50mm2_mm", "LCSA50")]
recs = []
for cid in pc.case_id:
    lcc = mod.analyze_case(cid, PRED / f"{cid}.nii.gz", keep_largest=True); uni = mod.analyze_case(cid, PRED / f"{cid}.nii.gz", keep_largest=False)
    recs.append(dict(case_id=cid, **{f"lcc_{k}": lcc[k] for k, _ in FEAT}, **{f"uni_{k}": uni[k] for k, _ in FEAT}))
R = pd.DataFrame(recs).merge(man[["case_id"] + [k for k, _ in FEAT]], on="case_id").merge(d[["case_id", "pred_min_zero", "pred_len50_zero"]], on="case_id")
rows2 = []; L2 = ["", "Part 2: ICC(2,1) manual (corrected) vs predicted, three conventions, n=%d" % len(R)]
for k, lab in FEAT:
    v1, c1 = icc(R[k], R[f"lcc_{k}"]); v2, c2 = icc(R[k], R[f"uni_{k}"])
    gz = {"min_area_mm2": "pred_min_zero", "len_below_50mm2_mm": "pred_len50_zero"}.get(k)
    v3, c3 = icc(R[k], R[gz]) if gz else (np.nan, (np.nan, np.nan))
    rows2.append(dict(feature=lab, ICC_largestCC=v1, lo1=c1[0], hi1=c1[1], ICC_union=v2, lo2=c2[0], hi2=c2[1], ICC_gapzero=v3, lo3=c3[0], hi3=c3[1]))
    L2.append(f"  {lab:<24s} largest-CC {v1:.3f} [{c1[0]:.2f},{c1[1]:.2f}]   union {v2:.3f} [{c2[0]:.2f},{c2[1]:.2f}]   gap-as-zero {v3:.3f}")
pd.DataFrame(rows2).to_csv(FA / "38_conventions_icc.csv", index=False, float_format="%.4f")
with open(FA / "38_gap_convention.txt", "a") as fh: fh.write("\n".join(L2))
print("\n".join(L2))
