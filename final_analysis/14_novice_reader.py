#!/usr/bin/env python3
"""
Third-reader (novice, medical student) airway agreement vs the reference annotator.

Companion to 13_inter_rater.py (expert vs reference). The novice drew the same 30-case
interobserver subset in two rounds:
  round1 = first pass (delivered 2026-07-02)
  round2 = revised / FINAL pass (delivered 2026-07-10)

For each round, agreement vs reference is computed the same way the paper evaluates
agreement:
  - Dice + HD95 on the RAW delivered masks (same helpers as 09/13).
  - Full morphometry per reader via analyze_case with keep_largest=True (largest
    26-connected component), IDENTICAL to the paper's expert-vs-reference table
    (13_inter_rater). The novice's round-2 masks are single-component after one
    right pyriform fossa was reconnected in the second delivery (+19-voxel bridge);
    the only residual islands are single-voxel strays in two cases that largest-CC
    drops (zero morphometric effect; ~0 effect on raw Dice/HD, well below HD95 p95).
  - ICC(2,1) absolute agreement (pingouin ICC2) on the 6 reported features.

Outputs (final_analysis/):
  14_novice_percase_round1.csv / _round2.csv
  14_novice_icc_round1.csv     / _round2.csv
  14_novice_report.txt         (round1, round2, and the learning-curve delta)
"""

import os
import json
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.ndimage import binary_erosion, distance_transform_edt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
NOVICE_R1 = BASE / _CFG.get("novice_masks_round1", "novice_masks/round1")
NOVICE_R2 = BASE / _CFG.get("novice_masks_round2", "novice_masks/round2")
REF_DIR = BASE / _CFG.get("reference_masks", "reference_masks")
EXPERT_DIR = BASE / _CFG.get("expert_masks", "expert_masks")
OUT = BASE / "final_analysis"

spec = importlib.util.spec_from_file_location("morph01", OUT / "01_run_full_morphology.py")
morph01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morph01)
analyze_case = morph01.analyze_case  # analyze_case(case_id, mask_file, keep_largest=True)

ICC_FEATURES = [
    "volume_ml",
    "sphericity",
    "compactness",
    "min_area_mm2",
    "rl_dev_max_abs_mm",
    "len_below_50mm2_mm",
]

FEATURE_LABEL = {
    "volume_ml": "airway volume (mL)",
    "sphericity": "sphericity",
    "compactness": "compactness",
    "min_area_mm2": "minimum CSA (mm2)",
    "rl_dev_max_abs_mm": "max midline shift (mm)",
    "len_below_50mm2_mm": "length <50 mm2 (mm)",
}


def dice(a, b):
    a, b = a > 0, b > 0
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return 2.0 * inter / s if s > 0 else np.nan


def surface(mask):
    return mask & ~binary_erosion(mask)


def hd95(a, b, spacing):
    a, b = a > 0, b > 0
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    sa, sb = surface(a), surface(b)
    if sa.sum() == 0 or sb.sum() == 0:
        return np.nan
    dt_to_b = distance_transform_edt(~sb, sampling=spacing)
    dt_to_a = distance_transform_edt(~sa, sampling=spacing)
    d_ab = dt_to_b[sa]
    d_ba = dt_to_a[sb]
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def reference_map():
    m = {}
    for f in REF_DIR.glob("*a.nii.gz"):
        m[int(f.name.replace("a.nii.gz", ""))] = f
    return m


def novice_map(d):
    m = {}
    for f in d.glob("*.nii.gz"):
        m[int(f.name.replace(".nii.gz", ""))] = f
    return m


def agreement(novice_dir, ref_map, ref_name):
    a_map = novice_map(novice_dir)
    nums = sorted(set(a_map) & set(ref_map))
    rows = []
    for n in nums:
        cid = f"case{n:03d}"
        af, rf = a_map[n], ref_map[n]
        a_nii, r_nii = nib.load(str(af)), nib.load(str(rf))
        a_arr, r_arr = a_nii.get_fdata(), r_nii.get_fdata()
        if a_arr.shape != r_arr.shape:
            print(f"  SKIP {cid}: shape mismatch {a_arr.shape} vs {r_arr.shape}")
            continue
        spacing = tuple(float(z) for z in a_nii.header.get_zooms()[:3])
        # largest-CC morphometry (keep_largest=True), matching 13_inter_rater
        a_m = analyze_case(cid, af, keep_largest=True)
        r_m = analyze_case(cid, rf, keep_largest=True)
        if a_m is None or r_m is None:
            print(f"  SKIP {cid}: empty mask")
            continue
        row = {"case_id": cid,
               "dice": dice(a_arr, r_arr),
               "hd95_mm": hd95(a_arr, r_arr, spacing)}
        for k, v in a_m.items():
            if k != "case_id":
                row[f"novice_{k}"] = v
        for k, v in r_m.items():
            if k != "case_id":
                row[f"{ref_name}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def icc_table(frame, ref_name):
    out = []
    for feat in ICC_FEATURES:
        acol, rcol = f"novice_{feat}", f"{ref_name}_{feat}"
        if acol not in frame or rcol not in frame:
            continue
        sub = frame[["case_id", acol, rcol]].dropna()
        n = len(sub)
        if n < 3:
            out.append({"feature": feat, "ICC": np.nan, "CI_lower": np.nan,
                        "CI_upper": np.nan, "n": n})
            continue
        long = pd.concat([
            pd.DataFrame({"subject": sub["case_id"], "rater": "novice", "value": sub[acol]}),
            pd.DataFrame({"subject": sub["case_id"], "rater": ref_name.capitalize(), "value": sub[rcol]}),
        ])
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        icc = pg.intraclass_corr(data=long, targets="subject", raters="rater",
                                 ratings="value", nan_policy="omit")
        r = icc[icc["Type"] == "ICC2"].iloc[0]
        ci = r["CI95%"]
        out.append({"feature": feat, "ICC": float(r["ICC"]),
                    "CI_lower": float(ci[0]), "CI_upper": float(ci[1]), "n": int(n)})
    return out


def write_section(fp, title, frame, icc_rows):
    fp.write(title + "\n" + "-" * len(title) + "\n")
    fp.write(f"  n cases: {len(frame)}\n")
    fp.write(f"  Dice:    mean={frame['dice'].mean():.3f}  median={frame['dice'].median():.3f}  "
             f"range {frame['dice'].min():.3f}-{frame['dice'].max():.3f}\n")
    fp.write(f"  HD95mm:  median={frame['hd95_mm'].median():.2f}  "
             f"IQR {frame['hd95_mm'].quantile(0.25):.2f}-{frame['hd95_mm'].quantile(0.75):.2f}  "
             f"max {frame['hd95_mm'].max():.2f}\n")
    fp.write("  ICC(2,1) absolute agreement (pingouin ICC2, largest-CC):\n")
    fp.write(f"  {'feature':<26s} {'ICC':>6s}  {'95% CI':>18s}  {'n':>4s}\n")
    for r in icc_rows:
        ci = f"[{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]"
        fp.write(f"  {FEATURE_LABEL.get(r['feature'], r['feature']):<26s} {r['ICC']:>6.3f}  {ci:>18s}  {r['n']:>4d}\n")
    fp.write("\n")


def main():
    vmap = reference_map()

    print("=== novice round1 vs reference ===")
    df1 = agreement(NOVICE_R1, vmap, "reference")
    icc1 = icc_table(df1, "reference")
    df1.to_csv(OUT / "14_novice_percase_round1.csv", index=False)
    pd.DataFrame(icc1).to_csv(OUT / "14_novice_icc_round1.csv", index=False, float_format="%.4f")

    print("=== novice round2 (FINAL) vs reference ===")
    df2 = agreement(NOVICE_R2, vmap, "reference")
    icc2 = icc_table(df2, "reference")
    df2.to_csv(OUT / "14_novice_percase_round2.csv", index=False)
    pd.DataFrame(icc2).to_csv(OUT / "14_novice_icc_round2.csv", index=False, float_format="%.4f")

    report = OUT / "14_novice_report.txt"
    with open(report, "w") as fp:
        fp.write("Third-reader agreement: novice (medical student) vs reference annotator\n")
        fp.write("=" * 70 + "\n\n")
        fp.write(f"novice round1: {NOVICE_R1}\n")
        fp.write(f"novice round2: {NOVICE_R2}\n")
        fp.write(f"Reference   : {REF_DIR}\n")
        fp.write("Morphometry: largest-CC (keep_largest=True), matching 13_inter_rater; Dice/HD95 on raw masks.\n\n")
        write_section(fp, "ROUND 2 (FINAL) - novice vs reference", df2, icc2)
        write_section(fp, "ROUND 1 (first pass) - novice vs reference", df1, icc1)

        fp.write("LEARNING CURVE (round1 -> round2, vs reference)\n")
        fp.write("-" * 43 + "\n")
        fp.write(f"  Dice mean : {df1['dice'].mean():.3f} -> {df2['dice'].mean():.3f}  "
                 f"(delta {df2['dice'].mean()-df1['dice'].mean():+.3f})\n")
        fp.write(f"  HD95 med  : {df1['hd95_mm'].median():.2f} -> {df2['hd95_mm'].median():.2f} mm  "
                 f"(delta {df2['hd95_mm'].median()-df1['hd95_mm'].median():+.2f})\n")
        m1 = {r["feature"]: r["ICC"] for r in icc1}
        m2 = {r["feature"]: r["ICC"] for r in icc2}
        for f in ICC_FEATURES:
            fp.write(f"  ICC {FEATURE_LABEL[f]:<26s}: {m1.get(f, float('nan')):.3f} -> "
                     f"{m2.get(f, float('nan')):.3f}  (delta {m2.get(f, float('nan'))-m1.get(f, float('nan')):+.3f})\n")
        fp.write("\nBenchmark (from 13_inter_rater, n=30): expert vs reference Dice 0.875, "
                 "feature ICC 0.927-0.978.\n")

    print(f"\nSaved report: {report}\n")
    with open(report) as fp:
        print(fp.read())


if __name__ == "__main__":
    main()
