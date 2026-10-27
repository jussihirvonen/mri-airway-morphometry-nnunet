#!/usr/bin/env python3
"""
Inter-rater agreement: expert reader vs reference annotator on dual-drawn airway segmentations.

For each case drawn by both readers:
  - Full 70-feature morphometry per reader (via analyze_case from 01_run_full_morphology.py)
  - Dice + HD95 between masks (same helpers as 09_seg_agreement.py)

Outputs:
  - 13_inter_rater_percase.csv  (per-case values for both readers + dice + hd95)
  - 13_inter_rater_icc.csv      (ICC(2,1) absolute agreement, 95% CI, n)
  - 13_inter_rater_report.txt   (human-readable summary)

ICC is computed with pingouin (ICC2 row = ICC(2,1) absolute agreement, proper CI).
Mirrors the abscess design (volume, sphericity, compactness) and adds the
5 prespecified airway features.
"""

import os
import json
import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.ndimage import binary_erosion, distance_transform_edt

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
EXPERT_DIR = BASE / _CFG.get("expert_masks", "expert_masks")
REF_DIR = BASE / _CFG.get("reference_masks", "reference_masks")
OUT = BASE / "final_analysis"

# Load analyze_case from 01_run_full_morphology.py (numeric module name workaround)
spec = importlib.util.spec_from_file_location("morph01", OUT / "01_run_full_morphology.py")
morph01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morph01)
analyze_case = morph01.analyze_case

# Features for ICC table (abscess parity + airway prespecified)
ICC_FEATURES = [
    "volume_ml",           # abscess + airway prespecified
    "sphericity",          # abscess parity
    "compactness",         # abscess + airway prespecified
    "min_area_mm2",        # airway prespecified
    "rl_dev_max_abs_mm",   # airway prespecified
    "len_below_50mm2_mm",  # airway prespecified
]

# Publication subset matching the original n=100-era interobserver table.
# Used to keep the manuscript's reported n=20 consistent with the prior draft.
PUB_SUBSET_20 = [2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20, 23, 28, 30, 62, 71, 82]


def dice(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return 2.0 * inter / s if s > 0 else np.nan


def surface(mask):
    return mask & ~binary_erosion(mask)


def hd95(a, b, spacing):
    a = a > 0
    b = b > 0
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    sa = surface(a)
    sb = surface(b)
    if sa.sum() == 0 or sb.sum() == 0:
        return np.nan
    dt_to_b = distance_transform_edt(~sb, sampling=spacing)
    dt_to_a = distance_transform_edt(~sa, sampling=spacing)
    d_ab = dt_to_b[sa]
    d_ba = dt_to_a[sb]
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def find_dual_cases():
    expert = {}
    for f in EXPERT_DIR.glob("case*airway.nii.gz"):
        num = int(f.name.replace("case", "").replace("airway.nii.gz", ""))
        expert[num] = f
    reference = {}
    for f in REF_DIR.glob("*a.nii.gz"):
        num = int(f.name.replace("a.nii.gz", ""))
        reference[num] = f
    overlap = sorted(set(expert) & set(reference))
    return [(f"case{n:03d}", expert[n], reference[n]) for n in overlap]


def main():
    cases = find_dual_cases()
    print(f"Dual-drawn cases: {len(cases)}")

    rows = []
    for cid, jf, vf in cases:
        j_nii = nib.load(str(jf))
        v_nii = nib.load(str(vf))
        j_arr = j_nii.get_fdata()
        v_arr = v_nii.get_fdata()
        if j_arr.shape != v_arr.shape:
            print(f"  SKIP {cid}: shape mismatch J{j_arr.shape} vs V{v_arr.shape}")
            continue
        spacing = tuple(float(z) for z in j_nii.header.get_zooms()[:3])

        j_morph = analyze_case(cid, jf)
        v_morph = analyze_case(cid, vf)
        if j_morph is None or v_morph is None:
            print(f"  SKIP {cid}: empty mask")
            continue

        d = dice(j_arr, v_arr)
        h = hd95(j_arr, v_arr, spacing)

        row = {"case_id": cid, "dice": d, "hd95_mm": h}
        for k, v in j_morph.items():
            if k == "case_id":
                continue
            row[f"expert_{k}"] = v
        for k, v in v_morph.items():
            if k == "case_id":
                continue
            row[f"reference_{k}"] = v
        rows.append(row)
        print(f"  {cid}: Dice={d:.3f}, HD95={h:.2f} mm, "
              f"Vol J={j_morph['volume_ml']:.1f}/V={v_morph['volume_ml']:.1f} mL")

    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    percase_csv = OUT / "13_inter_rater_percase.csv"
    df.to_csv(percase_csv, index=False)
    print(f"\nSaved: {percase_csv}  ({len(df)} cases, full available)")

    pub_ids = {f"case{n:03d}" for n in PUB_SUBSET_20}
    df_pub = df[df["case_id"].isin(pub_ids)].reset_index(drop=True)
    percase_pub_csv = OUT / "13_inter_rater_percase_n20.csv"
    df_pub.to_csv(percase_pub_csv, index=False)
    print(f"Saved: {percase_pub_csv}  ({len(df_pub)} cases, publication subset)")

    def _icc_table(frame):
        out = []
        for feat in ICC_FEATURES:
            jcol = f"expert_{feat}"
            vcol = f"reference_{feat}"
            if jcol not in frame.columns or vcol not in frame.columns:
                continue
            sub = frame[["case_id", jcol, vcol]].dropna()
            n = len(sub)
            if n < 3:
                out.append({"feature": feat, "ICC": np.nan, "CI_lower": np.nan,
                            "CI_upper": np.nan, "n": n})
                continue
            long = pd.concat([
                pd.DataFrame({"subject": sub["case_id"], "rater": "expert", "value": sub[jcol]}),
                pd.DataFrame({"subject": sub["case_id"], "rater": "reference", "value": sub[vcol]}),
            ])
            long["value"] = pd.to_numeric(long["value"], errors="coerce")
            icc = pg.intraclass_corr(data=long, targets="subject", raters="rater",
                                     ratings="value", nan_policy="omit")
            row_icc2 = icc[icc["Type"] == "ICC2"].iloc[0]
            ci = row_icc2["CI95%"]
            out.append({
                "feature": feat,
                "ICC": float(row_icc2["ICC"]),
                "CI_lower": float(ci[0]),
                "CI_upper": float(ci[1]),
                "n": int(n),
            })
        return out

    icc_full = _icc_table(df)
    icc_pub = _icc_table(df_pub)

    pd.DataFrame(icc_full).to_csv(OUT / "13_inter_rater_icc.csv", index=False, float_format="%.4f")
    pd.DataFrame(icc_pub).to_csv(OUT / "13_inter_rater_icc_n20.csv", index=False, float_format="%.4f")
    print(f"Saved: 13_inter_rater_icc.csv (n={len(df)}) and 13_inter_rater_icc_n20.csv (n={len(df_pub)})")

    def _section(title, frame, icc_rows, fp):
        fp.write(title + "\n" + "-" * len(title) + "\n")
        fp.write(f"  n cases: {len(frame)}\n")
        fp.write(f"  Dice:    mean={frame['dice'].mean():.3f}  median={frame['dice'].median():.3f}  "
                 f"range {frame['dice'].min():.3f}–{frame['dice'].max():.3f}\n")
        fp.write(f"  HD95mm:  median={frame['hd95_mm'].median():.2f}  "
                 f"IQR {frame['hd95_mm'].quantile(0.25):.2f}–{frame['hd95_mm'].quantile(0.75):.2f}  "
                 f"max {frame['hd95_mm'].max():.2f}\n")
        fp.write(f"  ICC(2,1) absolute agreement (pingouin):\n")
        fp.write(f"  {'feature':<24s} {'ICC':>6s}  {'95% CI':>16s}  {'n':>4s}\n")
        for r in icc_rows:
            ci_str = f"[{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]"
            fp.write(f"  {r['feature']:<24s} {r['ICC']:>6.3f}  {ci_str:>16s}  {r['n']:>4d}\n")
        fp.write("\n")

    report = OUT / "13_inter_rater_report.txt"
    with open(report, "w") as fp:
        fp.write("Inter-rater agreement: expert reader vs reference annotator (airway segmentation)\n")
        fp.write("=" * 65 + "\n\n")
        fp.write(f"Source — expert reader: {EXPERT_DIR}\n")
        fp.write(f"Source — reference annotator: {REF_DIR}\n\n")
        _section("PUBLICATION (n=20, matches prior airway table)", df_pub, icc_pub, fp)
        _section(f"FULL AVAILABLE (n={len(df)})", df, icc_full, fp)
    print(f"Saved: {report}")

    print("\nSummary:")
    print(f"  n=20 (publication subset):")
    print(f"    Dice mean {df_pub['dice'].mean():.3f}, HD95 median {df_pub['hd95_mm'].median():.2f}")
    for r in icc_pub:
        print(f"    ICC {r['feature']:<24s}  {r['ICC']:.3f} [{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]")
    print(f"  n=24 (full available):")
    print(f"    Dice mean {df['dice'].mean():.3f}, HD95 median {df['hd95_mm'].median():.2f}")
    for r in icc_full:
        print(f"    ICC {r['feature']:<24s}  {r['ICC']:.3f} [{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]")


if __name__ == "__main__":
    main()
