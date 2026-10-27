#!/usr/bin/env python3
"""
Segmentation agreement for Paper 2 (automated nnU-Net vs manual reference).

Inputs:
  manual reference : nnunet_setup/Dataset005_NeckAirway/labelsTr/caseXXX.nii.gz (322)
  predicted (OOF)  : final_analysis/predicted_masks/caseXXX.nii.gz (322, out-of-fold)

Outputs:
  09_seg_agreement_percase.csv   per-case Dice, HD95, and the 5 pre-specified
                                 features from manual and predicted masks
  09_seg_agreement_report.txt    Dice/HD95 summary + ICC(2,1) + Bland-Altman

Morphometry is computed by importing analyze_case from 01_run_full_morphology.py
so manual and predicted masks pass through identical code.
"""

import os
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import binary_erosion, distance_transform_edt
import pingouin as pg

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
LABELS = BASE / "nnunet_setup" / "Dataset005_NeckAirway" / "labelsTr"
PRED = BASE / "final_analysis" / "predicted_masks"
OUT = BASE / "final_analysis"

PRESPEC = [
    ("volume_ml", "Airway volume (mL)"),
    ("min_area_mm2", "Minimum cross-sectional area (mm2)"),
    ("compactness", "Compactness"),
    ("rl_dev_max_abs_mm", "Maximum midline shift (mm)"),
    ("len_below_50mm2_mm", "Length below 50 mm2 (mm)"),
]

# import analyze_case from 01_run_full_morphology.py
spec = importlib.util.spec_from_file_location("morph01", OUT / "01_run_full_morphology.py")
morph01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morph01)
analyze_case = morph01.analyze_case


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


def main():
    rows = []
    pred_files = sorted(PRED.glob("case*.nii.gz"))
    print(f"predicted masks: {len(pred_files)}")
    for pf in pred_files:
        cid = pf.stem.replace(".nii", "")  # caseXXX
        lf = LABELS / pf.name
        if not lf.exists():
            print(f"  missing manual for {cid}")
            continue
        mn = nib.load(str(lf))
        pn = nib.load(str(pf))
        m = mn.get_fdata()
        p = pn.get_fdata()
        spacing = tuple(float(z) for z in mn.header.get_zooms()[:3])
        d = dice(m, p)
        h = hd95(m, p, spacing)

        man = analyze_case(cid, lf)                       # manual: corrected -> single component
        prd = analyze_case(cid, pf, keep_largest=False)   # predicted: UNION of all components (no largest-CC)
        _, pred_ncomp = ndimage.label(p > 0, structure=ndimage.generate_binary_structure(3, 3))
        row = {"case_id": cid, "dice": d, "hd95_mm": h, "pred_ncomp": pred_ncomp}
        for feat, _ in PRESPEC:
            row[f"manual_{feat}"] = man[feat] if man else np.nan
            row[f"pred_{feat}"] = prd[feat] if prd else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "09_seg_agreement_percase.csv", index=False)

    out = []
    def pr(s=""):
        print(s); out.append(str(s))

    n_disjoint = int((df["pred_ncomp"] > 1).sum())
    pr("=" * 70)
    pr(f"SEGMENTATION AGREEMENT (out-of-fold, n={len(df)})")
    pr("=" * 70)
    pr(f"Predicted masks evaluated as UNION of all connected components (no postprocessing).")
    pr(f"Disjoint (multi-component) predictions: {n_disjoint}/{len(df)}")
    pr(f"\nDice : mean={df.dice.mean():.3f}  SD={df.dice.std():.3f}  "
       f"median={df.dice.median():.3f}  IQR={df.dice.quantile(.25):.3f}-{df.dice.quantile(.75):.3f}  "
       f"min={df.dice.min():.3f}")
    hd = df.hd95_mm.dropna()
    pr(f"HD95 (mm): mean={hd.mean():.2f}  SD={hd.std():.2f}  "
       f"median={hd.median():.2f}  IQR={hd.quantile(.25):.2f}-{hd.quantile(.75):.2f}  "
       f"max={hd.max():.2f}  (n={len(hd)})")

    pr("\n" + "-" * 70)
    pr("MORPHOMETRIC AGREEMENT: manual vs predicted (5 pre-specified features)")
    pr("-" * 70)
    pr(f"{'feature':34s} {'ICC(2,1)':>10s} {'95% CI':>18s} {'bias':>9s} {'LoA':>20s}")
    for feat, label in PRESPEC:
        sub = df[[f"manual_{feat}", f"pred_{feat}"]].dropna()
        n = len(sub)
        long = pd.DataFrame({
            "target": list(range(n)) * 2,
            "rater": ["manual"] * n + ["pred"] * n,
            "score": list(sub[f"manual_{feat}"]) + list(sub[f"pred_{feat}"]),
        })
        icc = pg.intraclass_corr(data=long, targets="target", raters="rater", ratings="score")
        r21 = icc[icc.Type == "ICC2"].iloc[0]
        diff = sub[f"pred_{feat}"] - sub[f"manual_{feat}"]
        bias = diff.mean()
        loa_lo = bias - 1.96 * diff.std()
        loa_hi = bias + 1.96 * diff.std()
        ci = r21["CI95%"]
        pr(f"{label:34s} {r21['ICC']:>10.3f} [{ci[0]:>6.3f},{ci[1]:>6.3f}] "
           f"{bias:>9.2f} [{loa_lo:>8.2f},{loa_hi:>8.2f}]")

    (OUT / "09_seg_agreement_report.txt").write_text("\n".join(out))
    print(f"\nSaved {OUT/'09_seg_agreement_percase.csv'}")
    print(f"Saved {OUT/'09_seg_agreement_report.txt'}")


if __name__ == "__main__":
    main()
