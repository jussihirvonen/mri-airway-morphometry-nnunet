#!/usr/bin/env python3
"""Table 5 provenance: reproduction of ICU associations from manual vs
DL-predicted (union-postprocessed) airway features.

Sources:
  09_seg_agreement_percase.csv  (manual_* and pred_* features, union preds,
                                 corrected manual masks, n=322)
  airway_morphology_merged.csv   (icu_binary)

Written 2026-08-13 to close the provenance gap for manuscript Table 5:
these values previously existed only in the manuscript (computed on the fly
during the 2026-06-22 redline session). 16_dl_replicates_* files are the
STALE pre-correction largest-CC versions and must not be used for Table 5.

Outputs: 17_table5_reproduction.csv, 17_table5_reproduction_report.txt
"""
import os
from pathlib import Path
import pandas as pd
from scipy.stats import mannwhitneyu

BASE = str(Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])) / "final_analysis")

per = pd.read_csv(f"{BASE}/09_seg_agreement_percase.csv")
icu = pd.read_csv(f"{BASE}/airway_morphology_merged.csv")[["case_id", "icu_binary"]]
df = per.merge(icu, on="case_id")
assert len(df) == 322 and int(df.icu_binary.sum()) == 33, "cohort mismatch"

FEATS = [
    ("min_area_mm2", "Minimum cross-sectional area (minCSA)"),
    ("compactness", "Airway compactness"),
    ("rl_dev_max_abs_mm", "Maximum absolute midline shift (maxMLS)"),
    ("len_below_50mm2_mm", "Length of airway below 50 mm2 (LCSA50)"),
]

rows = []
for src, label in [("manual", "manual"), ("pred", "predicted")]:
    for col, name in FEATS:
        x = df[f"{src}_{col}"]
        icu_v = x[df.icu_binary == 1]
        non_v = x[df.icu_binary == 0]
        U, p = mannwhitneyu(icu_v, non_v, alternative="two-sided")
        auc = U / (len(icu_v) * len(non_v))
        r = abs(2 * auc - 1)  # |rank-biserial|
        rows.append({
            "feature": name, "source": label, "n": len(df),
            "p": round(p, 4), "r_rb_abs": round(r, 2),
            "AUC": round(max(auc, 1 - auc), 3),
        })

out = pd.DataFrame(rows)
out.to_csv(f"{BASE}/17_table5_reproduction.csv", index=False)

with open(f"{BASE}/17_table5_reproduction_report.txt", "w") as f:
    f.write("TABLE 5 SOURCE — ICU associations, manual vs DL-predicted (union) features\n")
    f.write("n=322, ICU=33; Mann-Whitney U (two-sided), |rank-biserial| r, univariate AUC\n")
    f.write("Manual masks: corrected 2026-06-22 set. Predicted: union of all components.\n\n")
    f.write(out.to_string(index=False))
    f.write("\n")

print(out.to_string(index=False))
