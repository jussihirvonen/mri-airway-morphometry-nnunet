#!/usr/bin/env python3
"""Origin analysis (odontogenic vs pharyngotonsillar, Supp Table 3) on
CORRECTED masks.

Why: r/results_report.txt (2026-06-16) section 7 is the manuscript's source
for the origin comparison, but it ran on r/airway_features_with_outcomes.csv
(2026-06-01), which predates the 2026-06-22 mask corrections. This script
recomputes the comparison with corrected features.

Group definitions copied from r/analysis_pipeline.R section 7:
  odontogenic       = hammas == 1 & (peritonsinf is NA | peritonsinf == 0)
  pharyngotonsillar = peritonsinf == 1 & (hammas is NA | hammas == 0)

Stats match R: wilcox.test two-sided (normal approximation with continuity
correction at these group sizes); Fisher's exact for ICU rates.

VALIDATE mode reruns on the old Jun-1 features and must reproduce
r/results_report.txt before the corrected run is trusted.

Outputs: 18_origin_recompute_report.txt
"""
import os
from pathlib import Path
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact

BASE = str(Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])) / "final_analysis")

FEATS = [
    ("volume_ml", "Airway volume (mL)"),
    ("min_area_mm2", "Min CSA (mm2)"),
    ("compactness", "Compactness"),
    ("rl_dev_max_abs_mm", "Max midline shift (mm)"),
    ("len_below_50mm2_mm", "Length <50 mm2 (mm)"),
]

clin = pd.read_csv(f"{BASE}/r/airway_features_with_outcomes.csv")
corrected = pd.read_csv(f"{BASE}/reference_full_morphology.csv")

def run(df, tag, out):
    odon = df[(df.hammas == 1) & (df.peritonsinf.isna() | (df.peritonsinf == 0))]
    phar = df[(df.peritonsinf == 1) & (df.hammas.isna() | (df.hammas == 0))]
    out.append(f"--- {tag} ---")
    out.append(f"odontogenic n={len(odon)}, pharyngotonsillar n={len(phar)}")
    for col, label in FEATS:
        a = odon[col].dropna()
        b = phar[col].dropna()
        U, p = mannwhitneyu(a, b, alternative="two-sided",
                            method="asymptotic", use_continuity=True)
        rb = abs(1 - 2 * U / (len(a) * len(b)))
        fmt = lambda s: f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})"
        out.append(f"{label:24s} odon {fmt(a):>22s}  phar {fmt(b):>22s}  p={p:.2e}  r={rb:.2f}")
    icu_o, icu_p = int(odon.icu_binary.sum()), int(phar.icu_binary.sum())
    _, fp = fisher_exact([[icu_o, len(odon) - icu_o], [icu_p, len(phar) - icu_p]])
    out.append(f"ICU: odontogenic {icu_o}/{len(odon)} ({100*icu_o/len(odon):.1f}%) "
               f"vs pharyngotonsillar {icu_p}/{len(phar)} ({100*icu_p/len(phar):.1f}%); "
               f"Fisher p={fp:.3f}")
    out.append("")

out = ["ORIGIN ANALYSIS RECOMPUTE (Supp Table 3) — 2026-08-15",
       "Reference for validation: r/results_report.txt section 7 (2026-06-16)", ""]

# 1) validation on OLD (Jun 1) features
run(clin, "VALIDATION: old Jun-1 features (must match r/results_report.txt)", out)

# 2) corrected features joined onto same clinical columns
feat_cols = [c for c, _ in FEATS]
merged = clin.drop(columns=feat_cols).merge(
    corrected[["case_id"] + feat_cols], on="case_id", validate="1:1")
assert len(merged) == 322, len(merged)
run(merged, "CORRECTED masks (2026-06-22 feature set) — CANONICAL", out)

report = "\n".join(out)
with open(f"{BASE}/18_origin_recompute_report.txt", "w") as f:
    f.write(report + "\n")
print(report)
