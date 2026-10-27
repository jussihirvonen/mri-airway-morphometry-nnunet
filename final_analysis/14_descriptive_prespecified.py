#!/usr/bin/env python3
"""
Descriptive statistics for the features cited in Paper 1's "Descriptive Airway
Morphometry" paragraph. The report from 03_analyze.py covers most features but
omits the two derived ones (rl_dev_max_abs_mm, len_below_50mm2_mm), so this
script writes them all into a single CSV that the manuscript can reference.

Output:
  14_descriptive_prespecified.csv  (feature, n, mean, sd, median, q25, q75, min, max)
"""
import os
from pathlib import Path
import pandas as pd

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
SRC = BASE / "final_analysis" / "airway_morphology_merged.csv"
OUT = BASE / "final_analysis" / "14_descriptive_prespecified.csv"

# Features cited in the manuscript's descriptive paragraph.
FEATURES = [
    "volume_ml",
    "min_area_mm2",
    "mean_area_mm2",
    "cc_length_mm",
    "compactness",
    "rl_dev_max_abs_mm",
    "len_below_50mm2_mm",
]

df = pd.read_csv(SRC)
print(f"loaded {len(df)} cases from {SRC.name}")

rows = []
for f in FEATURES:
    if f not in df.columns:
        print(f"  MISSING column: {f}")
        continue
    v = df[f].dropna()
    rows.append({
        "feature": f,
        "n": len(v),
        "mean": v.mean(),
        "sd": v.std(),
        "median": v.median(),
        "q25": v.quantile(0.25),
        "q75": v.quantile(0.75),
        "min": v.min(),
        "max": v.max(),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False, float_format="%.4f")
print(f"\nSaved: {OUT}")
print(out.round(2).to_string(index=False))
