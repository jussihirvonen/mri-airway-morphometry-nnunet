#!/usr/bin/env python3
"""
Pre-specified 5-feature ICU comparison table (abs-positive cohort only).

Features were locked on physiological grounds (see airway_paper_outline.md):
  1. Airway volume        -- global caliber
  2. Min cross-sectional area -- focal stenosis
  3. Compactness          -- 3D spread / shape, abscess-paper analogue
  4. Max lateral midline shift (abs) -- mass effect
  5. Length below 50 mm²  -- length over which caliber is critically narrowed

Cohort restriction: abs_positive == 1 (n=223). Matches the established
Turku ICU model's scope. No-ICU vs ICU groups compared by Mann-Whitney U;
effect size = rank-biserial correlation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

BASE = Path(__file__).parent
MERGED = BASE / "airway_morphology_merged.csv"
OUT = BASE / "05_prespecified_icu_table.csv"

FEATURES = [
    ("volume_ml", "Airway volume (mL)", 1),
    ("min_area_mm2", "Minimum cross-sectional area (mm²)", 0),
    ("compactness", "Airway compactness", 3),
    ("rl_dev_max_abs_mm", "Maximum lateral midline shift (mm)", 1),
    ("len_below_50mm2_mm", "Length below 50 mm² (mm)", 0),
]


def fmt(x, dp):
    if dp == 0:
        return f"{x:.0f}"
    return f"{x:.{dp}f}"


def iqr_str(s, dp):
    q1, m, q3 = np.percentile(s, [25, 50, 75])
    return f"{fmt(m, dp)} ({fmt(q1, dp)}-{fmt(q3, dp)})"


def rank_biserial(x, y):
    u, _ = mannwhitneyu(x, y, alternative='two-sided')
    return 2 * u / (len(x) * len(y)) - 1


def p_str(p):
    if p < 0.001:
        return "0.000**"
    if p < 0.01:
        return f"{p:.3f}**"
    if p < 0.05:
        return f"{p:.3f}*"
    return f"{p:.3f}"


def build_table(df):
    icu_pos = df[df.icu_binary == 1]
    icu_neg = df[df.icu_binary == 0]
    n_pos = len(icu_pos)
    n_neg = len(icu_neg)
    rows = []
    for col, label, dp in FEATURES:
        a = icu_neg[col].dropna()
        b = icu_pos[col].dropna()
        u, p = mannwhitneyu(b, a, alternative='two-sided')
        rb = 2 * u / (len(a) * len(b)) - 1  # ICU minus No-ICU direction
        rows.append({
            "Feature": label,
            f"No-ICU (n={n_neg})": iqr_str(a, dp),
            f"ICU (n={n_pos})": iqr_str(b, dp),
            "p": p_str(p),
            "Effect (r_rb)": f"{rb:+.3f}",
        })
    return pd.DataFrame(rows), n_neg, n_pos


def main():
    df = pd.read_csv(MERGED)

    # PRIMARY: full cohort
    full, n_neg, n_pos = build_table(df)
    full_path = OUT.parent / "05_prespecified_icu_table_full.csv"
    full.to_csv(full_path, index=False)
    print(f"PRIMARY full cohort: n={len(df)}  (No-ICU {n_neg}, ICU {n_pos})")
    print(full.to_string(index=False))

    # SENSITIVITY: abscess-positive subset
    pos_df = df[df.abs_positive == 1].copy()
    sens, sn_neg, sn_pos = build_table(pos_df)
    sens_path = OUT.parent / "05_prespecified_icu_table_abspos.csv"
    sens.to_csv(sens_path, index=False)
    # keep legacy filename pointing at the abs+ table for backward compatibility
    sens.to_csv(OUT, index=False)
    print(f"\nSENSITIVITY abs+ cohort: n={len(pos_df)}  (No-ICU {sn_neg}, ICU {sn_pos})")
    print(sens.to_string(index=False))
    print(f"\nWrote {full_path}\nWrote {sens_path}")


if __name__ == "__main__":
    main()
