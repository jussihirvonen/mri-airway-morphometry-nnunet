#!/usr/bin/env python3
"""Side-by-side interobserver table: novice (third reader, round-2 final) vs reference,
next to the paper's primary expert vs reference benchmark. All values read from the 13_/14_ CSVs.
Outputs 15_reader_benchmark_table.md and .csv."""
import os
import pandas as pd
from pathlib import Path

OUT = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])) / "final_analysis"

FEATS = [
    ("volume_ml", "Airway volume (mL)"),
    ("min_area_mm2", "Minimum cross-sectional area (mm2)"),
    ("rl_dev_max_abs_mm", "Maximum midline shift (mm)"),
    ("len_below_50mm2_mm", "Length below 50 mm2 (mm)"),
    ("compactness", "Compactness"),
    ("sphericity", "Sphericity"),
]

def load(icc_csv, percase_csv):
    icc = pd.read_csv(icc_csv).set_index("feature")
    pc = pd.read_csv(percase_csv)
    return icc, pc

a_icc, a_pc = load(OUT / "14_novice_icc_round2.csv", OUT / "14_novice_percase_round2.csv")
j_icc, j_pc = load(OUT / "13_inter_rater_icc.csv", OUT / "13_inter_rater_percase.csv")

def dice_str(pc):
    return f"{pc['dice'].mean():.3f} ({pc['dice'].min():.3f}–{pc['dice'].max():.3f})"

def hd_str(pc):
    return f"{pc['hd95_mm'].median():.2f} ({pc['hd95_mm'].quantile(.25):.2f}–{pc['hd95_mm'].quantile(.75):.2f})"

def icc_str(icc, feat):
    r = icc.loc[feat]
    return f"{r['ICC']:.3f} ({r['CI_lower']:.3f}–{r['CI_upper']:.3f})"

A = "novice vs reference (third reader)"
J = "expert vs reference (primary)"

rows = []
rows.append(("Cases, n", str(len(a_pc)), str(len(j_pc))))
rows.append(("Dice, mean (range)", dice_str(a_pc), dice_str(j_pc)))
rows.append(("HD95 mm, median (IQR)", hd_str(a_pc), hd_str(j_pc)))
rows.append(("ICC(2,1), estimate (95% CI):", "", ""))
for key, label in FEATS:
    rows.append((" " + label, icc_str(a_icc, key), icc_str(j_icc, key)))

# --- markdown ---
w0 = max(len(r[0]) for r in rows)
md = []
md.append("### Interobserver agreement: novice third reader vs expert benchmark")
md.append("")
md.append(f"| {'Metric':<{w0}} | {A} | {J} |")
md.append(f"|{'-'*(w0+2)}|{'-'*(len(A)+2)}|{'-'*(len(J)+2)}|")
for m, a, j in rows:
    md.append(f"| {m:<{w0}} | {a} | {j} |")
md.append("")
md.append("ICC(2,1) = two-way random-effects, absolute-agreement, single-rater intraclass "
          "correlation (pingouin). Dice and HD95 computed on raw masks; morphometric features "
          "on the largest connected component. Both comparisons use the same 30-case subset "
          "and identical pipeline (13_inter_rater.py / 14_novice_reader.py).")
md_text = "\n".join(md)
(OUT / "15_reader_benchmark_table.md").write_text(md_text + "\n")

# --- csv ---
pd.DataFrame(rows, columns=["Metric", A, J]).to_csv(OUT / "15_reader_benchmark_table.csv", index=False)

print(md_text)
print(f"\nSaved: {OUT/'15_reader_benchmark_table.md'}\n       {OUT/'15_reader_benchmark_table.csv'}")
