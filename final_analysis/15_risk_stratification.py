#!/usr/bin/env python3
"""
Risk stratification mirroring the abscess paper's "Risk Stratification by Size
and Shape" final-paragraph analysis.

Three groups, defined a priori from the abscess paper's framework:
  1. Small abscess  (manual diameter < 40 mm, matching Heikkinen 2022)
  2. Large abscess + adequate airway compactness  (≥ median compactness
     among large abscesses)
  3. Large abscess + poor airway compactness  (< median compactness)

Outcomes: ICU admission, prolonged hospitalization.

Output:
  15_risk_strata.csv  — counts, rates, RR, Fisher p per group
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
SRC = BASE / "final_analysis" / "airway_morphology_merged.csv"
OUT = BASE / "final_analysis" / "15_risk_strata.csv"

df = pd.read_csv(SRC)
df = df.dropna(subset=["abs_diamax_mri", "compactness", "icu_binary"]).copy()
df["large"] = (df["abs_diamax_mri"] >= 40).astype(int)

small  = df[df["large"] == 0]
large  = df[df["large"] == 1]
median_comp = float(large["compactness"].median())
large_good = large[large["compactness"] >= median_comp]
large_bad  = large[large["compactness"] <  median_comp]

def block(grp, label):
    n  = len(grp)
    icu  = int(grp["icu_binary"].sum())
    los_n = int(grp["hospd_prolonged"].notna().sum())
    los  = int((grp["hospd_prolonged"] == 1).sum())
    return dict(group=label, n=n,
                icu_n=icu, icu_pct=100*icu/n if n else np.nan,
                los_n=los, los_eval=los_n, los_pct=100*los/los_n if los_n else np.nan)

rows = [block(small, "Small (<40 mm)"),
        block(large_good, "Large (≥40 mm) + good compactness"),
        block(large_bad,  "Large (≥40 mm) + poor compactness")]

# Reference: small abscess (lowest risk)
ref_icu = rows[0]["icu_pct"]
ref_los = rows[0]["los_pct"]
for r in rows:
    r["icu_RR_vs_small"] = r["icu_pct"] / ref_icu if ref_icu else np.nan
    r["los_RR_vs_small"] = r["los_pct"] / ref_los if ref_los else np.nan

# Within-large comparison (bad vs good)
_, p_icu_within = stats.fisher_exact(
    [[large_bad.icu_binary.sum(), len(large_bad) - large_bad.icu_binary.sum()],
     [large_good.icu_binary.sum(), len(large_good) - large_good.icu_binary.sum()]])
los_bad = int((large_bad["hospd_prolonged"] == 1).sum())
los_bad_n = int(large_bad["hospd_prolonged"].notna().sum())
los_good = int((large_good["hospd_prolonged"] == 1).sum())
los_good_n = int(large_good["hospd_prolonged"].notna().sum())
_, p_los_within = stats.fisher_exact(
    [[los_bad,  los_bad_n  - los_bad],
     [los_good, los_good_n - los_good]])

print(f"Compactness median among large abscesses: {median_comp:.3f}")
print(f"{'group':<40s} {'n':>4s} {'ICU n (%)':>14s} {'RR vs small':>12s} {'LOS n (%)':>14s}")
for r in rows:
    icu_str = f"{r['icu_n']} ({r['icu_pct']:.1f}%)"
    los_str = f"{r['los_n']}/{r['los_eval']} ({r['los_pct']:.1f}%)"
    print(f"  {r['group']:<38s} {r['n']:>4d} {icu_str:>14s} "
          f"{r['icu_RR_vs_small']:>12.2f} {los_str:>14s}")
print(f"\nWithin large: bad vs good compactness")
print(f"  ICU Fisher exact p = {p_icu_within:.4f}")
print(f"  Prolonged stay Fisher exact p = {p_los_within:.4f}")

# Save three-step strata
out_df = pd.DataFrame(rows)
out_df["compactness_median_among_large"] = median_comp
out_df["p_within_large_ICU"]   = p_icu_within
out_df["p_within_large_LOS"]   = p_los_within
out_df.to_csv(OUT, index=False, float_format="%.4f")
print(f"\nWrote {OUT}")


# -------------------------------------------------------------------------
# 2x2 joint stratification: abscess sphericity × airway compactness
#   (within large abscesses; abs+ with radiomic sphericity available)
# -------------------------------------------------------------------------
joint_src = pd.read_csv(SRC)
joint_src = joint_src[(joint_src.abs_positive == 1) &
                      joint_src.abs_diamax_mri.notna() &
                      joint_src.abs_sphericity_radiomics.notna() &
                      joint_src.compactness.notna() &
                      joint_src.icu_binary.notna()].copy()
joint_src["large"] = (joint_src.abs_diamax_mri >= 40).astype(int)
jl = joint_src[joint_src.large == 1].copy()
sph_med = float(jl["abs_sphericity_radiomics"].median())
air_med = float(jl["compactness"].median())

jl["abs_round"]  = jl["abs_sphericity_radiomics"] >= sph_med
jl["airway_ok"] = jl["compactness"] >= air_med

# Spearman correlation (orthogonality check)
rho_abs_plus, p_corr_abs_plus = stats.spearmanr(joint_src["abs_sphericity_radiomics"],
                                                 joint_src["compactness"])
rho_large, p_corr_large = stats.spearmanr(jl["abs_sphericity_radiomics"], jl["compactness"])

# 2x2 cells
cells = []
for round_ok in (True, False):
    for airway_ok in (True, False):
        g = jl[(jl["abs_round"] == round_ok) & (jl["airway_ok"] == airway_ok)]
        n = len(g)
        icu = int(g.icu_binary.sum())
        pct = 100 * icu / n if n else float("nan")
        cells.append({"abs_round": round_ok, "airway_ok": airway_ok,
                      "n": n, "icu_n": icu, "icu_pct": pct})
cells_df = pd.DataFrame(cells)
cells_df["sphericity_median_among_large"] = sph_med
cells_df["airway_compactness_median_among_large"] = air_med
cells_df["spearman_rho_abs_plus"] = rho_abs_plus
cells_df["spearman_p_abs_plus"]   = p_corr_abs_plus
cells_df["spearman_rho_large"]    = rho_large
cells_df["spearman_p_large"]      = p_corr_large
cells_df.to_csv(BASE / "final_analysis" / "15_joint_strata.csv",
                index=False, float_format="%.4f")

print(f"\n2x2 within large abscesses (n={len(jl)})")
print(f"  Sphericity median: {sph_med:.3f}; Airway compactness median: {air_med:.3f}")
print(f"  Spearman abs sphericity ~ airway compactness:")
print(f"    in abs+ cohort (n={len(joint_src)}): rho={rho_abs_plus:+.3f}  p={p_corr_abs_plus:.4f}")
print(f"    in large abscesses (n={len(jl)}):     rho={rho_large:+.3f}  p={p_corr_large:.4f}")
print(f"  {'Cell':<46s} {'n':>4s} {'ICU n (%)':>14s}")
for _, c in cells_df.iterrows():
    lab = ("Abs round + airway OK"   if c.abs_round and c.airway_ok else
           "Abs round + airway BAD"  if c.abs_round else
           "Abs irregular + airway OK" if c.airway_ok else
           "Abs irregular + airway BAD")
    print(f"  {lab:<46s} {int(c['n']):>4d}   {int(c['icu_n']):>2d} ({c['icu_pct']:5.1f}%)")
print(f"\nWrote {BASE / 'final_analysis' / '15_joint_strata.csv'}")
