#!/usr/bin/env python3
"""Statistically compare novice-vs-reference vs expert-vs-reference ICC(2,1), per feature.

Both ICCs are measured on the SAME 30 cases against the SAME reference annotator, so they are
statistically DEPENDENT. Two tests:
  (1) Paired cluster bootstrap of dICC = ICC(expert,reference) - ICC(novice,reference): resample the 30
      cases with replacement, recompute BOTH ICCs on the same resample -> bootstrap distribution
      of the difference -> 95% CI + two-sided p (share of resamples on the other side of 0, x2).
      Resampling the same cases into both ICCs preserves the shared-reference dependence.
  (2) Paired Wilcoxon signed-rank on per-case |reader - reference|: is the novice's absolute
      deviation from the reference systematically larger than the expert's? (robust, interpretable)

Reads 13_/14_ per-case CSVs. Outputs 16_icc_compare_report.txt + 16_icc_compare.csv.
"""
import os
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import wilcoxon
from pathlib import Path

OUT = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])) / "final_analysis"
np.random.seed(42)
B = 10000

FEATS = [
    ("volume_ml", "Airway volume (mL)"),
    ("min_area_mm2", "Minimum CSA (mm2)"),
    ("rl_dev_max_abs_mm", "Maximum midline shift (mm)"),
    ("len_below_50mm2_mm", "Length <50 mm2 (mm)"),
    ("compactness", "Compactness"),
    ("sphericity", "Sphericity"),
]


def icc21(x, y):
    """Fast ICC(2,1) absolute agreement for two raters (n x 2). x,y arrays len n."""
    M = np.column_stack([x, y]).astype(float)
    n, k = M.shape
    grand = M.mean()
    row = M.mean(axis=1)
    col = M.mean(axis=0)
    SSR = k * np.sum((row - grand) ** 2)          # between subjects
    SSC = n * np.sum((col - grand) ** 2)          # between raters
    SST = np.sum((M - grand) ** 2)
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE)
    return (MSR - MSE) / denom if denom != 0 else np.nan


def pingouin_icc21(x, y, cases):
    long = pd.concat([
        pd.DataFrame({"s": cases, "r": "A", "v": x}),
        pd.DataFrame({"s": cases, "r": "B", "v": y}),
    ])
    icc = pg.intraclass_corr(data=long, targets="s", raters="r", ratings="v", nan_policy="omit")
    return float(icc[icc["Type"] == "ICC2"].iloc[0]["ICC"])


# --- load & merge on case_id ---
a = pd.read_csv(OUT / "14_novice_percase_round2.csv")
j = pd.read_csv(OUT / "13_inter_rater_percase.csv")
m = a.merge(j, on="case_id", suffixes=("", "_j"))
assert len(m) == 30, f"expected 30 matched cases, got {len(m)}"

# sanity: the reference annotator's values should be identical in both pipelines
vdiff = {f: float(np.abs(m[f"reference_{f}"] - m[f"reference_{f}_j"]).max()) for f, _ in FEATS}
reference_ok = all(v < 1e-6 for v in vdiff.values())

rows = []
print(f"{'feature':<24s} {'ICC_A':>7s} {'ICC_J':>7s} {'dICC(J-A)':>10s} {'95% CI':>18s} "
      f"{'p_boot':>8s} {'p_wilcox':>9s} {'medAbsDev novice/expert':>16s}")
print("-" * 108)
for key, label in FEATS:
    reference = m[f"reference_{key}"].to_numpy(float)
    novice = m[f"novice_{key}"].to_numpy(float)
    expert = m[f"expert_{key}"].to_numpy(float)

    icc_A = icc21(reference, novice)
    icc_J = icc21(reference, expert)
    # validate fast vs pingouin on the full sample
    pa = pingouin_icc21(reference, novice, m["case_id"])
    pj = pingouin_icc21(reference, expert, m["case_id"])
    assert abs(icc_A - pa) < 1e-6 and abs(icc_J - pj) < 1e-6, f"ICC mismatch {key}: {icc_A} vs {pa}"

    # (1) paired cluster bootstrap of the difference
    n = len(reference)
    diffs = np.empty(B)
    for b in range(B):
        idx = np.random.randint(0, n, n)
        diffs[b] = icc21(reference[idx], expert[idx]) - icc21(reference[idx], novice[idx])
    diffs = diffs[np.isfinite(diffs)]
    dICC = icc_J - icc_A
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_boot = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_boot = min(p_boot, 1.0)

    # (2) paired Wilcoxon on |reader - reference|
    dev_A = np.abs(novice - reference)
    dev_J = np.abs(expert - reference)
    try:
        w_p = wilcoxon(dev_A, dev_J, zero_method="wilcox").pvalue
    except ValueError:
        w_p = np.nan

    print(f"{label:<24s} {icc_A:>7.3f} {icc_J:>7.3f} {dICC:>+10.3f} "
          f"[{lo:>+6.3f},{hi:>+6.3f}] {p_boot:>8.4f} {w_p:>9.4f} "
          f"{np.median(dev_A):>7.2f}/{np.median(dev_J):<7.2f}")
    rows.append({"feature": label, "ICC_novice": round(icc_A, 3), "ICC_expert": round(icc_J, 3),
                 "dICC_JminusA": round(dICC, 3), "CI_lo": round(lo, 3), "CI_hi": round(hi, 3),
                 "p_bootstrap": round(p_boot, 4), "p_wilcoxon_absdev": round(w_p, 4),
                 "medAbsDev_novice": round(float(np.median(dev_A)), 3),
                 "medAbsDev_expert": round(float(np.median(dev_J)), 3)})

df = pd.DataFrame(rows)
df.to_csv(OUT / "16_icc_compare.csv", index=False)

# Holm-Bonferroni note over 6 features
def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    for rank, i in enumerate(order):
        adj[i] = min((len(pvals) - rank) * pvals[i], 1.0)
    return np.maximum.accumulate(adj[order])[np.argsort(order)] if False else adj

with open(OUT / "16_icc_compare_report.txt", "w") as fp:
    fp.write("Statistical comparison of ICC(2,1): novice-vs-reference vs expert-vs-reference (n=30, paired)\n")
    fp.write("=" * 82 + "\n\n")
    fp.write(f"Bootstrap resamples B={B}, seed=42. reference values identical across pipelines: {reference_ok}\n")
    fp.write("dICC = ICC(expert,reference) - ICC(novice,reference); positive => expert agrees better.\n")
    fp.write("p_boot = two-sided paired cluster-bootstrap p for dICC != 0.\n")
    fp.write("p_wilcoxon = paired Wilcoxon signed-rank on per-case |reader - reference|.\n\n")
    fp.write(df.to_string(index=False))
    fp.write("\n\nBonferroni across 6 features: alpha = 0.05/6 = 0.0083.\n")
print(f"\nSaved: 16_icc_compare.csv, 16_icc_compare_report.txt  (reference identical across pipelines: {reference_ok})")
