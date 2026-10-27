#!/usr/bin/env python3
"""
Revision analysis (EJR-D-26-04240, reviewer point 7): robustness of the
post-hoc compactness split within large abscesses (>=40 mm, n=68).

(1) Selection disclosure: the within-large median split was tested for all
    five pre-specified features; report all five Fisher p values and the
    Bonferroni-adjusted p for the selected feature (x5).
(2) Continuous test (no median split): Mann-Whitney U of compactness between
    ICU and non-ICU within large abscesses; Firth-free univariate AUC.
(3) Internal validation of the median split by repeated stratified 5-fold
    cross-validation (200 repeats): the median is estimated on the training
    folds and applied to the held-out fold; held-out assignments are pooled
    and the ICU rate per stratum, relative risk and Fisher p recomputed.
(4) Bootstrap (B=2000) of the whole procedure (resample the 68 patients,
    re-estimate the median, recompute RR) -> percentile CI for RR.
(5) Prolonged stay within large abscesses by the same split (the manuscript
    called the pattern "concordant"; report the actual rates and p).

Output: 33_risk_strat_validation.txt / .csv
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
FA = BASE / "final_analysis"
df = pd.read_csv(FA / "airway_morphology_merged.csv").dropna(subset=["abs_diamax_mri", "compactness", "icu_binary"]).copy()
large = df[df.abs_diamax_mri >= 40].copy()
y = large.icu_binary.astype(int).values
FEATS = [("volume_ml", "Airway volume"), ("min_area_mm2", "Minimum CSA"), ("compactness", "Compactness"),
         ("rl_dev_max_abs_mm", "Max midline shift"), ("len_below_50mm2_mm", "LCSA50")]
L = [f"Large abscesses (>=40 mm): n={len(large)}, ICU={y.sum()} ({100*y.mean():.1f}%)", ""]

def split_stats(x, y, med, low_is_bad):
    bad = x < med if low_is_bad else x >= med
    n_bad, n_good = bad.sum(), (~bad).sum()
    icu_bad, icu_good = y[bad].sum(), y[~bad].sum()
    tab = [[icu_bad, n_bad - icu_bad], [icu_good, n_good - icu_good]]
    p = stats.fisher_exact(tab)[1]
    r_bad = icu_bad / n_bad if n_bad else np.nan; r_good = icu_good / n_good if n_good else np.nan
    rr = r_bad / r_good if r_good > 0 else np.inf
    return n_bad, icu_bad, r_bad, n_good, icu_good, r_good, rr, p

L.append("(1) Median split within large abscesses, all five pre-specified features (Fisher exact):")
rows = []
for col, lab in FEATS:
    x = large[col].values; med = np.median(x)
    low_is_bad = col in ("volume_ml", "min_area_mm2", "compactness")
    n_bad, icu_bad, r_bad, n_good, icu_good, r_good, rr, p = split_stats(x, y, med, low_is_bad)
    L.append(f"   {lab:<18s} median={med:.3f}  adverse half ICU {icu_bad}/{n_bad} ({100*r_bad:.1f}%) vs {icu_good}/{n_good} ({100*r_good:.1f}%)  RR={rr:.2f}  p={p:.4f}  p_Bonf(x5)={min(1,5*p):.3f}")
    rows.append(dict(analysis="median_split", feature=lab, median=med, icu_adverse=icu_bad, n_adverse=n_bad, icu_other=icu_good, n_other=n_good, RR=rr, p=p, p_bonf5=min(1, 5*p)))
L.append("")
L.append("(2) Continuous association within large abscesses (no split):")
for col, lab in FEATS:
    a = large[y == 0][col]; b = large[y == 1][col]
    u, p = stats.mannwhitneyu(b, a, alternative="two-sided"); rb = 2*u/(len(a)*len(b)) - 1
    auc = roc_auc_score(y, large[col].values)
    L.append(f"   {lab:<18s} median ICU {b.median():.3f} vs non-ICU {a.median():.3f}  r_rb={rb:+.3f}  AUC={max(auc,1-auc):.3f}  p={p:.4f}")
    rows.append(dict(analysis="continuous_MWU", feature=lab, p=p, r_rb=rb, auc=max(auc, 1-auc)))
L.append("")
# (3) repeated CV of the compactness split
x = large.compactness.values
rep = 200; pooled_bad = np.zeros(len(x)); counts = np.zeros(len(x))
rr_cv = []; p_cv = []
rng = np.random.RandomState(42)
for r in range(rep):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
    assign = np.zeros(len(x), dtype=bool)
    for tr, te in skf.split(x, y):
        med = np.median(x[tr]); assign[te] = x[te] < med
    n_bad = assign.sum(); icu_bad = y[assign].sum(); n_good = (~assign).sum(); icu_good = y[~assign].sum()
    rb_ = icu_bad / n_bad; rg_ = icu_good / n_good if n_good else np.nan
    rr_cv.append(rb_ / rg_ if rg_ > 0 else np.nan); p_cv.append(stats.fisher_exact([[icu_bad, n_bad-icu_bad], [icu_good, n_good-icu_good]])[1])
    pooled_bad += assign; counts += 1
rr_cv = np.array(rr_cv); p_cv = np.array(p_cv)
L.append(f"(3) Repeated stratified 5-fold CV of the compactness median split ({rep} repeats, median from training folds, held-out pooled):")
L.append(f"   RR (poor vs preserved) median {np.nanmedian(rr_cv):.2f} [IQR {np.nanpercentile(rr_cv,25):.2f}-{np.nanpercentile(rr_cv,75):.2f}], range {np.nanmin(rr_cv):.2f}-{np.nanmax(rr_cv):.2f}")
L.append(f"   Fisher p median {np.median(p_cv):.4f}; repeats with p<0.05: {100*(p_cv<0.05).mean():.0f}%")
L.append(f"   Stability of stratum assignment: {100*np.mean((pooled_bad/counts==0)|(pooled_bad/counts==1)):.0f}% of patients assigned to the same stratum in every repeat")
L.append("")
# (4) bootstrap of the whole procedure
B = 2000; rr_b = []
for b in range(B):
    idx = rng.randint(0, len(x), len(x)); xb, yb = x[idx], y[idx]
    med = np.median(xb); bad = xb < med
    if bad.sum() == 0 or (~bad).sum() == 0 or yb[~bad].sum() == 0: rr_b.append(np.nan); continue
    rr_b.append((yb[bad].mean()) / (yb[~bad].mean()))
rr_b = np.array(rr_b)
L.append(f"(4) Bootstrap of the full procedure (B={B}, median re-estimated per resample): RR 95% percentile CI {np.nanpercentile(rr_b,2.5):.2f}-{np.nanpercentile(rr_b,97.5):.2f}; {np.isnan(rr_b).sum()} resamples undefined (no ICU in preserved stratum)")
# exact CI for RR from the observed 2x2 (log method)
med = np.median(x); bad = x < med
a, n1, c, n2 = y[bad].sum(), bad.sum(), y[~bad].sum(), (~bad).sum()
rr = (a/n1)/(c/n2); se = np.sqrt(1/a - 1/n1 + 1/c - 1/n2)
L.append(f"   Observed split: RR={rr:.2f}, large-sample 95% CI {np.exp(np.log(rr)-1.96*se):.2f}-{np.exp(np.log(rr)+1.96*se):.2f}")
L.append("")
# (5) prolonged stay
ls = large.dropna(subset=["hospd_prolonged"]).copy(); xl = ls.compactness.values; yl = ls.hospd_prolonged.astype(int).values
medl = np.median(x)  # same split value as ICU analysis (0.158)
bad = xl < medl
tab = [[yl[bad].sum(), bad.sum()-yl[bad].sum()], [yl[~bad].sum(), (~bad).sum()-yl[~bad].sum()]]
L.append(f"(5) Prolonged stay (>5 d) within large abscesses by the same compactness split (median {medl:.3f}): poor {yl[bad].sum()}/{bad.sum()} ({100*yl[bad].mean():.1f}%) vs preserved {yl[~bad].sum()}/{(~bad).sum()} ({100*yl[~bad].mean():.1f}%), RR={yl[bad].mean()/yl[~bad].mean():.2f}, Fisher p={stats.fisher_exact(tab)[1]:.3f}")
L.append(f"   Table 3 (whole cohort) prolonged-stay p for compactness = 0.634 -> the within-large pattern is same-direction but not significant; manuscript wording 'concordant' must be qualified.")
pd.DataFrame(rows).to_csv(FA / "33_risk_strat_validation.csv", index=False, float_format="%.5f")
(FA / "33_risk_strat_validation.txt").write_text("\n".join(L))
print("\n".join(L))

# ---------------------------------------------------------------------------
# (6) Selection-aware repeated CV (second-opinion point 3): within each training
# fold ALL FIVE features are median-split, the feature with the smallest Fisher p
# is selected, and its training-fold median is applied to the held-out fold.
# ---------------------------------------------------------------------------
FEATC = [c for c, _ in FEATS]
rr_sa = []; p_sa = []; chosen = []
for r in range(rep):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
    assign = np.zeros(len(large), dtype=bool)
    for tr, te in skf.split(large[FEATC].values, y):
        best = None
        for col in FEATC:
            xtr = large[col].values[tr]; med = np.median(xtr); low_is_bad = col in ("volume_ml", "min_area_mm2", "compactness")
            n_bad, icu_bad, r_bad, n_good, icu_good, r_good, rr, p = split_stats(xtr, y[tr], med, low_is_bad)
            if best is None or p < best[0]: best = (p, col, med, low_is_bad)
        p_, col, med, low_is_bad = best; chosen.append(col)
        xte = large[col].values[te]; assign[te] = (xte < med) if low_is_bad else (xte >= med)
    n_bad = assign.sum(); icu_bad = y[assign].sum(); n_good = (~assign).sum(); icu_good = y[~assign].sum()
    rb_ = icu_bad / n_bad if n_bad else np.nan; rg_ = icu_good / n_good if n_good else np.nan
    rr_sa.append(rb_ / rg_ if rg_ and rg_ > 0 else np.nan); p_sa.append(stats.fisher_exact([[icu_bad, n_bad - icu_bad], [icu_good, n_good - icu_good]])[1])
rr_sa = np.array(rr_sa); p_sa = np.array(p_sa)
from collections import Counter
cnt = Counter(chosen)
L.append("")
L.append(f"(6) Selection-aware repeated 5-fold CV ({rep} repeats; in each training fold the best of the five median splits is chosen and applied to the held-out fold):")
L.append(f"   feature chosen in training folds: " + ", ".join(f"{k} {100*v/len(chosen):.0f}%" for k, v in cnt.most_common()))
L.append(f"   pooled held-out RR (adverse vs other half) median {np.nanmedian(rr_sa):.2f} [IQR {np.nanpercentile(rr_sa,25):.2f}-{np.nanpercentile(rr_sa,75):.2f}], range {np.nanmin(rr_sa):.2f}-{np.nanmax(rr_sa):.2f}; Fisher p median {np.median(p_sa):.3f}; repeats with p<0.05: {100*(p_sa<0.05).mean():.0f}%")
(FA / "33_risk_strat_validation.txt").write_text("\n".join(L)); print("\n".join(L[-4:]))
