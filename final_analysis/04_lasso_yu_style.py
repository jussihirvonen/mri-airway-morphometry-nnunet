#!/usr/bin/env python3
"""
Yu et al. 2026-style statistical pipeline applied to the DNI airway cohort.

Mirrors Yu's analytic choices:
- Z-standardization of all continuous variables.
- LASSO with 10-fold CV for predictor selection.
- Refit with OLS for interpretable Z-betas.
- Bootstrap 95% CIs (10 000 reps, stratified by abs_positive).
- VIF check.
- Heatmap figure of standardized betas across airway metrics x predictors.

Two analytic blocks:
 (A) Yu-style: predict each airway metric from clinical predictors.
 (B) Inverse: predict ICU from airway features + clinical features (LASSO logistic).
"""

import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])) / "final_analysis"
DATA = OUT_DIR / "airway_morphology_merged.csv"

SEED = 42
N_BOOT = 10_000

# ── Load ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df["crp"] = pd.to_numeric(df["crp"], errors="coerce")
df["leuk"] = pd.to_numeric(df["leuk"], errors="coerce")
df["rl_dev_max_abs"] = df["rl_dev_max_mm"].abs()

# Restrict to abs+ for predictor analyses (where abscess features are defined)
pos = df[df.abs_positive == 1].copy().reset_index(drop=True)

# ── Define metrics and predictors ───────────────────────────────────
AIRWAY_METRICS = [
    "compactness",
    "min_area_mm2",
    "ap_at_narrowest_mm",
    "hydraulic_diam_at_narrowest_mm",
    "narrowed_length_25pct_mm",
    "narrowed_length_50pct_mm",
    "rl_dev_max_abs",
    "elongation",
    "volume_ml",
    "mean_area_mm2",
    "stenosis_pct",
]

PREDICTORS = [
    ("abs_volume_radiomics_ml", "Abscess volume"),
    ("abs_max3d_radiomics", "Abscess max 3D diameter"),
    ("abs_sphericity_radiomics", "Abscess sphericity"),
    ("abs_compactness_radiomics", "Abscess compactness"),
    ("rpe_volume_ml", "RPE volume"),
    ("mediast_binary", "Mediastinal extension"),
    ("crp", "CRP"),
    ("leuk", "WBC"),
    ("age", "Age"),
    ("sex", "Sex (M=1)"),
]

LABELS = dict(PREDICTORS)
PCOLS = [p[0] for p in PREDICTORS]


# ── Helpers ─────────────────────────────────────────────────────────
def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def fit_lasso_select(X, y, cv=10, seed=SEED):
    """Standardize X & y per-column, fit LassoCV, return non-zero predictors and betas."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # PER-COLUMN standardization (fix: previously used scalar mean/std)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    x_std[x_std == 0] = 1.0  # guard
    Xz = (X - x_mean) / x_std
    y_mean = y.mean()
    y_std = y.std(ddof=0) if y.std(ddof=0) > 0 else 1.0
    yz = (y - y_mean) / y_std
    lasso = LassoCV(cv=cv, random_state=seed, max_iter=20000).fit(Xz, yz)
    return lasso, Xz, yz


def refit_ols(Xz, yz, selected_idx):
    """OLS refit on selected predictors, return betas + p-values."""
    if len(selected_idx) == 0:
        return None
    Xs = Xz[:, selected_idx]
    Xs1 = np.column_stack([np.ones(len(Xs)), Xs])
    coef, *_ = np.linalg.lstsq(Xs1, yz, rcond=None)
    yhat = Xs1 @ coef
    resid = yz - yhat
    df_resid = len(yz) - Xs1.shape[1]
    sigma2 = (resid ** 2).sum() / df_resid
    cov = sigma2 * np.linalg.inv(Xs1.T @ Xs1)
    se = np.sqrt(np.diag(cov))
    t_stat = coef / se
    p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df_resid))
    return {"coef": coef[1:], "p": p[1:], "se": se[1:]}


def boot_corr(x, y, n=N_BOOT, seed=SEED):
    """Bootstrap 95% CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    n_obs = len(x)
    rhos = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            rhos.append(r)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    rho_obs, p_obs = stats.spearmanr(x, y)
    return rho_obs, lo, hi, p_obs


def vif_check(X, names):
    rows = []
    Xn = X.dropna().values
    Xn1 = np.column_stack([np.ones(len(Xn)), Xn])
    for i, name in enumerate(names):
        try:
            v = variance_inflation_factor(Xn1, i + 1)
        except Exception:
            v = np.nan
        rows.append((name, v))
    return pd.DataFrame(rows, columns=["predictor", "VIF"])


# ── A. LASSO: predict airway metrics from clinical predictors ───────
print("="*78)
print("A. LASSO — predict airway metrics from clinical predictors (abs+ subset)")
print("="*78)

results = {}     # metric -> dict with betas, p, selected
heatmap_beta = np.zeros((len(AIRWAY_METRICS), len(PCOLS)))
heatmap_p = np.ones_like(heatmap_beta)

for i, metric in enumerate(AIRWAY_METRICS):
    cols = PCOLS + [metric]
    sub = pos[cols].dropna()
    if len(sub) < 30:
        print(f"  {metric}: skipped (n={len(sub)} after dropna)")
        continue
    X = sub[PCOLS].values
    y = sub[metric].values

    # Standardize then LASSO
    lasso, Xz, yz = fit_lasso_select(X, y)
    selected = np.where(np.abs(lasso.coef_) > 1e-8)[0]

    if len(selected) == 0:
        print(f"  {metric}: LASSO retained NO predictors (alpha={lasso.alpha_:.4f})")
        continue

    refit = refit_ols(Xz, yz, selected)
    for k, idx in enumerate(selected):
        heatmap_beta[i, idx] = refit["coef"][k]
        heatmap_p[i, idx] = refit["p"][k]

    sel_names = [LABELS[PCOLS[idx]] for idx in selected]
    sel_betas = [f"{LABELS[PCOLS[idx]]}={refit['coef'][k]:+.2f}(p={refit['p'][k]:.3f})"
                 for k, idx in enumerate(selected)]
    print(f"  {metric:32s} n={len(sub):3d}  alpha={lasso.alpha_:.4f}")
    for b in sel_betas:
        print(f"    {b}")

    results[metric] = {
        "n": len(sub), "alpha": float(lasso.alpha_),
        "selected": [PCOLS[j] for j in selected],
        "betas": refit["coef"].tolist(),
        "p_values": refit["p"].tolist(),
    }

# Save numeric results
(OUT_DIR / "04_lasso_results.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved {OUT_DIR/'04_lasso_results.json'}")


# ── A.bootstrap: 95% CIs for Spearman vs RPE volume (selected metrics) ──
print()
print("="*78)
print("A2. Bootstrap 95% CIs for Spearman correlations (RPE volume vs airway, abs+)")
print("="*78)
print(f"{'Metric':32s} {'rho':>7s} {'95% CI':>20s}  {'p':>8s}  {'n':>4s}")
bs_results = []
for metric in AIRWAY_METRICS:
    sub = pos[["rpe_volume_ml", metric]].dropna()
    if len(sub) < 20:
        continue
    rho, lo, hi, p = boot_corr(sub.rpe_volume_ml.values, sub[metric].values)
    print(f"  {metric:30s} {rho:+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {p:>8.4f}  {len(sub):>4d}")
    bs_results.append({"metric": metric, "rho": rho, "ci_lo": lo, "ci_hi": hi, "p": p, "n": len(sub)})

pd.DataFrame(bs_results).to_csv(OUT_DIR / "04_bootstrap_rpe_correlations.csv", index=False)


# ── A3. VIF check ───────────────────────────────────────────────────
print()
print("="*78)
print("A3. VIF (multicollinearity) of the 10 predictors in abs+ subset")
print("="*78)
X_all = pos[PCOLS]
vifs = vif_check(X_all, PCOLS)
vifs["label"] = vifs.predictor.map(LABELS)
print(vifs[["label", "VIF"]].round(2).to_string(index=False))
vifs.to_csv(OUT_DIR / "04_vif_predictors.csv", index=False)


# ── A4. Heatmap figure ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6.5))
# Convention: rows = airway metrics, cols = predictors
im = ax.imshow(heatmap_beta, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(PCOLS)))
ax.set_xticklabels([LABELS[p] for p in PCOLS], rotation=45, ha="right")
ax.set_yticks(range(len(AIRWAY_METRICS)))
ax.set_yticklabels(AIRWAY_METRICS)
ax.set_xlabel("Predictor")
ax.set_ylabel("Airway metric")
ax.set_title("LASSO standardized betas: airway metrics predicted by clinical/imaging features\n(abs+ subset, n~209; blank = LASSO dropped)")
# Annotate cells
for i in range(heatmap_beta.shape[0]):
    for j in range(heatmap_beta.shape[1]):
        if abs(heatmap_beta[i, j]) > 1e-8:
            sig = "**" if heatmap_p[i, j] < 0.01 else ("*" if heatmap_p[i, j] < 0.05 else "")
            ax.text(j, i, f"{heatmap_beta[i, j]:+.2f}\n{sig}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(heatmap_beta[i, j]) > 0.5 else "black")
fig.colorbar(im, ax=ax, label="Z-β")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_lasso_heatmap.png", dpi=150)
plt.savefig(OUT_DIR / "04_lasso_heatmap.pdf")
print(f"\nSaved heatmap: {OUT_DIR/'04_lasso_heatmap.png'}")


# ── B. LASSO logistic: predict ICU from airway + clinical features ──
print()
print("="*78)
print("B. LASSO LOGISTIC — predict ICU from airway + clinical + abscess (abs+)")
print("="*78)
ICU_FEATURES = [
    "compactness", "min_area_mm2", "narrowed_length_25pct_mm", "rl_dev_max_abs",
    "ap_at_narrowest_mm", "hydraulic_diam_at_narrowest_mm",
    "abs_max3d_radiomics", "abs_volume_radiomics_ml", "abs_sphericity_radiomics",
    "rpe_volume_ml", "mediast_binary",
    "crp", "leuk", "age", "sex",
]
sub = pos[ICU_FEATURES + ["icu_binary"]].dropna()
y = sub.icu_binary.values.astype(int)
X = sub[ICU_FEATURES].values
print(f"  n={len(sub)}, ICU+={int(y.sum())}")

scaler = StandardScaler()
Xz = scaler.fit_transform(X)
lr = LogisticRegressionCV(
    Cs=20, cv=10, penalty="l1", solver="saga",
    scoring="roc_auc", max_iter=50000, random_state=SEED
).fit(Xz, y)

coefs = lr.coef_[0]
selected = np.where(np.abs(coefs) > 1e-8)[0]
print(f"  Best C={lr.C_[0]:.4f}")
print(f"  Selected {len(selected)} of {len(ICU_FEATURES)} features:")
for idx in np.argsort(-np.abs(coefs)):
    if abs(coefs[idx]) < 1e-8:
        continue
    print(f"    {ICU_FEATURES[idx]:32s} Z-beta={coefs[idx]:+.3f}")

icu_lasso = {"C": float(lr.C_[0]),
             "selected_features": [ICU_FEATURES[i] for i in selected],
             "coefs": coefs[selected].tolist(),
             "all_features": ICU_FEATURES,
             "all_coefs": coefs.tolist(),
             "n": int(len(sub)),
             "icu_positive": int(y.sum())}
(OUT_DIR / "04_lasso_icu_logistic.json").write_text(json.dumps(icu_lasso, indent=2))
print(f"  Saved {OUT_DIR/'04_lasso_icu_logistic.json'}")

print()
print("All Yu-style analyses complete.")
