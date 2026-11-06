#!/usr/bin/env python3
"""
Does the DL-predicted morphometry reproduce Paper 1's ICU-association findings?

For each of the 5 pre-specified airway features, redo the Paper 1 univariate
ICU comparison (Mann-Whitney U, rank-biserial r, single-feature AUC) and the
incremental ICU model (Reference model + 4 airway features), but substitute the
DL-PREDICTED feature values for the manual ones. Compare side-by-side.

Output:
  16_dl_replicates_paper1.csv     side-by-side univariate ICU results
  16_dl_replicates_incremental.csv  incremental ICU model (manual vs predicted)
  16_dl_replicates_report.txt     human-readable
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])) / "final_analysis"
PER = BASE / "09_seg_agreement_percase.csv"
MERGED = BASE / "airway_morphology_merged.csv"
OUT = BASE

FEATURES = [
    ("volume_ml",          "Airway volume (mL)",        1),
    ("min_area_mm2",       "mCSA (mm²)",                0),
    ("compactness",        "Compactness",               3),
    ("rl_dev_max_abs_mm",  "maxMLS (mm)",               1),
    ("len_below_50mm2_mm", "LCSA50 (mm)",               0),
]


# DeLong (Sun & Xu 2014) for AUC comparison
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1); i = j
    T2 = np.empty(N, dtype=float); T2[J] = T + 1
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(positive[r, :])
        ty[r, :] = _midrank(negative[r, :])
        tz[r, :] = _midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1:
        sx = np.array([[sx]]); sy = np.array([[sy]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def auc_ci_delong(y, scores):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)
    order = (-y).argsort()
    pst = scores[order][np.newaxis, :]
    aucs, cov = _fast_delong(pst, int(y.sum()))
    se = float(np.sqrt(cov[0, 0]))
    return float(aucs[0]), max(0.0, float(aucs[0]) - 1.96 * se), min(1.0, float(aucs[0]) + 1.96 * se)


def delong_p_two_models(y, scores_a, scores_b):
    y = np.asarray(y).astype(int)
    order = (-y).argsort()
    pst = np.vstack((scores_a, scores_b))[:, order]
    aucs, cov = _fast_delong(pst, int(y.sum()))
    l = np.array([[1, -1]])
    var = float(np.dot(np.dot(l, cov), l.T).item())
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    p = float(2 * (1 - norm.cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), z, p


def fit_logistic_get_probs(X, y):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e9).fit(Xs, y)
    return m.predict_proba(Xs)[:, 1]


def mw_rb(values, group):
    g = np.asarray(group); v = np.asarray(values)
    a = v[g == 1]; b = v[g == 0]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - 2 * u / (len(a) * len(b))
    return float(np.median(a)), float(np.median(b)), float(p), float(-r)


def main():
    per = pd.read_csv(PER)
    clin = pd.read_csv(MERGED)[["case_id", "icu_binary", "abs_positive",
                                "crp", "rpe_present", "retroinf", "abs_diamax_mri",
                                "hospd_prolonged"]]
    # RPE presence: complete clinical flag (retroinf), not volumetric rpe_present.
    assert clin["retroinf"].notna().all(), "retroinf must be complete"
    clin["rpe_present"] = clin["retroinf"]
    df = per.merge(clin, on="case_id", how="inner")
    df = df.dropna(subset=["icu_binary"])
    print(f"Merged cohort: n={len(df)}, ICU={int(df.icu_binary.sum())}\n")

    # ── Univariate ICU: manual vs predicted ──────────────────────────────
    rows = []
    for key, label, dp in FEATURES:
        manual_col = f"manual_{key}"; pred_col = f"pred_{key}"
        sub = df[[manual_col, pred_col, "icu_binary"]].dropna()
        y = sub["icu_binary"].values.astype(int)

        for source, col in [("manual", manual_col), ("predicted", pred_col)]:
            v = sub[col].values
            med_icu, med_no, p, r = mw_rb(v, y)
            # Single-feature AUC (direction-aware)
            try:
                auc, lo, hi = auc_ci_delong(y, v if r >= 0 else -v)
            except Exception:
                auc, lo, hi = float("nan"), float("nan"), float("nan")
            rows.append(dict(feature=label, source=source, n=len(sub),
                             med_ICU=med_icu, med_nonICU=med_no,
                             p=p, r_rb=abs(r), AUC=auc,
                             AUC_lo=lo, AUC_hi=hi))
    uni = pd.DataFrame(rows)
    uni.to_csv(OUT / "16_dl_replicates_paper1.csv", index=False, float_format="%.4f")

    # ── Incremental ICU model: manual vs predicted airway features ──────
    AIRWAY4 = ["min_area_mm2", "compactness", "rl_dev_max_abs_mm", "len_below_50mm2_mm"]
    REF = ["crp", "rpe_present", "abs_diamax_mri"]
    inc_rows = []
    for cohort_label, sub_df in [
        ("Full cohort", df.dropna(subset=REF + [f"manual_{f}" for f in AIRWAY4] +
                                         [f"pred_{f}" for f in AIRWAY4])),
        ("Abscess-positive", df[df.abs_positive == 1].dropna(
            subset=REF + [f"manual_{f}" for f in AIRWAY4] +
                   [f"pred_{f}" for f in AIRWAY4]))
    ]:
        y = sub_df["icu_binary"].values.astype(int)
        n = len(y); n_icu = int(y.sum())

        # M1: Reference model alone (same for both)
        X_ref = sub_df[REF].values
        probs_ref = fit_logistic_get_probs(X_ref, y)
        auc_ref, lo_ref, hi_ref = auc_ci_delong(y, probs_ref)

        # M2-manual: Reference + manual 4 airway
        X_man = sub_df[REF + [f"manual_{f}" for f in AIRWAY4]].values
        probs_man = fit_logistic_get_probs(X_man, y)
        auc_man, lo_man, hi_man = auc_ci_delong(y, probs_man)
        _, _, z_man, p_man = delong_p_two_models(y, probs_man, probs_ref)

        # M2-predicted: Reference + DL-predicted 4 airway
        X_pred = sub_df[REF + [f"pred_{f}" for f in AIRWAY4]].values
        probs_pred = fit_logistic_get_probs(X_pred, y)
        auc_pred, lo_pred, hi_pred = auc_ci_delong(y, probs_pred)
        _, _, z_pred, p_pred = delong_p_two_models(y, probs_pred, probs_ref)

        # Direct comparison: manual vs predicted models
        _, _, z_mp, p_mp = delong_p_two_models(y, probs_man, probs_pred)

        for name, auc, lo, hi, d, z, p in [
            ("Reference model",                                          auc_ref, lo_ref, hi_ref, 0.0, np.nan, np.nan),
            ("Reference + 4 MANUAL airway features",                   auc_man, lo_man, hi_man,
             auc_man - auc_ref, z_man, p_man),
            ("Reference + 4 DL-PREDICTED airway features",             auc_pred, lo_pred, hi_pred,
             auc_pred - auc_ref, z_pred, p_pred),
        ]:
            inc_rows.append(dict(cohort=cohort_label, n=n, ICU=n_icu, model=name,
                                 AUC=auc, AUC_lo=lo, AUC_hi=hi,
                                 delta_vs_ref=d, DeLong_z=z, p_vs_ref=p))
        # Save the manual-vs-predicted direct comparison
        inc_rows.append(dict(cohort=cohort_label, n=n, ICU=n_icu,
                             model="-- manual airway vs predicted airway (head-to-head) --",
                             AUC=np.nan, AUC_lo=np.nan, AUC_hi=np.nan,
                             delta_vs_ref=auc_man - auc_pred,
                             DeLong_z=z_mp, p_vs_ref=p_mp))

    inc = pd.DataFrame(inc_rows)
    inc.to_csv(OUT / "16_dl_replicates_incremental.csv", index=False, float_format="%.4f")

    # ── Report ────────────────────────────────────────────────────────────
    rpt = OUT / "16_dl_replicates_report.txt"
    with open(rpt, "w") as fp:
        fp.write("Does DL-predicted morphometry reproduce Paper 1's ICU effect?\n")
        fp.write("=" * 75 + "\n\n")
        fp.write(f"Cohort: n={len(df)}, ICU={int(df.icu_binary.sum())}\n\n")

        fp.write("UNIVARIATE ICU COMPARISON (Mann–Whitney U)\n")
        fp.write("-" * 75 + "\n")
        fp.write(f"{'Feature':<22s} {'Source':>10s} {'ICU med':>10s} {'non-ICU':>10s} "
                 f"{'p':>10s} {'r_rb':>6s} {'AUC':>6s}\n")
        prev_feat = None
        for _, r in uni.iterrows():
            ftag = r.feature if r.feature != prev_feat else ""
            prev_feat = r.feature
            fp.write(f"{ftag:<22s} {r.source:>10s} {r.med_ICU:>10.2f} {r.med_nonICU:>10.2f} "
                     f"{r.p:>10.4f} {r.r_rb:>6.2f} {r.AUC:>6.3f}\n")

        fp.write("\n\nINCREMENTAL ICU MODELS (logistic + DeLong)\n")
        fp.write("-" * 75 + "\n")
        for cohort in inc.cohort.unique():
            fp.write(f"\n  {cohort}\n")
            sub = inc[inc.cohort == cohort]
            n = int(sub.n.iloc[0]); icu = int(sub.ICU.iloc[0])
            fp.write(f"  (n={n}, ICU={icu})\n")
            fp.write(f"  {'Model':<48s} {'AUC':>14s} {'ΔAUC':>8s} {'p':>8s}\n")
            for _, r in sub.iterrows():
                if "--" in r.model:
                    fp.write(f"  {r.model:<48s} {'':>14s} {r.delta_vs_ref:>+8.3f} "
                             f"{r.p_vs_ref:>8.3f}\n")
                else:
                    auc_str = f"{r.AUC:.3f} ({r.AUC_lo:.3f}-{r.AUC_hi:.3f})"
                    if r.model == "Reference model":
                        fp.write(f"  {r.model:<48s} {auc_str:>14s} {'(ref)':>8s}\n")
                    else:
                        fp.write(f"  {r.model:<48s} {auc_str:>14s} "
                                 f"{r.delta_vs_ref:>+8.3f} {r.p_vs_ref:>8.3f}\n")

        fp.write("\nINTERPRETATION CRIB\n")
        fp.write("-" * 75 + "\n")
        fp.write("  - Univariate: if 'predicted' p-values and direction match 'manual' for\n")
        fp.write("    the four ICU-associated features, the DL pipeline preserves Paper 1's\n")
        fp.write("    univariate effect, even with imperfect ICC.\n")
        fp.write("  - Incremental: if 'Reference + 4 PREDICTED airway' AUC ≈ 'Reference + 4 MANUAL',\n")
        fp.write("    and the head-to-head DeLong p > 0.05, the two pipelines are\n")
        fp.write("    statistically indistinguishable for ICU prediction.\n")

    print(f"Wrote {OUT / '16_dl_replicates_paper1.csv'}")
    print(f"Wrote {OUT / '16_dl_replicates_incremental.csv'}")
    print(f"Wrote {rpt}")
    print()
    print(open(rpt).read())


if __name__ == "__main__":
    main()
