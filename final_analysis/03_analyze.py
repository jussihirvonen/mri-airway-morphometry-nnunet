#!/usr/bin/env python3
"""
Statistical analysis on the FULL airway cohort (n=315 with clinical data).

Sections:
  1. Cohort description (overall, abs+/abs- split)
  2. Inter-observer agreement (carried from n=100 — uses pre-existing csv)
  3. Descriptive airway morphometry (overall and by abs status)
  4. ICU prediction (Mann-Whitney + LOO-CV) — overall + abs+ subset
  5. Prolonged LOS prediction
  6. RPE correlations (binary + continuous + partial controlling abscess)
  7. Abscess morphology correlations (abs+ subset)
  8. Incremental nested LOO-CV (abscess → +RPE → +airway)
  9. DNI vs control comparison (abs+ vs abs-)
  10. Mediastinal extension correlation
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

OUT_DIR = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1])) / "final_analysis"
MERGED = OUT_DIR / "airway_morphology_merged.csv"
RPT = OUT_DIR / "03_analysis_report.txt"

OUT = []


def pr(s=""):
    print(s)
    OUT.append(str(s))


def loo_auc(X, y):
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    sc = StandardScaler()
    for tr, te in loo.split(X):
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        m = LogisticRegression(max_iter=1000, solver='lbfgs')
        m.fit(Xtr, y[tr])
        probs[te] = m.predict_proba(Xte)[:, 1]
    try:
        return roc_auc_score(y, probs), probs
    except ValueError:
        return float('nan'), probs


def perm_auc_pvalue(X, y, n_perm=2000, seed=42):
    rng = np.random.default_rng(seed)
    obs_auc, _ = loo_auc(X, y)
    if np.isnan(obs_auc):
        return obs_auc, float('nan')
    null = []
    y_arr = np.asarray(y)
    for _ in range(n_perm):
        y_perm = rng.permutation(y_arr)
        a, _ = loo_auc(X, y_perm)
        null.append(a)
    null = np.asarray(null)
    p = (np.sum(null >= obs_auc) + 1) / (n_perm + 1)
    return obs_auc, p


def mw(values, group):
    g0 = values[group == 0]
    g1 = values[group == 1]
    u, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
    r_rb = 1 - 2 * u / (len(g0) * len(g1))
    return u, p, r_rb, np.median(g0), np.median(g1)


def main():
    df = pd.read_csv(MERGED)
    # RPE presence: use the complete clinical flag (retroinf, 0/1 for all 322),
    # not the volumetric derivation rpe_present (NA for 23 cases without a
    # volumetric RPE segmentation; agrees perfectly where both are defined).
    assert df["retroinf"].notna().all(), "retroinf must be complete"
    df["rpe_present"] = df["retroinf"]
    df = df[df.icu_binary.notna()].reset_index(drop=True)  # drop cases without clinical
    pr("=" * 75)
    pr(f"FULL-COHORT AIRWAY ANALYSIS — n = {len(df)}")
    pr("=" * 75)

    # ─────────────────────────────────────────────────────────────
    # 1. Cohort description
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("1. COHORT DESCRIPTION")
    pr("─" * 75)
    pr(f"  n = {len(df)}")
    pr(f"  age: mean={df.age.mean():.1f}, median={df.age.median():.0f}, IQR={df.age.quantile(0.25):.0f}-{df.age.quantile(0.75):.0f}")
    if 'sex' in df.columns:
        pr(f"  sex: {df.sex.value_counts(dropna=False).to_dict()}")
    pr(f"  abscess+: {(df.abs_positive == 1).sum()}  abscess-: {(df.abs_positive == 0).sum()}  unknown: {df.abs_positive.isna().sum()}")
    pr(f"  ICU: {(df.icu_binary == 1).sum()} ({100 * (df.icu_binary == 1).mean():.1f}%)")
    pr(f"  hospd: median={df.hospd.median():.1f}, IQR={df.hospd.quantile(0.25):.0f}-{df.hospd.quantile(0.75):.0f}")
    pr(f"  prolonged LOS (>5d): {(df.hospd_prolonged == 1).sum()} ({100 * (df.hospd_prolonged == 1).mean():.1f}%)")
    pr(f"  RPE present: {(df.rpe_present == 1).sum()}/{df.rpe_present.notna().sum()}")
    pr(f"  RPE volume (mL): median={df.rpe_volume_ml.median():.2f}, mean={df.rpe_volume_ml.mean():.2f}")
    pr(f"  mediastinal edema: {(df.medoed_binary == 1).sum()}/{df.medoed_binary.notna().sum()}")

    pos = df[df.abs_positive == 1].copy()
    neg = df[df.abs_positive == 0].copy()
    pr(f"\n  Abscess+ subgroup (n={len(pos)}):")
    pr(f"    ICU: {(pos.icu_binary == 1).sum()} ({100 * (pos.icu_binary == 1).mean():.1f}%)")
    pr(f"    RPE+: {(pos.rpe_present == 1).sum()}/{pos.rpe_present.notna().sum()}")
    pr(f"  Abscess- (control) subgroup (n={len(neg)}):")
    pr(f"    ICU: {(neg.icu_binary == 1).sum()} ({100 * (neg.icu_binary == 1).mean():.1f}%)")
    pr(f"    RPE+: {(neg.rpe_present == 1).sum()}/{neg.rpe_present.notna().sum()}")

    # ─────────────────────────────────────────────────────────────
    # 2. Inter-observer (read from existing CSV from n=100 stage)
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("2. INTER-OBSERVER (unchanged from n=20 overlap subset)")
    pr("─" * 75)
    iobs_csv = OUT_DIR.parent / "interobserver_results.csv"
    try:
        iobs = pd.read_csv(iobs_csv)
        if 'dice' in iobs.columns:
            pr(f"  Dice: mean={iobs.dice.mean():.3f}, range {iobs.dice.min():.2f}-{iobs.dice.max():.2f}")
        else:
            pr(f"  cols: {list(iobs.columns)}")
            pr(iobs.describe().to_string())
    except Exception as e:
        pr(f"  (read error: {e})")

    # ─────────────────────────────────────────────────────────────
    # 3. Descriptive airway morphometry
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("3. AIRWAY MORPHOMETRY — DESCRIPTIVE")
    pr("─" * 75)
    feats = ['volume_ml', 'cc_length_mm', 'min_area_mm2', 'min_area_5sl', 'mean_area_mm2',
             'max_area_mm2', 'stenosis_pct', 'elongation', 'compactness',
             'narrowed_length_25pct_mm', 'narrowed_length_50pct_mm', 'tortuosity',
             'ap_mean_mm', 'lat_mean_mm', 'ap_at_narrowest_mm', 'lat_at_narrowest_mm',
             'rl_dev_max_mm', 'circularity_min', 'hydraulic_diam_min_mm']
    rows = []
    for f in feats:
        if f not in df.columns:
            continue
        v = df[f].dropna()
        rows.append({'feature': f, 'mean': v.mean(), 'sd': v.std(),
                     'median': v.median(), 'q25': v.quantile(0.25),
                     'q75': v.quantile(0.75), 'min': v.min(), 'max': v.max()})
    desc = pd.DataFrame(rows).set_index('feature').round(2)
    pr(desc.to_string())

    # ─────────────────────────────────────────────────────────────
    # 4. ICU prediction
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("4. ICU PREDICTION (full cohort, n={})".format((df.icu_binary.notna()).sum()))
    pr("─" * 75)
    icu = df.icu_binary.values.astype(int)
    icu_feats = ['min_area_mm2', 'min_area_5sl', 'hydraulic_diam_at_narrowest_mm',
                 'compactness', 'elongation', 'narrowed_length_25pct_mm',
                 'stenosis_pct', 'volume_ml', 'mean_area_mm2', 'circularity_min',
                 'ap_at_narrowest_mm', 'lat_at_narrowest_mm']
    pr(f"  Mann-Whitney U:")
    pr(f"  {'feature':32s} {'p':>10s}  {'r_rb':>8s}  {'non-ICU med':>12s}  {'ICU med':>10s}")
    for f in icu_feats:
        if f not in df.columns:
            continue
        mask = df[f].notna()
        v = df.loc[mask, f].values
        g = icu[mask]
        u, p, r_rb, m0, m1 = mw(v, g)
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        pr(f"  {f:32s} {p:10.4f}{sig:2s} {r_rb:+8.3f}  {m0:12.1f}  {m1:10.1f}")

    pr(f"\n  LOO-CV (logistic regression, standardized features):")
    for feats_l, label in [
        (['min_area_mm2'], 'min_area_1sl alone'),
        (['min_area_5sl'], 'min_area_5sl alone'),
        (['elongation'], 'elongation alone'),
        (['compactness'], 'compactness alone'),
        (['min_area_mm2', 'elongation'], 'min_area_1sl + elongation'),
        (['min_area_5sl', 'elongation'], 'min_area_5sl + elongation'),
        (['min_area_mm2', 'elongation', 'compactness'], 'min_area + elongation + compactness'),
    ]:
        sub = df[feats_l + ['icu_binary']].dropna()
        X = sub[feats_l].values
        y = sub['icu_binary'].values.astype(int)
        auc, _ = loo_auc(X, y)
        pr(f"    {label:45s} AUC={auc:.3f}  (n={len(sub)}, ICU={int(y.sum())})")

    # Permutation p-values for top single + combined predictors
    pr(f"\n  Permutation p-values for LOO-CV AUC (2000 perms):")
    for feats_l, label in [
        (['min_area_mm2'], 'min_area_1sl'),
        (['min_area_mm2', 'elongation'], 'min_area_1sl + elongation'),
    ]:
        sub = df[feats_l + ['icu_binary']].dropna()
        X = sub[feats_l].values
        y = sub['icu_binary'].values.astype(int)
        auc, p_perm = perm_auc_pvalue(X, y, n_perm=2000)
        pr(f"    {label:35s} AUC={auc:.3f}  perm p={p_perm:.4f}  (n={len(sub)})")

    # ─────────────────────────────────────────────────────────────
    # 5. Prolonged LOS prediction
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("5. PROLONGED LOS (>5d)")
    pr("─" * 75)
    los_mask = df.hospd_prolonged.notna()
    losdf = df[los_mask].copy()
    los = losdf.hospd_prolonged.values.astype(int)
    pr(f"  n={len(losdf)}, prolonged={int(los.sum())}")
    for f in icu_feats:
        if f not in losdf.columns:
            continue
        valid = losdf[f].notna()
        v = losdf.loc[valid, f].values
        g = los[valid]
        if g.sum() < 3 or (len(g) - g.sum()) < 3:
            continue
        u, p, r_rb, m0, m1 = mw(v, g)
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        pr(f"  {f:32s} p={p:.4f}{sig:3s} r_rb={r_rb:+.3f}  short={m0:.1f}  prol={m1:.1f}")

    # ─────────────────────────────────────────────────────────────
    # 6. RPE correlations
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("6. RPE CORRELATIONS")
    pr("─" * 75)
    rpe_bin_df = df[df.rpe_present.notna()].copy()
    rpe = rpe_bin_df.rpe_present.values.astype(int)
    pr(f"\n  6a. RPE PRESENCE (binary, n={len(rpe_bin_df)}: RPE+={int(rpe.sum())}, RPE-={int((1-rpe).sum())})")
    for f in icu_feats + ['stenosis_pct', 'ap_mean_mm', 'rl_dev_max_mm']:
        if f not in rpe_bin_df.columns:
            continue
        valid = rpe_bin_df[f].notna()
        v = rpe_bin_df.loc[valid, f].values
        g = rpe[valid]
        if g.sum() < 3 or (len(g) - g.sum()) < 3:
            continue
        u, p, r_rb, m0, m1 = mw(v, g)
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        pr(f"  {f:32s} p={p:.4f}{sig:3s} r_rb={r_rb:+.3f}  No={m0:.1f}  RPE={m1:.1f}")

    pr(f"\n  6b. RPE VOLUME (continuous, Spearman) — all cases with rpevol data")
    rpe_v_df = df[df.rpe_volume_ml.notna()].copy()
    pr(f"  n={len(rpe_v_df)}, RPE>0: {(rpe_v_df.rpe_volume_ml > 0).sum()}")
    for f in ['min_area_mm2', 'min_area_5sl', 'elongation', 'volume_ml',
              'narrowed_length_25pct_mm', 'narrowed_length_50pct_mm',
              'stenosis_pct', 'ap_mean_mm', 'mean_area_mm2', 'compactness']:
        if f not in rpe_v_df.columns:
            continue
        valid = rpe_v_df[[f, 'rpe_volume_ml']].dropna()
        if len(valid) < 10:
            continue
        rho, p = stats.spearmanr(valid['rpe_volume_ml'], valid[f])
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        pr(f"  {f:32s} rho={rho:+.3f}  p={p:.4f}{sig}  n={len(valid)}")

    pr(f"\n  6c. RPE volume — RPE>0 only (Spearman)")
    sub = rpe_v_df[rpe_v_df.rpe_volume_ml > 0].copy()
    pr(f"  n={len(sub)}")
    for f in ['min_area_mm2', 'min_area_5sl', 'elongation', 'volume_ml',
              'narrowed_length_25pct_mm', 'narrowed_length_50pct_mm']:
        if f not in sub.columns:
            continue
        valid = sub[[f, 'rpe_volume_ml']].dropna()
        if len(valid) < 10:
            continue
        rho, p = stats.spearmanr(valid['rpe_volume_ml'], valid[f])
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        pr(f"  {f:32s} rho={rho:+.3f}  p={p:.4f}{sig}")

    # ─────────────────────────────────────────────────────────────
    # 7. Abscess morphology correlations (abs+ subset)
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("7. ABSCESS MORPHOLOGY vs AIRWAY (abscess+ subset)")
    pr("─" * 75)
    posdf = df[df.abs_positive == 1].copy()
    pr(f"  n={len(posdf)}")
    abs_feats = [
        ('abs_volume_radiomics_ml', 'Abs volume (radiomics, mL)'),
        ('abs_max3d_radiomics', 'Abs max 3D diameter'),
        ('abs_sphericity_radiomics', 'Abs sphericity'),
        ('abs_compactness_radiomics', 'Abs compactness'),
        ('abs_volume_mri', 'Abs volume (clinical master, mL or mm³)'),
        ('abs_diamax_mri', 'Abs max diameter (clinical master)'),
    ]
    airway = ['min_area_mm2', 'min_area_5sl', 'elongation', 'volume_ml',
              'narrowed_length_25pct_mm', 'stenosis_pct', 'mean_area_mm2',
              'compactness']
    for af, alabel in abs_feats:
        if af not in posdf.columns or posdf[af].notna().sum() < 30:
            continue
        pr(f"\n  --- {alabel} ({af}) — n with data: {posdf[af].notna().sum()} ---")
        for wf in airway:
            valid = posdf[[af, wf]].dropna()
            if len(valid) < 15:
                continue
            rho, p = stats.spearmanr(valid[af], valid[wf])
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            pr(f"    {wf:32s} rho={rho:+.3f}  p={p:.4f}{sig}")

    # ─────────────────────────────────────────────────────────────
    # 8. Partial correlations: RPE vol → airway controlling abscess vol
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("8. PARTIAL CORRELATIONS: RPE vol → airway (controlling abs vol)")
    pr("─" * 75)
    p_df = df[['rpe_volume_ml', 'abs_volume_radiomics_ml', 'min_area_mm2',
               'min_area_5sl', 'elongation', 'volume_ml',
               'narrowed_length_25pct_mm']].dropna()
    pr(f"  n with RPE vol + abs radiomics vol: {len(p_df)}")
    for wf in ['min_area_mm2', 'min_area_5sl', 'elongation', 'volume_ml',
               'narrowed_length_25pct_mm']:
        s_r, s_p = stats.spearmanr(p_df['rpe_volume_ml'], p_df[wf])
        lr = LinearRegression()
        lr.fit(p_df[['abs_volume_radiomics_ml']], p_df[wf])
        r_air = p_df[wf].values - lr.predict(p_df[['abs_volume_radiomics_ml']])
        lr.fit(p_df[['abs_volume_radiomics_ml']], p_df['rpe_volume_ml'])
        r_rpe = p_df['rpe_volume_ml'].values - lr.predict(p_df[['abs_volume_radiomics_ml']])
        p_r, p_p = stats.pearsonr(r_rpe, r_air)
        pr(f"  {wf:32s}  simple rho={s_r:+.3f} p={s_p:.4f}  partial r={p_r:+.3f} p={p_p:.4f}")

    # ─────────────────────────────────────────────────────────────
    # 9. Incremental LOO-CV models for ICU
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("9. INCREMENTAL ICU MODELS (LOO-CV) — abscess+ subset with complete data")
    pr("─" * 75)
    inc = posdf[['icu_binary', 'abs_volume_radiomics_ml', 'abs_max3d_radiomics',
                  'rpe_present', 'rpe_volume_ml', 'min_area_mm2',
                  'min_area_5sl', 'elongation']].dropna()
    pr(f"  n with complete data: {len(inc)}  ICU+={int(inc.icu_binary.sum())}")
    if len(inc) >= 30 and inc.icu_binary.sum() >= 5:
        for fs, lab in [
            (['abs_volume_radiomics_ml'], 'Abs volume only'),
            (['abs_volume_radiomics_ml', 'abs_max3d_radiomics'], 'Abs vol + diameter'),
            (['abs_volume_radiomics_ml', 'abs_max3d_radiomics', 'rpe_present'], 'Abs + RPE(bin)'),
            (['abs_volume_radiomics_ml', 'abs_max3d_radiomics', 'rpe_volume_ml'], 'Abs + RPE vol'),
            (['min_area_mm2', 'elongation'], 'Airway only (1sl)'),
            (['min_area_5sl', 'elongation'], 'Airway only (5sl)'),
            (['abs_volume_radiomics_ml', 'abs_max3d_radiomics', 'rpe_volume_ml',
              'min_area_mm2', 'elongation'], 'Abs + RPE + Airway(1sl)'),
            (['abs_volume_radiomics_ml', 'abs_max3d_radiomics', 'rpe_volume_ml',
              'min_area_5sl', 'elongation'], 'Abs + RPE + Airway(5sl)'),
        ]:
            X = inc[fs].values
            y = inc['icu_binary'].values.astype(int)
            auc, _ = loo_auc(X, y)
            pr(f"  {lab:45s} AUC={auc:.3f}")

    # ─────────────────────────────────────────────────────────────
    # 10. DNI (abs+) vs control (abs-) airway comparison
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("10. DNI vs CONTROL airway morphometry")
    pr("─" * 75)
    pos_v = df[df.abs_positive == 1].copy()
    neg_v = df[df.abs_positive == 0].copy()
    pr(f"  abs+ n={len(pos_v)}, abs- n={len(neg_v)}")
    for f in feats:
        if f not in df.columns:
            continue
        a = pos_v[f].dropna().values
        b = neg_v[f].dropna().values
        if len(a) < 5 or len(b) < 5:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        r_rb = 1 - 2 * u / (len(a) * len(b))
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        pr(f"  {f:32s} p={p:.4f}{sig:3s} r_rb={r_rb:+.3f}  abs+med={np.median(a):.1f}  abs-med={np.median(b):.1f}")

    # ─────────────────────────────────────────────────────────────
    # 11. Mediastinal edema
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("11. MEDIASTINAL EDEMA vs AIRWAY")
    pr("─" * 75)
    me_df = df[df.medoed_binary.notna()].copy()
    me = me_df.medoed_binary.values.astype(int)
    if me.sum() >= 5:
        pr(f"  n={len(me_df)}, mediastinal edema+={int(me.sum())}")
        for f in icu_feats:
            if f not in me_df.columns:
                continue
            valid = me_df[f].notna()
            v = me_df.loc[valid, f].values
            g = me[valid]
            if g.sum() < 3 or (len(g) - g.sum()) < 3:
                continue
            u, p, r_rb, m0, m1 = mw(v, g)
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            pr(f"  {f:32s} p={p:.4f}{sig:3s} r_rb={r_rb:+.3f}  noMedOed={m0:.1f}  medOed={m1:.1f}")

    # ─────────────────────────────────────────────────────────────
    # 12. Incremental ICU model — FULL COHORT, Turku-matched reference
    #     Turku reference = CRP + RPE(present) + max abscess diameter
    #     (absdiamax = 0 in abscess-negative patients, per Heikkinen model)
    #     Airway block = pre-specified features (min CSA, compactness,
    #     max midline shift, length<50 mm²)
    # ─────────────────────────────────────────────────────────────
    pr("\n" + "─" * 75)
    pr("12. INCREMENTAL ICU MODEL — FULL COHORT (Turku-matched, n up to 323)")
    pr("─" * 75)
    turku = ['crp', 'rpe_present', 'abs_diamax_mri']
    airway_ps = ['min_area_mm2', 'compactness', 'rl_dev_max_abs_mm', 'len_below_50mm2_mm']
    needed = turku + airway_ps + ['icu_binary']
    fc = df[needed].dropna().reset_index(drop=True)
    y_fc = fc['icu_binary'].values.astype(int)
    pr(f"  complete-case n={len(fc)}  ICU+={int(y_fc.sum())}")

    model_probs = {}
    for fs, lab in [
        (turku, 'Turku ref (CRP+RPE+absdiamax)'),
        (turku + ['min_area_mm2'], 'Turku + min CSA'),
        (turku + airway_ps, 'Turku + airway (pre-specified 4)'),
        (airway_ps, 'Airway pre-specified alone'),
    ]:
        X = fc[fs].values
        auc, probs = loo_auc(X, y_fc)
        model_probs[lab] = probs
        pr(f"  {lab:38s} AUC={auc:.3f}")

    # Paired bootstrap ΔAUC: Turku+airway vs Turku ref (on fixed LOO probs)
    def boot_delta(probs_a, probs_b, y, n_boot=2000, seed=42):
        rng = np.random.default_rng(seed)
        n = len(y)
        base = roc_auc_score(y, probs_a) - roc_auc_score(y, probs_b)
        deltas = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if y[idx].sum() == 0 or y[idx].sum() == len(idx):
                continue
            d = roc_auc_score(y[idx], probs_a[idx]) - roc_auc_score(y[idx], probs_b[idx])
            deltas.append(d)
        deltas = np.asarray(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_two = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
        return base, lo, hi, p_two

    ref = model_probs['Turku ref (CRP+RPE+absdiamax)']
    for lab in ['Turku + min CSA', 'Turku + airway (pre-specified 4)']:
        d, lo, hi, pv = boot_delta(model_probs[lab], ref, y_fc)
        pr(f"  ΔAUC {lab} vs Turku ref: {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={pv:.3f}")

    with open(RPT, 'w') as fp:
        fp.write('\n'.join(OUT))
    print(f"\n\nReport saved: {RPT}")


if __name__ == "__main__":
    main()
