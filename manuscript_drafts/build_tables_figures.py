#!/usr/bin/env python3
"""
Build all tables and figures for Paper 1 morphometry (post-2026-06-01 refresh).

Tables (Word, abscess-paper style for direct comparability):
  Table 1  Cohort characteristics — single column (n=322), no abs split
  Table 2  Inter-observer agreement — 5 prespecified features, ICC + 95% CI
  Table 3  Univariate features × ICU and prolonged stay (abs Table 2 layout)
  Table 4  Incremental ICU model comparison (standard logistic + DeLong)

Figures (PNG + PDF, 300 dpi):
  (Figure 1 = flowchart, user will draw)
  Figure 2  Convergent + discriminant validity heatmap (5 features × 7 markers)
  Figure 3  5-panel features × ICU box/strip plots
  Figure 4  Inter-observer scatter (5 panels), ICC annotated

Supplementary:
  SuppTable1  Full descriptive morphometry (all features)
  SuppTable2  Full convergent-validity panel (effect + p for full/abs+/abs−)
  SuppFig1    Cross-cohort validity heatmap (copied from existing)
  SuppFig2    Bland–Altman for airway volume + min CSA inter-observer

All outputs land in manuscript_drafts/tables_figures/.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
FA = BASE / "final_analysis"
OUT = BASE / "manuscript_drafts" / "tables_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLACK = RGBColor(0, 0, 0)

# Load data once
MERGED = pd.read_csv(FA / "airway_morphology_merged.csv")
# RPE presence: use the complete clinical flag (retroinf, 0/1 for all 322)
# rather than the volumetric derivation rpe_present (NA for the 23 cases without
# a volumetric RPE segmentation; 8 of them RPE-positive). The two agree
# perfectly where both are defined. This drives Table 1 (RPE count) and Table 4
# (reference Turku ICU model complete-case n: 297 -> 320).
assert MERGED["retroinf"].notna().all(), "retroinf must be complete"
MERGED["rpe_present"] = MERGED["retroinf"]
PANEL = pd.read_csv(FA / "08_severity_compare_combined.csv")
IRPC = pd.read_csv(FA / "13_inter_rater_percase_n20.csv")
IRCC = pd.read_csv(FA / "13_inter_rater_icc_n20.csv")


# ----------------------------------------------------------------------
# docx helpers
# ----------------------------------------------------------------------
def setup_doc(doc):
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(2)
        s.left_margin = s.right_margin = Cm(2)
    n = doc.styles["Normal"]
    n.font.name = "Times New Roman"; n.font.size = Pt(12); n.font.color.rgb = BLACK
    n.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    n.paragraph_format.space_after = Pt(0)


def _style_run(r, bold=False, italic=False, size=12):
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.font.color.rgb = BLACK
    r.font.bold = bold
    r.font.italic = italic


def title(doc, t):
    p = doc.add_paragraph()
    r = p.add_run(t); _style_run(r, bold=True, size=12)
    p.paragraph_format.space_after = Pt(4)
    return p


def caption(doc, t):
    p = doc.add_paragraph()
    r = p.add_run(t); _style_run(r, size=10, italic=True)
    p.paragraph_format.space_before = Pt(4)
    return p


def make_table(doc, headers, rows, col_widths_cm=None, header_indent_rows=None, bold_rows=None):
    """Build a simple grid table. headers: list of strings. rows: list of list of strings.
       header_indent_rows: set of row indices (0-based among data rows) to indent (sub-rows).
       bold_rows: set of row indices to bold (e.g., outcome subheaders)."""
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Table Grid"
    # header row
    hdr = tbl.rows[0].cells
    for j, h in enumerate(headers):
        hdr[j].text = ""
        p = hdr[j].paragraphs[0]
        r = p.add_run(h); _style_run(r, bold=True)
    # data rows
    indent_set = set(header_indent_rows or [])
    bold_set = set(bold_rows or [])
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            txt = str(val) if val is not None else ""
            if j == 0 and i in indent_set:
                txt = "    " + txt
            r = p.add_run(txt)
            _style_run(r, bold=(i in bold_set))
    # column widths
    if col_widths_cm:
        for j, w in enumerate(col_widths_cm):
            for r in tbl.rows:
                r.cells[j].width = Cm(w)
    return tbl


# ----------------------------------------------------------------------
# formatters
# ----------------------------------------------------------------------
def med_iqr(values, decimals=1):
    v = pd.Series(values).dropna()
    if len(v) == 0: return "—"
    return f"{v.median():.{decimals}f} ({v.quantile(0.25):.{decimals}f}–{v.quantile(0.75):.{decimals}f})"


def n_pct(yes_count, total):
    if total == 0: return "—"
    return f"{int(yes_count)} ({100*yes_count/total:.1f})"


def fmt_p(p):
    if pd.isna(p): return "—"
    if p < 0.001:
        return f"{p:.1e}".replace("e-0", " × 10⁻")
    return f"{p:.3f}"


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def auc_ci_delong(y, scores):
    """Single-classifier AUC + 95% CI via DeLong variance estimator."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)
    order = (-y).argsort()
    label_1_count = int(y.sum())
    pst = scores[order][np.newaxis, :]
    auc, cov = _fast_delong(pst, label_1_count)
    se = float(np.sqrt(cov[0, 0]))
    lo = max(0.0, auc[0] - 1.96 * se)
    hi = min(1.0, auc[0] + 1.96 * se)
    return float(auc[0]), float(lo), float(hi)


def delong_p_two_models(y, scores_a, scores_b):
    """Returns (auc_a, auc_b, z, p) DeLong test for two correlated AUCs."""
    y = np.asarray(y).astype(int)
    order = (-y).argsort()
    label_1_count = int(y.sum())
    pst = np.vstack((scores_a, scores_b))[:, order]
    aucs, cov = _fast_delong(pst, label_1_count)
    l = np.array([[1, -1]])
    var = float(np.dot(np.dot(l, cov), l.T).item())
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    p = float(2 * (1 - norm.cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), z, p


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1); i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Sun & Xu (2014) fast DeLong implementation."""
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


def fit_logistic_get_probs(X, y):
    """Standard in-sample logistic regression; returns predicted probabilities."""
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e9).fit(Xs, y)
    return m.predict_proba(Xs)[:, 1]


def mw_rb(values, group):
    """Mann–Whitney U + rank-biserial r between values for group==1 vs group==0."""
    g = np.asarray(group); v = np.asarray(values)
    a = v[g == 1]; b = v[g == 0]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - 2 * u / (len(a) * len(b))
    return float(-r), float(p)  # positive r = higher in group==1


# ======================================================================
# TABLE 1 — Cohort characteristics (single column, n=322)
# ======================================================================
def build_table1():
    df = MERGED
    n_total = len(df)
    rows = []
    rows.append(["Age, years", med_iqr(df.age, 0)])
    n_male = int((df.sex == 1).sum())
    n_female = int((df.sex == 2).sum())
    rows.append(["Sex (male/female)", f"{n_male}/{n_female} ({100*n_male/n_total:.1f}/{100*n_female/n_total:.1f})"])
    rows.append(["CRP, mg/L", med_iqr(df.crp, 0)])
    rows.append(["Hospital stay, days", med_iqr(df.hospd, 0)])
    rows.append(["Prolonged stay (>5 days)", n_pct((df.hospd_prolonged == 1).sum(), df.hospd_prolonged.notna().sum())])
    rows.append(["ICU admission", n_pct((df.icu_binary == 1).sum(), n_total)])
    rows.append(["Abscess present on MRI", n_pct((df.abs_positive == 1).sum(), n_total)])
    # abscess max diameter, abscess-positive only
    abs_pos = df[df.abs_positive == 1]
    rows.append(["Maximum abscess diameter, mm (abscess-positive only)", med_iqr(abs_pos.abs_diamax_mri, 0)])
    rows.append(["Retropharyngeal edema present",
                 n_pct((df.rpe_present == 1).sum(), df.rpe_present.notna().sum())])
    rows.append(["Retropharyngeal edema volume, mL", med_iqr(df.rpe_volume_ml, 1)])
    rows.append(["Mediastinal edema present",
                 n_pct((df.medoed_binary == 1).sum(), df.medoed_binary.notna().sum())])

    doc = Document(); setup_doc(doc)
    title(doc, "Table 1. Patient characteristics.")
    make_table(doc, ["Characteristic", f"All patients (n = {n_total})"], rows,
               col_widths_cm=[10, 6])
    caption(doc,
        "Values are median (interquartile range) for continuous variables and n (%) for "
        "categorical variables, unless otherwise indicated. Maximum abscess diameter is "
        "reported in abscess-positive patients only. CRP, C-reactive protein.")
    doc.save(OUT / "Table1.docx")
    return rows


# ======================================================================
# TABLE 2 — Inter-observer agreement
# ======================================================================
def build_table2():
    feature_labels = {
        "volume_ml":           "Airway volume, mL",
        "min_area_mm2":        "Minimum cross-sectional area, mm²",
        "compactness":         "Airway compactness",
        "rl_dev_max_abs_mm":   "maxMLS (mm)",
        "len_below_50mm2_mm":  "Length of airway below 50 mm², mm",
    }
    icc = IRCC.set_index("feature")
    rows = []
    for feat, lab in feature_labels.items():
        if feat not in icc.index: continue
        r = icc.loc[feat]
        rows.append([lab,
                     f"{float(r['ICC']):.3f}",
                     f"{float(r['CI_lower']):.3f}–{float(r['CI_upper']):.3f}"])

    # Footer Dice / HD95 from per-case file
    dice = IRPC["dice"]
    hd95 = IRPC["hd95_mm"]

    doc = Document(); setup_doc(doc)
    title(doc, "Table 2. Inter-observer agreement on a 20-case dual-segmented subset.")
    make_table(doc, ["Feature", "ICC(2,1)", "95% CI"], rows,
               col_widths_cm=[10, 3, 4])
    caption(doc,
        f"Dice similarity coefficient: mean {dice.mean():.3f} (range {dice.min():.3f}–{dice.max():.3f}). "
        f"95th-percentile Hausdorff distance: median {hd95.median():.1f} mm "
        f"(interquartile range {hd95.quantile(0.25):.1f}–{hd95.quantile(0.75):.1f}; max {hd95.max():.1f}). "
        "ICC(2,1) is the two-way random-effects, single-rater, absolute-agreement intraclass "
        "correlation coefficient. CI, confidence interval.")
    doc.save(OUT / "Table2.docx")


# ======================================================================
# TABLE 3 — Univariate ICU + prolonged stay (abscess-paper Table 2 style)
# ======================================================================
def build_table3():
    """Layout per abs paper Table 2:
       Feature | Positive (n=K1) | Negative (n=K0) | p | r | AUC
       Outcome blocks: ICU admission, Prolonged hospitalization.
    """
    feature_keys = [
        ("volume_ml",          "Airway volume, mL",                       1),
        ("min_area_mm2",       "Minimum cross-sectional area, mm²",       0),
        ("compactness",        "Airway compactness",                      3),
        ("rl_dev_max_abs_mm",  "maxMLS (mm)",               1),
        ("len_below_50mm2_mm", "Length of airway below 50 mm², mm",       0),
    ]
    blocks = [("icu_binary", "ICU admission"), ("hospd_prolonged", "Prolonged hospitalization")]

    rows = []; bold_rows = []; indent_rows = []
    row_idx = 0
    for outcome, label in blocks:
        sub = MERGED[[outcome] + [k for k, _, _ in feature_keys]].dropna(subset=[outcome])
        pos_n = int((sub[outcome] == 1).sum()); neg_n = int((sub[outcome] == 0).sum())
        rows.append([f"{label}", f"(n = {pos_n})", f"(n = {neg_n})", "", "", ""])
        bold_rows.append(row_idx); row_idx += 1
        for key, lab, dp in feature_keys:
            s = sub[[outcome, key]].dropna()
            v = s[key].values; g = s[outcome].values
            r, p = mw_rb(v, g)
            pos_med = med_iqr(s.loc[s[outcome] == 1, key], dp)
            neg_med = med_iqr(s.loc[s[outcome] == 0, key], dp)
            try:
                auc, _, _ = auc_ci_delong(g, v if r >= 0 else -v)
            except Exception:
                auc = float("nan")
            rows.append([lab, pos_med, neg_med, fmt_p(p), f"{abs(r):.2f}",
                         f"{auc:.3f}" if not np.isnan(auc) else "—"])
            indent_rows.append(row_idx); row_idx += 1

    doc = Document(); setup_doc(doc)
    title(doc, "Table 3. Pre-specified airway features by ICU admission and prolonged hospitalization.")
    make_table(doc, ["Feature", "Positive", "Negative", "p value", "r", "AUC"], rows,
               col_widths_cm=[7, 3.5, 3.5, 2.5, 1.5, 1.5],
               header_indent_rows=set(indent_rows), bold_rows=set(bold_rows))
    caption(doc,
        "Values are median (interquartile range). p, Mann–Whitney U test (two-sided). "
        "r, absolute rank-biserial correlation. AUC, area under the receiver operating "
        "characteristic curve for each feature alone. Prolonged hospitalization is defined "
        "as hospital stay > 5 days. Cohort: full study population (n = 322).")
    doc.save(OUT / "Table3.docx")


# ======================================================================
# TABLE 4 — Incremental ICU model comparison (standard logistic + DeLong)
# ======================================================================
def build_table4():
    feats_airway4 = ["min_area_mm2", "compactness", "rl_dev_max_abs_mm", "len_below_50mm2_mm"]
    models = {
        "Reference (Turku ICU-risk model)":     ["crp", "rpe_present", "abs_diamax_mri"],
        "Reference + minimum CSA":              ["crp", "rpe_present", "abs_diamax_mri",
                                                 "min_area_mm2"],
        "Reference + four airway features":     ["crp", "rpe_present", "abs_diamax_mri",
                                                 *feats_airway4],
    }

    rows = []; indent_rows = []; csv_rows = []
    for cohort_label, df_sub in [
        ("Full cohort", MERGED.dropna(subset=["icu_binary", "crp", "rpe_present", "abs_diamax_mri"] + feats_airway4)),
        ("Abscess-positive subset", MERGED[MERGED["abs_positive"] == 1].dropna(
            subset=["icu_binary", "crp", "rpe_present", "abs_diamax_mri"] + feats_airway4))
    ]:
        y = df_sub["icu_binary"].values.astype(int)
        n_total = len(y); n_icu = int(y.sum())
        # subgroup header row
        rows.append([f"{cohort_label} (n = {n_total}, ICU = {n_icu})", "", "", "", ""])
        # Fit ref first
        ref_probs = fit_logistic_get_probs(df_sub[models["Reference (Turku ICU-risk model)"]].values, y)
        ref_auc, ref_lo, ref_hi = auc_ci_delong(y, ref_probs)
        rows.append([list(models.keys())[0],
                     f"{ref_auc:.3f} ({ref_lo:.3f}–{ref_hi:.3f})",
                     "—", "—", "(reference)"])
        indent_rows.append(len(rows) - 1)
        csv_rows.append({"cohort": cohort_label, "n": n_total, "ICU_events": n_icu,
                         "model": "Reference (Turku ICU-risk model)",
                         "AUC": ref_auc, "CI_lo": ref_lo, "CI_hi": ref_hi,
                         "delta_vs_ref": 0.0, "DeLong_z": float("nan"),
                         "p_vs_ref": float("nan")})
        for name, feats in list(models.items())[1:]:
            probs = fit_logistic_get_probs(df_sub[feats].values, y)
            auc, lo, hi = auc_ci_delong(y, probs)
            _, _, z, p = delong_p_two_models(y, probs, ref_probs)
            d = auc - ref_auc
            rows.append([name,
                         f"{auc:.3f} ({lo:.3f}–{hi:.3f})",
                         f"{d:+.3f}",
                         f"{z:+.2f}",
                         fmt_p(p)])
            indent_rows.append(len(rows) - 1)
            csv_rows.append({"cohort": cohort_label, "n": n_total, "ICU_events": n_icu,
                             "model": name, "AUC": auc, "CI_lo": lo, "CI_hi": hi,
                             "delta_vs_ref": d, "DeLong_z": z, "p_vs_ref": p})
    pd.DataFrame(csv_rows).to_csv(OUT / "Table4_data.csv", index=False, float_format="%.4f")

    doc = Document(); setup_doc(doc)
    title(doc, "Table 4. Incremental value of airway features for ICU prediction.")
    make_table(doc, ["Model", "AUC (95% CI)", "ΔAUC vs reference", "DeLong z", "p value"], rows,
               col_widths_cm=[7, 4.2, 2.8, 1.8, 2.2],
               header_indent_rows=set(indent_rows),
               bold_rows={r for r in range(len(rows)) if r not in indent_rows})
    caption(doc,
        "Standard logistic-regression models on standardized features; AUC 95% confidence "
        "intervals by DeLong's method. ΔAUC compared with the reference Turku ICU-risk model "
        "(C-reactive protein + retropharyngeal-edema presence + maximum abscess diameter); "
        "between-model comparisons by DeLong's test for correlated AUCs. The reference model "
        "and the abscess-positive sensitivity reference use the established Turku model "
        "definition with maximum abscess diameter set to zero in abscess-negative patients. "
        "Minimum CSA, minimum airway cross-sectional area. Four airway features: minimum CSA, "
        "compactness, maxMLS, LCSA50.")
    doc.save(OUT / "Table4.docx")


# ======================================================================
# FIGURE 2 — Convergent + discriminant validity heatmap
# ======================================================================
def build_figure2():
    # Lookup keys must match 08_severity_compare_combined.csv exactly.
    # Four dichotomous axes (Mann-Whitney U, rank-biserial r) plus two continuous axes
    # (CRP, abscess diameter; Spearman rho). RPE volume is not shown (no panel data).
    feat_order = ["Volume (mL)", "Min CSA (mm²)", "Compactness",
                  "Max midline shift", "Length <50 mm²"]
    axes_order = ["ICU", "Prolonged LOS", "RPE+", "Mediastinal edema",
                  "CRP", "Abscess diam"]
    feat_labels = ["LV", "mCSA", "Compactness",
                   "maxMLS", "LCSA50"]
    axis_labels = ["ICU admission", "Prolonged stay", "RPE", "ME",
                   "CRP", "Abscess diameter"]
    sub = PANEL[PANEL["Cohort"] == "Full cohort"]

    eff = np.full((len(feat_order), len(axes_order)), np.nan)
    pmat = np.full_like(eff, np.nan)
    for i, f in enumerate(feat_order):
        for j, a in enumerate(axes_order):
            row = sub[(sub["Airway feature"] == f) & (sub["Severity axis"] == a)]
            if not len(row): continue
            eff[i, j] = float(row.iloc[0]["Effect"])
            pmat[i, j] = float(row.iloc[0]["p"])
    if np.isnan(eff).any():
        missing = [(feat_order[i], axes_order[j])
                   for i in range(len(feat_order)) for j in range(len(axes_order))
                   if np.isnan(eff[i, j])]
        raise ValueError(f"Figure 2: unmatched PANEL cells (check label keys): {missing}")

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    vmax = max(abs(np.nanmin(eff)), abs(np.nanmax(eff)))
    im = ax.imshow(eff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(axes_order))); ax.set_yticks(range(len(feat_order)))
    ax.set_xticklabels(axis_labels, rotation=30, ha="right")
    ax.set_yticklabels(feat_labels)
    for i in range(len(feat_order)):
        for j in range(len(axes_order)):
            sig = "*" if pmat[i, j] < 0.05 else ""
            text_color = "white" if abs(eff[i, j]) > 0.5 * vmax else "black"
            ax.text(j, i - 0.15, f"{eff[i, j]:+.2f}", ha="center", va="center",
                    color=text_color, fontsize=9)
            if sig:
                ax.text(j, i + 0.22, sig, ha="center", va="center",
                        color=text_color, fontsize=13, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Effect size  (rank-biserial r or Spearman ρ)", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Pre-specified airway feature")
    plt.tight_layout()
    fig.savefig(OUT / "Figure2.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure2.pdf",          bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# FIGURE 3 — 5-panel features by ICU
# ======================================================================
def build_figure3():
    panels = [
        ("volume_ml",          "Airway volume (mL)"),
        ("min_area_mm2",       "Min CSA (mm²)"),
        ("compactness",        "Compactness"),
        ("rl_dev_max_abs_mm",  "maxMLS (mm)"),
        ("len_below_50mm2_mm", "Length <50 mm² (mm)"),
    ]
    df = MERGED.dropna(subset=["icu_binary"])
    fig, axs = plt.subplots(1, 5, figsize=(13, 4.0))
    for ax, (key, label) in zip(axs, panels):
        sub = df[[key, "icu_binary"]].dropna()
        groups = [sub.loc[sub.icu_binary == 0, key].values,
                  sub.loc[sub.icu_binary == 1, key].values]
        bp = ax.boxplot(groups, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.2),
                        boxprops=dict(facecolor="#cfd8dc", edgecolor="black"))
        # strip jitter
        rng = np.random.default_rng(0)
        for i, vals in enumerate(groups, start=1):
            x = rng.normal(i, 0.07, size=len(vals))
            ax.scatter(x, vals, s=8, alpha=0.45, color="#37474f", edgecolor="none")
        # significance
        _, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        ymax = max(np.percentile(groups[0], 95), np.percentile(groups[1], 95))
        ymin = min(np.min(groups[0]), np.min(groups[1]))
        ax.set_ylim(ymin - 0.03 * (ymax - ymin + 1), ymax * 1.18 + 0.5)
        ax.text(1.5, ymax * 1.10 + 0.4, f"p = {fmt_p(p)}", ha="center", fontsize=10)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Non-ICU", "ICU"], fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT / "Figure3.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure3.pdf",          bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# FIGURE 4 — Inter-observer scatter (5 panels)
# ======================================================================
def build_figure4():
    panels = [
        ("volume_ml",          "Airway volume (mL)"),
        ("min_area_mm2",       "Min CSA (mm²)"),
        ("compactness",        "Compactness"),
        ("rl_dev_max_abs_mm",  "maxMLS (mm)"),
        ("len_below_50mm2_mm", "Length <50 mm² (mm)"),
    ]
    icc_idx = IRCC.set_index("feature")
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, (key, label) in zip(axs, panels):
        jcol, vcol = f"expert_{key}", f"reference_{key}"
        if jcol not in IRPC.columns or vcol not in IRPC.columns:
            continue
        j = IRPC[jcol].values; v = IRPC[vcol].values
        lo = min(np.min(j), np.min(v)); hi = max(np.max(j), np.max(v))
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
        ax.scatter(j, v, s=30, color="#1565c0", edgecolor="white", linewidth=0.5)
        r = icc_idx.loc[key]
        icc_str = f"ICC = {float(r['ICC']):.2f}\n[{float(r['CI_lower']):.2f}, {float(r['CI_upper']):.2f}]"
        ax.text(0.05, 0.95, icc_str, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bbb", alpha=0.95))
        ax.set_xlabel("Reader 1", fontsize=10)
        ax.set_ylabel("Reader 2", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlim(lo - 0.05 * (hi - lo + 0.1), hi + 0.05 * (hi - lo + 0.1))
        ax.set_ylim(lo - 0.05 * (hi - lo + 0.1), hi + 0.05 * (hi - lo + 0.1))
        ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT / "Figure4.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure4.pdf",          bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# FIGURE 5 — Joint stratification (abs sphericity × airway compactness)
# Two candidate visualisations; user picks one.
# ======================================================================
def _joint_cells():
    js = pd.read_csv(FA / "15_joint_strata.csv")
    # Order: Round+OK | Round+Bad | Irregular+OK | Irregular+Bad
    cells = []
    for round_ok, airway_ok, lab_short, lab_long in [
        (True,  True,  "Round\n+ OK airway",         "Round abscess, preserved airway"),
        (True,  False, "Round\n+ poor airway",       "Round abscess, poor airway compactness"),
        (False, True,  "Irregular\n+ OK airway",     "Irregular abscess, preserved airway"),
        (False, False, "Irregular\n+ poor airway",   "Irregular abscess, poor airway compactness"),
    ]:
        row = js[(js["abs_round"] == round_ok) & (js["airway_ok"] == airway_ok)].iloc[0]
        cells.append({"short": lab_short, "long": lab_long,
                      "n": int(row["n"]), "icu": int(row["icu_n"]),
                      "pct": float(row["icu_pct"]),
                      "round": round_ok, "airway_ok": airway_ok})
    return cells, js


def build_figure5():
    """Figure 5 — three-step risk stratification (small / large+good airway /
       large+poor airway) × two panels (ICU / prolonged stay).
       Replicates abscess paper Figure 4 layout and colors (Brewer Paired).
    """
    strata = pd.read_csv(FA / "15_risk_strata.csv")
    # Fixed display order
    order = ["Small (<40 mm)",
             "Large (≥40 mm) + good compactness",
             "Large (≥40 mm) + poor compactness"]
    short = ["Small\n(<40 mm)", "Large +\nGood airway", "Large +\nPoor airway"]
    fills = ["#A6CEE3", "#FB9A99", "#E31A1C"]   # abscess-paper palette

    def panel(ax, kind):
        if kind == "ICU":
            ax.set_title("ICU Admission", fontsize=12, fontweight="bold", pad=10)
            counts = [(int(strata.loc[strata.group == g, "icu_n"].iloc[0]),
                       int(strata.loc[strata.group == g, "n"].iloc[0]),
                       float(strata.loc[strata.group == g, "icu_pct"].iloc[0]))
                      for g in order]
        else:
            ax.set_title("Prolonged Hospital Stay", fontsize=12, fontweight="bold", pad=10)
            counts = [(int(strata.loc[strata.group == g, "los_n"].iloc[0]),
                       int(strata.loc[strata.group == g, "los_eval"].iloc[0]),
                       float(strata.loc[strata.group == g, "los_pct"].iloc[0]))
                      for g in order]
        rates = [c[2] for c in counts]
        ax.bar(range(len(order)), rates, color=fills, edgecolor="black",
               linewidth=0.4, width=0.7)
        for i, (ev, n, pct) in enumerate(counts):
            ax.text(i, pct + 1.5, f"{pct:.1f}%\n({ev}/{n})",
                    ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(short, fontsize=10)
        ax.set_ylim(0, 65)
        ax.set_xlim(-0.55, len(order) - 0.45)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", labelsize=10)

    fig, axs = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    panel(axs[0], "ICU")
    panel(axs[1], "Prolonged")
    axs[0].set_ylabel("Rate (%)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "Figure5.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure5.pdf",          bbox_inches="tight")
    plt.close(fig)


def build_figure6():
    """Figure 6 — 4-bar joint stratification of large abscesses
       (abscess sphericity × airway compactness)."""
    cells, _ = _joint_cells()
    pcts = [c["pct"] for c in cells]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    cmap = plt.colormaps["RdYlGn_r"]
    colors = [cmap(0.15 + 0.7 * (p / max(pcts))) for p in pcts]
    ax.bar(range(len(cells)), pcts, color=colors,
           edgecolor="black", linewidth=0.4, width=0.7)
    for i, c in enumerate(cells):
        ax.text(i, c["pct"] + 1.5, f"{c['pct']:.1f}%\n({c['icu']}/{c['n']})",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c["short"] for c in cells], fontsize=10)
    ax.set_ylabel("ICU admission rate (%)", fontsize=12)
    ax.set_ylim(0, max(pcts) * 1.30 + 5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Joint stratification of large abscesses (≥40 mm, n=67)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "Figure6.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "Figure6.pdf",          bbox_inches="tight")
    plt.close(fig)


def build_supp_fig3():
    """Supplementary Figure 3 — 2×2 heatmap (alternative visualisation of Figure 5)."""
    cells, _ = _joint_cells()
    grid = np.zeros((2, 2))
    n_grid = np.zeros((2, 2), dtype=int)
    icu_grid = np.zeros((2, 2), dtype=int)
    label_grid = [["", ""], ["", ""]]
    # rows: row 0 = poor airway (top); row 1 = preserved airway (bottom)
    # cols: col 0 = irregular abscess (left); col 1 = round abscess (right)
    for c in cells:
        i = 0 if not c["airway_ok"] else 1
        j = 0 if not c["round"]    else 1
        grid[i, j] = c["pct"]
        n_grid[i, j] = c["n"]; icu_grid[i, j] = c["icu"]
        label_grid[i][j] = c["long"]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=max(grid.flatten()) * 1.05,
                   aspect="auto")
    for i in range(2):
        for j in range(2):
            txt_color = "white" if grid[i, j] > 35 else "black"
            ax.text(j, i - 0.15, f"{icu_grid[i, j]}/{n_grid[i, j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color=txt_color)
            ax.text(j, i + 0.15, f"ICU {grid[i, j]:.1f}%",
                    ha="center", va="center", fontsize=12, color=txt_color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Irregular abscess\n(sphericity < median)",
                        "Round abscess\n(sphericity ≥ median)"], fontsize=10)
    ax.set_yticklabels(["Poor airway compactness\n(< median)",
                        "Preserved airway compactness\n(≥ median)"], fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.04)
    cbar.set_label("ICU admission rate (%)", fontsize=10)
    ax.set_title("Joint stratification of large abscesses (≥40 mm, n=67)",
                 fontsize=11, pad=12)
    plt.tight_layout()
    fig.savefig(OUT / "SuppFigure3.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "SuppFigure3.pdf", bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Supplementary
# ======================================================================
def build_supp_table1():
    """Full descriptive morphometry (median, IQR, range) for all available features."""
    skip = {"case_id", "accession_number", "abs_positive", "icu_binary", "hospd",
            "hospd_prolonged", "rpe_present", "rpe_volume_ml", "rpe_volume_ml_raw",
            "medoed_binary", "crp", "age", "sex", "abs_diamax_mri", "abs_volume_mri",
            "abs_max3d_radiomics", "abs_sphericity_radiomics", "abs_compactness_radiomics",
            "abs_volume_radiomics_ml", "abs_surface_area_radiomics", "absloc"}
    feats = [c for c in MERGED.columns if c not in skip and pd.api.types.is_numeric_dtype(MERGED[c])]
    rows = []
    for f in feats:
        v = MERGED[f].dropna()
        rows.append([f, len(v),
                     f"{v.mean():.3f}", f"{v.std():.3f}",
                     f"{v.median():.3f}", f"{v.quantile(0.25):.3f}",
                     f"{v.quantile(0.75):.3f}",
                     f"{v.min():.3f}", f"{v.max():.3f}"])
    doc = Document(); setup_doc(doc)
    title(doc, "Supplementary Table 1. Descriptive statistics for all morphometric features (n = 322).")
    make_table(doc, ["Feature", "n", "Mean", "SD", "Median", "Q25", "Q75", "Min", "Max"], rows,
               col_widths_cm=[6, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    caption(doc, "Values rounded to three decimal places.")
    doc.save(OUT / "SuppTable1.docx")


def build_supp_table2():
    """Full convergent-validity panel: effect + p for full cohort, abs+, abs-."""
    doc = Document(); setup_doc(doc)
    title(doc, "Supplementary Table 2. Full convergent-validity panel across cohorts.")
    rows = []
    for _, r in PANEL.iterrows():
        eff_raw = r["Effect"]; p_raw = r["p"]; n_raw = r["n"]
        eff = "" if pd.isna(eff_raw) or str(eff_raw) == "" else f"{float(eff_raw):+.3f}"
        p = "" if pd.isna(p_raw) or str(p_raw) == "" else fmt_p(float(p_raw))
        n = "" if pd.isna(n_raw) or str(n_raw) == "" else str(int(float(n_raw)))
        rows.append([r["Cohort"], r["Airway feature"], r["Severity axis"], eff, p, n])
    make_table(doc, ["Cohort", "Feature", "Marker", "Effect", "p", "n"], rows,
               col_widths_cm=[3, 4, 4, 2, 2, 1.5])
    caption(doc,
        "Effect = rank-biserial r for binary markers and Spearman ρ for continuous markers. "
        "ICU, RPE+, mediastinal edema, and prolonged stay are binary; CRP, RPE volume, "
        "and abscess diameter are continuous.")
    doc.save(OUT / "SuppTable2.docx")


def build_supp_table3():
    """Odontogenic vs pharyngotonsillar airway features."""
    # Pull hammas + peritonsinf from the clinical master workbook
    mri = pd.read_excel(
        BASE / _CFG.get("clinical_master_xlsx_copy", "clinical_master.xlsx"),
        sheet_name="data")
    extras = mri[["ac", "hammas", "peritonsinf"]].rename(columns={"ac": "accession_number"})
    df = MERGED.drop(columns=["hammas"], errors="ignore").copy()
    df["accession_number"] = pd.to_numeric(df["accession_number"], errors="coerce")
    extras["accession_number"] = pd.to_numeric(extras["accession_number"], errors="coerce")
    df = df.merge(extras, on="accession_number", how="left")

    odon = df[(df.hammas == 1) & ((df.peritonsinf == 0) | df.peritonsinf.isna())]
    phar = df[(df.peritonsinf == 1) & ((df.hammas == 0) | df.hammas.isna())]
    feats = [("volume_ml",          "Airway volume, mL",                       1),
             ("min_area_mm2",       "Minimum cross-sectional area, mm²",       0),
             ("compactness",        "Airway compactness",                      3),
             ("rl_dev_max_abs_mm",  "maxMLS (mm)",               1),
             ("len_below_50mm2_mm", "Length of airway below 50 mm², mm",       0)]
    rows = []
    for key, lab, dp in feats:
        o = odon[key].dropna(); p_ = phar[key].dropna()
        u, p = stats.mannwhitneyu(o, p_, alternative="two-sided")
        r = 1 - 2 * u / (len(o) * len(p_))
        rows.append([lab, med_iqr(o, dp), med_iqr(p_, dp), fmt_p(p), f"{abs(r):.2f}"])

    icu_o = int((odon.icu_binary == 1).sum())
    icu_p = int((phar.icu_binary == 1).sum())
    _, p_fisher = stats.fisher_exact([[icu_o, len(odon) - icu_o],
                                       [icu_p, len(phar) - icu_p]])

    doc = Document(); setup_doc(doc)
    title(doc, "Supplementary Table 3. Airway morphometry by infection origin.")
    make_table(doc,
        ["Feature", f"Odontogenic (n = {len(odon)})",
         f"Pharyngotonsillar (n = {len(phar)})", "p value", "r"],
        rows,
        col_widths_cm=[6.5, 4.5, 4.5, 2, 1.5])
    caption(doc,
        f"Odontogenic etiology defined as clinical-database variable hammas = 1; pharyngotonsillar etiology "
        f"as peritonsinf = 1. The two groups are mutually exclusive in this cohort (no patient "
        f"was coded as both). Values are median (interquartile range). p, Mann–Whitney U test "
        f"(two-sided); r, absolute rank-biserial correlation. "
        f"ICU rates were similar between groups ({icu_o}/{len(odon)} [{100*icu_o/len(odon):.1f}%] "
        f"odontogenic vs {icu_p}/{len(phar)} [{100*icu_p/len(phar):.1f}%] pharyngotonsillar; "
        f"Fisher's exact p = {p_fisher:.3f}).")
    doc.save(OUT / "SuppTable3.docx")


def build_supp_fig1():
    """Copy the existing cross-cohort heatmap."""
    src = FA / "08_severity_compare_heatmap.png"
    if src.exists():
        import shutil
        shutil.copy(src, OUT / "SuppFigure1.png")
    src_pdf = FA / "08_severity_compare_heatmap.pdf"
    if src_pdf.exists():
        import shutil
        shutil.copy(src_pdf, OUT / "SuppFigure1.pdf")


def build_supp_fig2():
    """Bland–Altman for airway volume + min CSA inter-observer."""
    feats = [("volume_ml", "Airway volume (mL)"),
             ("min_area_mm2", "Minimum CSA (mm²)")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (key, label) in zip(axs, feats):
        jcol, vcol = f"expert_{key}", f"reference_{key}"
        j = IRPC[jcol].values; v = IRPC[vcol].values
        mean = (j + v) / 2
        diff = j - v
        md = np.mean(diff); sd = np.std(diff, ddof=1)
        ax.scatter(mean, diff, s=30, color="#1565c0", edgecolor="white")
        ax.axhline(md, color="black", lw=1)
        ax.axhline(md + 1.96 * sd, color="grey", lw=1, ls="--")
        ax.axhline(md - 1.96 * sd, color="grey", lw=1, ls="--")
        ax.set_xlabel(f"Mean of two readers — {label}")
        ax.set_ylabel(f"Reader 1 − Reader 2")
        ax.set_title(f"Bias = {md:+.2f}, 95% LoA [{md - 1.96*sd:+.2f}, {md + 1.96*sd:+.2f}]",
                     fontsize=10)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT / "SuppFigure2.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "SuppFigure2.pdf",          bbox_inches="tight")
    plt.close(fig)


# ======================================================================
def main():
    print("Building Table 1...")  ; build_table1()
    print("Building Table 2...")  ; build_table2()
    print("Building Table 3...")  ; build_table3()
    print("Building Table 4...")  ; build_table4()
    print("Building Figure 2..."); build_figure2()
    print("Building Figure 3..."); build_figure3()
    print("Building Figure 4..."); build_figure4()
    print("Building Figure 5 (3-step strat, abscess-paper style)..."); build_figure5()
    print("Building Figure 6 (4-bar joint stratification)...");        build_figure6()
    print("Building SuppFigure3 (heatmap)...");                        build_supp_fig3()
    print("Building SuppTable1..."); build_supp_table1()
    print("Building SuppTable2..."); build_supp_table2()
    print("Building SuppTable3..."); build_supp_table3()
    print("Building SuppFigure1..."); build_supp_fig1()
    print("Building SuppFigure2..."); build_supp_fig2()
    print(f"\nAll outputs in {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
