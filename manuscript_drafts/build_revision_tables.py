#!/usr/bin/env python3
"""
Revision R1 (EJR-D-26-04240): build the revised Tables 3 and 4 and the new
Supplementary Tables S1-S6 as Word files, every number read from the saved
analysis outputs in final_analysis/ (provenance rule).

  Table 3 (revised)  05_prespecified_icu_table_full.csv + 08 grid (AUC) + 29_fdr_grid.csv (q)
  Table 4 (revised)  32_penalized_icu_models.csv (Firth, optimism-corrected) + Table4_data.csv (DeLong)
  Table S1  29_fdr_grid.csv (30-test grid with FDR q)
  Table S2  36_fixed_score_reference.csv (published fixed score as reference)
  Table S3  30_lcsa_threshold_sensitivity.csv
  Table S4  34_failure_analysis.csv (HD95 > 5 mm or Dice < 0.85)
  Table S5  27_pred_postproc_report.txt (largest-component vs union bounds)
  Table S6  35_anatomical_midline_tests.csv + 31_midline_positioning_tests.csv (positioning)
Output: manuscript_drafts/revision_R1/*.docx
"""
import os, re
import json
from pathlib import Path
import numpy as np, pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_LINE_SPACING

BASE = Path(os.environ.get("AIRWAY_ROOT", Path(__file__).resolve().parents[1]))
_CFG_FILE = BASE / "paths_local.json"  # untracked: maps generic keys to the local folder layout (README, "Local layout")
_CFG = json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {}
FA = BASE / "final_analysis"; OUT = BASE / "manuscript_drafts" / _CFG.get("submission_dir", "submission") / "revision_R1"; OUT.mkdir(exist_ok=True)
BLACK = RGBColor(0, 0, 0)
STRICT = True   # reviewer-requested content only (no bootstrap CIs for r in Table 3, no calibration column in Table 4)


def setup(doc):
    for s in doc.sections:
        s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Cm(2)
    n = doc.styles["Normal"]; n.font.name = "Times New Roman"; n.font.size = Pt(12); n.font.color.rgb = BLACK
    n.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE; n.paragraph_format.space_after = Pt(0)


def run(p, t, bold=False, italic=False, size=12):
    r = p.add_run(t); r.font.name = "Times New Roman"; r.font.size = Pt(size); r.font.color.rgb = BLACK; r.font.bold = bold; r.font.italic = italic
    return r


def title(doc, t):
    p = doc.add_paragraph(); run(p, t, bold=True); p.paragraph_format.space_after = Pt(4)


def note(doc, t):
    p = doc.add_paragraph(); run(p, t, size=10); p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(12)


def table(doc, headers, rows, bold_rows=(), indent_rows=(), size=10):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers)); tbl.style = "Table Grid"
    for j, h in enumerate(headers):
        c = tbl.rows[0].cells[j]; c.text = ""; run(c.paragraphs[0], h, bold=True, size=size)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = tbl.rows[i + 1].cells[j]; c.text = ""
            run(c.paragraphs[0], ("    " if (i in indent_rows and j == 0) else "") + str(v), bold=(i in bold_rows), size=size)
    return tbl


def fp(p):
    p = float(p)
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fq(q):
    q = float(q)
    return "<0.001" if q < 0.001 else f"{q:.3f}"


# ---------------------------------------------------------------- Table 3 revised
def table3():
    t3 = pd.read_csv(FA / "05_prespecified_icu_table_full.csv")
    grid = pd.read_csv(FA / "29_fdr_grid.csv"); g = grid[grid.Cohort == "Full cohort"]
    panel = pd.read_csv(FA / "08_severity_compare_combined.csv"); panel = panel[panel.Cohort == "Full cohort"]
    m = pd.read_csv(FA / "airway_morphology_merged.csv")
    from sklearn.metrics import roc_auc_score
    feats = [("Airway volume (mL)", "Volume (mL)", "volume_ml", 1), ("Minimum cross-sectional area (mm²)", "Min CSA (mm²)", "min_area_mm2", 0),
             ("Airway compactness", "Compactness", "compactness", 3), ("Maximum lateral midline shift (mm)", "Max midline shift", "rl_dev_max_abs_mm", 1),
             ("Length below 50 mm² (mm)", "Length <50 mm²", "len_below_50mm2_mm", 0)]
    names = {"Airway volume (mL)": "Airway volume, mL", "Minimum cross-sectional area (mm²)": "Minimum cross-sectional area, mm²", "Airway compactness": "Airway compactness",
             "Maximum lateral midline shift (mm)": "Maximum midline shift, mm", "Length below 50 mm² (mm)": "Length of airway below 50 mm², mm"}
    from scipy.stats import mannwhitneyu
    rng = np.random.RandomState(42)
    def rb_ci(x, y, B=2000):
        a, b = x[y == 0], x[y == 1]
        def rb(a, b):
            u, _ = mannwhitneyu(b, a, alternative="two-sided"); return 2 * u / (len(a) * len(b)) - 1
        est = abs(rb(a, b)); boots = []
        for _ in range(B):
            aa = rng.choice(a, len(a)); bb = rng.choice(b, len(b)); boots.append(abs(rb(aa, bb)))
        return est, np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    rows = []; bold = []
    rows.append(["ICU admission", "(n = 33)", "(n = 289)", "", "", "", ""]); bold.append(0)
    for lab, glab, col, dp in feats:
        r = t3[t3.Feature == lab].iloc[0]; gr = g[(g["Airway feature"] == glab) & (g["Severity axis"] == "ICU")].iloc[0]
        y = m.icu_binary.astype(int); auc = roc_auc_score(y, m[col]); auc = max(auc, 1 - auc)
        est, lo, hi = rb_ci(m[col].values, y.values)
        rows.append([names[lab], r["ICU (n=33)"].replace("-", "–"), r["No-ICU (n=289)"].replace("-", "–"), fp(gr.p), fq(gr.q_BH_30), f"{est:.2f}" if STRICT else f"{est:.2f} ({lo:.2f}–{hi:.2f})", f"{auc:.3f}"])
    rows.append(["Prolonged hospitalization", "(n = 47)", "(n = 272)", "", "", "", ""]); bold.append(len(rows) - 1)
    ml = m.dropna(subset=["hospd_prolonged"])
    for lab, glab, col, dp in feats:
        gr = g[(g["Airway feature"] == glab) & (g["Severity axis"] == "Prolonged LOS")].iloc[0]
        a = ml[ml.hospd_prolonged == 1][col]; b = ml[ml.hospd_prolonged == 0][col]
        def iqr(s):
            q1, md, q3 = np.percentile(s, [25, 50, 75]); f = (lambda x: f"{x:.0f}") if dp == 0 else (lambda x: f"{x:.{dp}f}")
            return f"{f(md)} ({f(q1)}–{f(q3)})"
        auc = roc_auc_score(ml.hospd_prolonged.astype(int), ml[col]); auc = max(auc, 1 - auc)
        est, lo, hi = rb_ci(ml[col].values, ml.hospd_prolonged.astype(int).values)
        rows.append([names[lab], iqr(a), iqr(b), fp(gr.p), fq(gr.q_BH_30), f"{est:.2f}" if STRICT else f"{est:.2f} ({lo:.2f}–{hi:.2f})", f"{auc:.3f}"])
    doc = Document(); setup(doc)
    title(doc, "Table 3. Pre-specified airway features by ICU admission and prolonged hospitalization.")
    table(doc, ["Feature", "Positive", "Negative", "p value", "q value (FDR)", "r" if STRICT else "r (95% CI)", "AUC"], rows, bold_rows=bold)
    note(doc, "Values are median (interquartile range). p, Mann–Whitney U test (two-sided). q value (FDR), false-discovery-rate adjusted across the 30 tests of the "
              "five features by six severity markers grid (Supplementary Table S1). r, absolute rank-biserial correlation" + ("" if STRICT else " with 95% confidence interval from a case-resampling bootstrap (2000 replicates)") + ". "
              "AUC, area under the receiver operating characteristic curve for each feature alone. Prolonged hospitalization is defined as hospital stay >5 days (available for 319 patients). Cohort: full study population (n = 322).")
    doc.save(OUT / "Table3_revised.docx")


# ---------------------------------------------------------------- Table 4 revised (Firth)
def table4():
    f = pd.read_csv(FA / "32_penalized_icu_models.csv"); d = pd.read_csv(BASE / "manuscript_drafts" / "tables_figures" / "Table4_data.csv")
    f40 = pd.read_csv(FA / "40_selection_aware_validation.csv").set_index("cohort")   # selection-aware optimism (univariate feature selection repeated per resample)
    txt = (FA / "32_penalized_icu_models.txt").read_text()
    lrt = {}
    for coh in ["Full cohort", "Abscess-positive subset"]:
        blk = txt.split(coh + ":")[1].split("\n\n")[0] if coh in txt else ""
        for mname in ["Reference + minimum CSA", "Reference + four airway features"]:
            blk = txt.split("Full cohort:")[1].split("Abscess-positive subset:")[0] if coh == "Full cohort" else txt.split("Abscess-positive subset:")[1]
            mm = re.search(r"Penalized LRT: " + re.escape(mname) + r" vs reference: chi2=([\d.]+) df=(\d+) p=([\d.]+)", blk)
            dl = re.search(r"DeLong \(Firth-fitted probabilities\): " + re.escape(mname) + r" vs reference p=([\d.]+)", blk)
            lrt[(coh, mname)] = (float(mm.group(1)), int(mm.group(2)), float(mm.group(3)), float(dl.group(1)))
    rows = []; bold = []; ind = []
    for coh, dcoh in [("Full cohort", "Full cohort"), ("Abscess-positive subset", "Abscess-positive subset")]:
        sub = f[f.cohort == coh]; dsub = d[d.cohort == dcoh]
        n = int(sub.n.iloc[0]); ev = int(sub.events.iloc[0])
        rows.append([f"{coh} (n = {n}, ICU = {ev})", "", "", "", "", "", ""] + ([] if STRICT else [""])); bold.append(len(rows) - 1)
        ref = sub[sub.model == "Reference (Turku ICU-risk model)"].iloc[0]
        rows.append(["Reference model (CRP, RPE, abscess diameter; refitted)", f"{int(ref.k)} ({ref.EPV:.0f})", f"{ref.AUC_apparent:.3f} ({ref.CI_lo:.3f}–{ref.CI_hi:.3f})", f"{ref.AUC_corrected:.3f}"] + ([] if STRICT else [f"{ref.cal_slope_validated:.2f}"]) + ["—", "—", "—"]); ind.append(len(rows) - 1)
        for mname in ["Reference + minimum CSA", "Reference + four airway features"]:
            r = sub[sub.model == mname].iloc[0]; chi, df_, p, pdl = lrt[(coh, mname)]
            corr = f40.loc[coh, "AUC_corrected"] if mname == "Reference + four airway features" else r.AUC_corrected   # 27 Oct: selection-aware value is the principal one for the selected model
            rows.append([mname, f"{int(r.k)} ({r.EPV:.1f})", f"{r.AUC_apparent:.3f} ({r.CI_lo:.3f}–{r.CI_hi:.3f})", f"{corr:.3f}"] + ([] if STRICT else [f"{r.cal_slope_validated:.2f}"]) + [
                         f"+{r.AUC_apparent - ref.AUC_apparent:.3f} / +{corr - ref.AUC_corrected:.3f}", f"{chi:.1f} ({df_}), {fp(p)}", f"{pdl:.3f}"]); ind.append(len(rows) - 1)
    doc = Document(); setup(doc)
    title(doc, "Table 4. Incremental value of airway features for ICU prediction (penalized logistic regression).")
    table(doc, ["Model", "Predictors (events per predictor)", "Apparent AUC (95% CI)", "Optimism-corrected AUC"] + ([] if STRICT else ["Bootstrap-validated calibration slope"]) + ["ΔAUC vs reference, apparent / corrected", "Penalized likelihood-ratio test, χ² (df), p", "DeLong p"], rows, bold_rows=bold, indent_rows=ind, size=8)
    note(doc, "Firth's bias-reduced penalized logistic regression on standardized predictors (R package logistf). Apparent AUC with DeLong 95% confidence intervals; optimism-corrected AUC" + ("" if STRICT else " and calibration slope") + " by "
              "bootstrap (500 resamples, Harrell's method" + ("" if STRICT else "; a slope below 1 indicates overfitting") + "). The reference model uses the predictors previously identified in this cohort (C-reactive protein, retropharyngeal-edema presence, "
              "maximum abscess diameter set to zero in abscess-negative patients) as continuous variables with coefficients refitted here; it is therefore not the published risk score, which is applied unchanged in Supplementary Table S2. Nested models were compared by the penalized likelihood-ratio test; DeLong's test for correlated "
              "AUCs (computed on the penalized model predictions) is retained for comparability but is conservative for nested models. Minimum CSA, minimum airway cross-sectional area. Four airway features: minimum CSA, compactness, "
              "maximum midline shift, length of airway below 50 mm². Minimum CSA was examined alone as the primary caliber feature; the four-feature model comprises the features univariately associated with ICU admission, an outcome-based selection, so its optimism-corrected AUC repeats that selection within each bootstrap resample; with the model treated as fixed the corrected AUC would be "
              + f"{f[(f.cohort == 'Full cohort') & (f.model == 'Reference + four airway features')].AUC_corrected.iloc[0]:.3f} in the full cohort and {f[(f.cohort == 'Abscess-positive subset') & (f.model == 'Reference + four airway features')].AUC_corrected.iloc[0]:.3f} in the abscess-positive subset.")
    doc.save(OUT / "Table4_revised.docx")


# ---------------------------------------------------------------- S1 FDR grid
def s1():
    g = pd.read_csv(FA / "29_fdr_grid.csv"); g = g[g.Cohort == "Full cohort"].sort_values("p")
    axn = {"ICU": "ICU admission", "Prolonged LOS": "Prolonged stay", "RPE+": "Retropharyngeal edema", "ME+": "Mediastinal edema", "CRP": "CRP", "Abscess diam": "Abscess diameter"}
    fn = {"Volume (mL)": "Airway volume", "Min CSA (mm²)": "Minimum CSA", "Compactness": "Compactness", "Max midline shift": "Maximum midline shift", "Length <50 mm²": "LCSA50"}
    rows = [[fn[r["Airway feature"]], axn[r["Severity axis"]], int(r.n), f"{r.Effect:+.2f}", fp(r.p), fq(r.q_BH_30),
             "yes" if r.q_BH_30 < 0.05 else "no"] for _, r in g.iterrows()]
    doc = Document(); setup(doc)
    title(doc, "Supplementary Table S1. All 30 feature-by-marker tests in the full cohort with false-discovery-rate control.")
    table(doc, ["Airway feature", "Severity marker", "n", "Effect size", "p value", "q value (FDR)", "Significant at q < 0.05"], rows)
    note(doc, "Effect size: rank-biserial correlation (binary markers, Mann–Whitney U test) or Spearman rank correlation (CRP, abscess diameter), positive when the feature is larger with the adverse marker. "
              f"q value (FDR): false-discovery-rate adjustment (Benjamini–Hochberg) across the 30 tests; {int((g.q_BH_30<0.05).sum())} of 14 nominally significant associations remain significant at q < 0.05. Sorted by p value. "
              "CSA, cross-sectional area; FDR, false discovery rate; LCSA50, length of airway below 50 mm².")
    doc.save(OUT / "SuppTable_S1_FDR_grid.docx")


# ---------------------------------------------------------------- S2 fixed score
def s2():
    f = pd.read_csv(FA / "36_fixed_score_reference.csv")
    rows = []; bold = []; ind = []
    for coh in ["Full cohort", "Abscess-positive subset"]:
        sub = f[f.cohort == coh]; n = int(sub.n.iloc[0]); ev = int(sub.events.iloc[0])
        rows.append([f"{coh} (n = {n}, ICU = {ev})", "", "", "", "", ""]); bold.append(len(rows) - 1)
        ref = sub.iloc[0]
        for _, r in sub.iterrows():
            rows.append([r.model, f"{r.AUC_apparent:.3f} ({r.CI_lo:.3f}–{r.CI_hi:.3f})", f"{r.AUC_corrected:.3f}" if r.model != "Fixed score alone" else "—",
                         "—" if r.model == "Fixed score alone" else f"+{r.AUC_apparent-ref.AUC_apparent:.3f} / +{r.AUC_corrected-ref.AUC_apparent:.3f}",
                         "—" if pd.isna(r.p_LRT_vs_score) else fp(r.p_LRT_vs_score), "—" if pd.isna(r.p_delong_vs_score) else f"{r.p_delong_vs_score:.3f}"]); ind.append(len(rows) - 1)
    doc = Document(); setup(doc)
    title(doc, "Supplementary Table S2. Incremental value of airway features over the published ICU risk score applied with its original coefficients.")
    table(doc, ["Model", "Apparent AUC (95% CI)", "Optimism-corrected AUC", "ΔAUC vs score, apparent / corrected", "Penalized likelihood-ratio test p", "DeLong p"], rows, bold_rows=bold, indent_rows=ind)
    note(doc, "Published score (3 × retropharyngeal edema + 2 × CRP > 171.5 mg/L + 2 × abscess diameter > 39.5 mm; range 0–7; Vierula et al. 2025) applied without refitting; AUC 0.810 in the 320 complete cases versus 0.82 in its 535-patient derivation cohort. "
              "Airway features were added to the fixed score in Firth penalized logistic models; optimism-corrected AUC from 500 bootstrap resamples. Four airway features: minimum CSA, compactness, maximum midline shift, LCSA50. "
              "CSA, cross-sectional area; LCSA50, length of airway below 50 mm².")
    doc.save(OUT / "SuppTable_S2_fixed_score.docx")


# ---------------------------------------------------------------- S3 LCSA thresholds
def s3():
    t = pd.read_csv(FA / "30_lcsa_threshold_sensitivity.csv")
    lab = {"len_below_25mm2_mm": "Length below 25 mm²", "len_below_40mm2_mm": "Length below 40 mm²", "len_below_50mm2_mm": "Length below 50 mm² (LCSA50, pre-specified)",
           "len_below_60mm2_mm": "Length below 60 mm²", "len_below_75mm2_mm": "Length below 75 mm²", "len_below_100mm2_mm": "Length below 100 mm²",
           "len_below_25pct_max_mm": "Length below 25% of the patient's maximum CSA", "len_below_33pct_max_mm": "Length below 33% of the patient's maximum CSA", "len_below_50pct_max_mm": "Length below 50% of the patient's maximum CSA"}
    rows = []; bold = []
    for coh in ["Full cohort", "Abscess-positive"]:
        sub = t[t.cohort == coh]; rows.append([f"{coh} (n = {int(sub.n.iloc[0])}, ICU = {int(sub.icu.iloc[0])})", "", "", "", "", "", ""]); bold.append(len(rows) - 1)
        for _, r in sub.iterrows():
            rows.append([lab[r.feature], f"{r.med_noicu:.1f}", f"{r.med_icu:.1f}", fp(r.p), f"{r.r_rb:+.2f}", f"{r.auc:.3f}", f"{r.pct_zero:.0f}"])
    doc = Document(); setup(doc)
    title(doc, "Supplementary Table S3. Sensitivity of the airway-narrowing length feature to the caliber threshold.")
    table(doc, ["Threshold definition", "Median, no ICU (mm)", "Median, ICU (mm)", "p value", "r", "AUC", "Patients with zero length (%)"], rows, bold_rows=bold)
    note(doc, "Feature: longest contiguous run of axial slices with cross-sectional area below the threshold, computed from the manual segmentations as for LCSA50; absolute thresholds of 25 to 100 mm² and relative thresholds as a fraction of each patient's maximum cross-sectional area. "
              "p, Mann–Whitney U test; r, rank-biserial correlation (positive = longer in ICU patients); AUC, area under the receiver operating characteristic curve. CSA, cross-sectional area.")
    doc.save(OUT / "SuppTable_S3_LCSA_thresholds.docx")


# ---------------------------------------------------------------- former S4 risk stratification (removed from the revised manuscript
# at R1, reviewer point 7; the numbers are reported in the response letter only; kept for the record, not called)
def s_risk_strata_removed():
    v = pd.read_csv(FA / "33_risk_strat_validation.csv"); txt = (FA / "33_risk_strat_validation.txt").read_text()
    ms = v[v.analysis == "median_split"]; cont = v[v.analysis == "continuous_MWU"]
    rows = []
    for _, r in ms.iterrows():
        c = cont[cont.feature == r.feature].iloc[0]
        rows.append([r.feature, f"{r['median']:.3f}" if r.feature == "Compactness" else f"{r['median']:.1f}", f"{int(r.icu_adverse)}/{int(r.n_adverse)} ({100*r.icu_adverse/r.n_adverse:.1f})",
                     f"{int(r.icu_other)}/{int(r.n_other)} ({100*r.icu_other/r.n_other:.1f})", f"{r.RR:.2f}", f"{r.p:.3f}", f"{r.p_bonf5:.3f}", f"{c.p:.3f}", f"{abs(c.r_rb):.2f}", f"{c.auc:.3f}"])
    cv = re.search(r"RR \(poor vs preserved\) median ([\d.]+) \[IQR ([\d.]+)-([\d.]+)\], range ([\d.]+)-([\d.]+)", txt); pc = re.search(r"repeats with p<0.05: (\d+)%", txt)
    sa = re.search(r"feature chosen in training folds: compactness (\d+)%[\s\S]*?pooled held-out RR \(adverse vs other half\) median ([\d.]+) \[IQR [\d.]+-[\d.]+\], range ([\d.]+)-([\d.]+); Fisher p median ([\d.]+); repeats with p<0.05: (\d+)%", txt)
    bs = re.search(r"RR 95% percentile CI ([\d.]+)-([\d.]+)", txt); ls = re.search(r"poor (\d+)/(\d+) \(([\d.]+)%\) vs preserved (\d+)/(\d+) \(([\d.]+)%\), RR=([\d.]+), Fisher p=([\d.]+)", txt)
    doc = Document(); setup(doc)
    title(doc, "Supplementary Table S4. Exploratory median split of each pre-specified airway feature within large abscesses (≥ 40 mm, n = 68, ICU admissions = 18).")
    table(doc, ["Feature", "Median (split value)", "ICU, adverse half, n/N (%)", "ICU, other half, n/N (%)", "Relative risk", "Fisher p", "p × 5 (Bonferroni)", "Continuous Mann–Whitney p", "r", "AUC"], rows, size=9)
    note(doc, "The adverse half is the half with lower volume, lower minimum CSA, lower compactness, greater midline shift, or longer LCSA50. The five splits were all tested; compactness was the "
              "only one with a nominally significant separation, and the comparison is reported as exploratory. Continuous comparison without dichotomization: Mann–Whitney U test of the feature between "
              f"ICU and non-ICU patients within large abscesses. Internal validation by repeated stratified five-fold cross-validation (200 repeats) in which the feature with the smallest Fisher p among the five median splits and its split value were chosen within the training folds and applied to the held-out fold: compactness was chosen in {sa.group(1)}% of training folds, pooled held-out relative risk median {sa.group(2)} (range {sa.group(3)} to {sa.group(4)}), Fisher p < 0.05 in {sa.group(6)}% of repeats. Cross-validation of the split value alone, with compactness fixed: relative risk median {cv.group(1)} (range {cv.group(4)} to {cv.group(5)}), Fisher p < 0.05 in {pc.group(1)}% of repeats. Bootstrap 95% percentile interval of the relative "
              f"risk {bs.group(1)} to {bs.group(2)} (2000 resamples with re-estimated split value). Prolonged hospital stay by the same compactness split: {ls.group(1)}/{ls.group(2)} ({ls.group(3)}%) versus "
              f"{ls.group(4)}/{ls.group(5)} ({ls.group(6)}%), relative risk {ls.group(7)}, Fisher p = {ls.group(8)}. CSA, cross-sectional area; LCSA50, length of airway below 50 mm².")
    doc.save(OUT / "SuppTable_S4_risk_strata.docx")


# ---------------------------------------------------------------- S4 failure analysis
def s4():
    t = pd.read_csv(FA / "34_failure_analysis.csv").sort_values(["hd95_mm", "dice"], ascending=[False, True]).reset_index(drop=True)
    t["interior"] = t.pred_zgap_slices - t.missing_slices_cranial - t.missing_slices_caudal   # empty slices strictly inside the predicted extent
    t["short"] = t.missing_slices_cranial + t.missing_slices_caudal                              # manual slices beyond the predicted extent
    def mech(r):
        c = r.category.split(";"); out = []
        if r.interior > 0: out.append(f"craniocaudal interruption of the predicted lumen ({int(r.interior)} empty slice{'s' if r.interior > 1 else ''}" + (", at a manually bridged pinpoint segment)" if any(x.startswith("ZGAP@pinpoint") for x in c) else ")"))
        ends = lambda cr, ca: ", ".join(x for x in ([f"{int(cr)} cranial"] if cr else []) + ([f"{int(ca)} caudal"] if ca else []))
        if r["short"] > 0: out.append(f"prediction {int(r['short'])} slice{'s' if r['short'] > 1 else ''} shorter than the manual extent ({ends(r.missing_slices_cranial, r.missing_slices_caudal)})")
        if "ENDPOINT-extra" in c: n_ex = int(r.extra_slices_cranial + r.extra_slices_caudal); out.append(f"{n_ex} additional end slice{'s' if n_ex > 1 else ''} ({ends(r.extra_slices_cranial, r.extra_slices_caudal)})")
        if "STRAY>200vox" in c: out.append(f"separate false-positive component(s) ({int(r.extra_component_vox)} voxels)")
        if "OVER" in c: out.append("over-segmentation")
        if "UNDER" in c: out.append("under-segmentation of a very small lumen")
        if "BOUNDARY" in c: out.append("in-plane boundary disagreement")
        return "; ".join(out)
    rows = [[i + 1, f"{r.dice:.2f}", f"{r.hd95_mm:.1f}", int(r.n_comp), f"{r.manual_volume_ml:.1f} / {r.pred_volume_ml:.1f}", f"{r.manual_minCSA:.0f} / {r.pred_minCSA:.0f}",
             "yes" if r.ICU else "no", (f"{r.abs_diam_mm:.0f}" if r.abscess else "none"), mech(r)] for i, r in t.iterrows()]
    doc = Document(); setup(doc)
    title(doc, f"Supplementary Table S4. Per-case analysis of the {len(t)} out-of-fold predictions with HD95 > 5 mm or Dice < 0.85.")
    pA = doc.add_paragraph(); run(pA, "A. Network predictions", bold=True)
    table(doc, ["No.", "Dice", "HD95 (mm)", "Predicted components", "Volume manual / predicted (mL)", "Minimum CSA manual / predicted (mm²)", "ICU", "Abscess diameter (mm)", "Automatically identified mechanism"], rows, size=8)
    nint = int((t.interior > 0).sum()); npin = int(t.category.str.contains("@pinpoint").sum()); nshort = int((t["short"] > 0).sum())
    ne = int(t.category.str.contains("ENDPOINT-extra").sum()); ns = int(t.category.str.contains("STRAY").sum()); no = int(t.category.str.contains("OVER").sum()); nu = int(t.category.str.contains("UNDER").sum())
    note(doc, f"Sorted by HD95; HD95 > 5 mm in 19 of 322 cases, five of them at 5.0 to 5.1 mm (one slice). Mechanisms were derived automatically from the mask pair and more than one may apply: "
              f"craniocaudal interruption of the predicted lumen in {nint} cases (coinciding with a manually bridged pinpoint segment in {npin}), prediction shorter than the manual extent in {nshort}, additional end slices in {ne}, "
              f"a separate false-positive component in {ns}, over-segmentation in {no}, and under-segmentation of a very small lumen in {nu}. No. 1 is the single gross failure, a large false-positive component outside the pharynx. "
              "HD95, 95th-percentile Hausdorff distance; CSA, cross-sectional area.")
    r39 = pd.read_csv(FA / "39_reader_outliers.csv"); letters = {c: chr(65 + i) for i, c in enumerate(dict.fromkeys(r39.case))}
    pB = doc.add_paragraph(); pB.paragraph_format.space_before = Pt(10); run(pB, "B. Human-reader cases with Dice < 0.85 in the 30-case interobserver substudy (Table 2)", bold=True)
    rowsB = [[r.reader, letters[r.case], f"{r.dice:.2f}", f"{r.hd95_mm:.1f}", f"{int(r.slice_diff):+d}", f"{r.ref_volume_ml:.1f} / {r.reader_volume_ml:.1f}", f"{r.ref_minCSA:.0f} / {r.reader_minCSA:.0f}", f"{r.ref_LCSA50:.0f} / {r.reader_LCSA50:.0f}", f"{r.ref_maxMLS:.1f} / {r.reader_maxMLS:.1f}", r.mechanism] for _, r in r39.iterrows()]
    table(doc, ["Reader", "Patient", "Dice", "HD95 (mm)", "Slice-count difference", "Volume reference / reader (mL)", "Minimum CSA reference / reader (mm²)", "LCSA50 reference / reader (mm)", "Maximum midline shift reference / reader (mm)", "Mechanism"], rowsB, size=8)
    note(doc, "Patients are lettered consistently across readers. Slice-count difference is positive when the reader segmented more slices than the reference annotator. The novice reader's lowest Dice values reflect under-segmentation of small lumina and a shorter craniocaudal extent, which lower volume and minimum CSA and lengthen LCSA50, whereas maximum midline shift is almost unaffected. "
              "CSA, cross-sectional area; HD95, 95th-percentile Hausdorff distance; LCSA50, length of airway below 50 mm².")
    doc.save(OUT / "SuppTable_S4_failure_cases.docx")


# ---------------------------------------------------------------- S5 union vs largest-CC
def s5():
    c = pd.read_csv(FA / "38_conventions_icc.csv").set_index("feature"); t38 = (FA / "38_gap_convention.txt").read_text()
    rows = []
    for lab in ["Airway volume", "Minimum CSA", "Compactness", "Maximum midline shift", "LCSA50"]:
        r = c.loc[lab]
        rows.append([lab, f"{r.ICC_largestCC:.3f} ({r.lo1:.2f}–{r.hi1:.2f})", f"{r.ICC_union:.3f} ({r.lo2:.2f}–{r.hi2:.2f})", f"{r.ICC_gapzero:.3f} ({r.lo3:.2f}–{r.hi3:.2f})" if pd.notna(r.ICC_gapzero) else "not applicable"])
    doc = Document(); setup(doc)
    title(doc, "Supplementary Table S5. Manual-versus-predicted feature agreement under alternative evaluation conventions for the predicted masks (n = 322).")
    table(doc, ["Feature", "ICC(2,1), predicted mask reduced to its largest connected component (95% CI)", "ICC(2,1), predicted mask evaluated as the union of all components (95% CI), as reported", "ICC(2,1), union with empty slices inside the predicted range counted as zero area (95% CI)"], rows)
    gap = re.search(r"predictions with an interior empty slice: (\d+) cases; gap length median (\d+) slices, max (\d+)", t38)
    pin2 = re.search(r"manual mask has a pinpoint slice in (\d+) cases", t38).group(1)
    note(doc, "107 of 322 predictions had more than one connected component: 48 with a craniocaudal discontinuity of the predicted lumen, 55 with small additional components lateral to the lumen, and 4 with gross over-segmentation. "
              f"Empty slices inside the predicted range occurred in {gap.group(1)} predictions (median {gap.group(2)} slice, maximum {gap.group(3)}), coinciding with a manually bridged pinpoint segment (7 cases, 9 slices) in {pin2}; counting them as zero area left the agreement unchanged. "
              "CSA, cross-sectional area; ICC, intraclass correlation coefficient; LCSA50, length of airway below 50 mm².")
    doc.save(OUT / "SuppTable_S5_postprocessing_bounds.docx")


# ---------------------------------------------------------------- S6 head positioning
def s6():
    t = (FA / "35_anatomical_midline.txt").read_text(); T = pd.read_csv(FA / "35_anatomical_midline_tests.csv")
    d = pd.read_csv(FA / "35_anatomical_midline.csv").merge(pd.read_csv(FA / "airway_morphology_merged.csv")[["case_id", "icu_binary"]].assign(case_id=lambda x: x.case_id.astype(str)), on="case_id")
    rows = []
    for c, lab in [("yaw_deg", "Yaw (rotation about the craniocaudal axis), °"), ("roll_deg", "Roll (in-plane tilt of the midsagittal plane), °"), ("head_offset_mm", "Lateral offset of the midsagittal plane from the image center, mm")]:
        a = d[c].abs(); rows.append([lab, f"{a.median():.1f} ({a.quantile(.25):.1f}–{a.quantile(.75):.1f})", f"{a.max():.1f}", f"{int((a>5).sum())} ({100*(a>5).mean():.0f})", f"{__import__('scipy.stats').stats.spearmanr(a, d.icu_binary).statistic:+.2f}"])
    doc = Document(); setup(doc)
    title(doc, "Supplementary Table S6. Head positioning and maximum midline shift referenced to the patient's midsagittal plane (n = 322).")
    para_ = doc.add_paragraph(); run(para_, "A. Positioning parameters estimated from bilateral image symmetry", bold=True)
    table(doc, ["Parameter (absolute value)", "Median (IQR)", "Maximum", "> 5° or > 5 mm, n (%)", "Spearman ρ with ICU admission"], rows)
    para_ = doc.add_paragraph(); para_.paragraph_format.space_before = Pt(8); run(para_, "B. Association of maximum midline shift with ICU admission by reference used", bold=True)
    lab = {"rl_dev_max_abs_mm": "Image midline (as reported, Table 3)", "maxMLS_img": "Image midline, recomputed in the resampled frame", "maxMLS_anat": "Midsagittal plane from image symmetry (fitted plane)",
           "maxMLS_anat_raw": "Midsagittal line from image symmetry, per slice (unsmoothed)", "meanMLS_anat": "Mean (rather than maximum) shift from the midsagittal plane"}
    rows2 = []; bold = []
    for coh in ["Full cohort", "Abscess-positive"]:
        sub = T[T.cohort == coh]; rows2.append([f"{coh} (n = {int(sub.n.iloc[0])}, ICU = {int(sub.icu.iloc[0])})", "", "", "", "", ""]); bold.append(len(rows2) - 1)
        for _, r in sub.iterrows():
            rows2.append([lab[r.variable], f"{r.med_noicu:.1f}", f"{r.med_icu:.1f}", fp(r.p), f"{r.r_rb:+.2f}", f"{r.auc:.3f}"])
    table(doc, ["Reference for midline shift", "Median, no ICU (mm)", "Median, ICU (mm)", "p value", "r", "AUC"], rows2, bold_rows=bold)
    rmse = re.search(r"midline fit RMSE median ([\d.]+) mm", t).group(1); nfb = int((d.roll_deg == 0).sum())
    note(doc, "A: absolute values of the positioning parameters estimated from bilateral image symmetry (Supplementary Methods). B: maximum midline shift recomputed as the largest distance between the per-slice airway centroid and the stated reference. "
              "p, Mann–Whitney U test; r, rank-biserial correlation (positive = larger in ICU patients); AUC, area under the receiver operating characteristic curve; IQR, interquartile range.")
    doc.save(OUT / "SuppTable_S6_positioning.docx")


if __name__ == "__main__":
    table3(); table4(); s1(); s2(); s3(); s4(); s5(); s6()
    print("built:", sorted(p.name for p in OUT.glob("*.docx")))
