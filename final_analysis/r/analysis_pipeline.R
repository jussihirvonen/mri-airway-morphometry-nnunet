#!/usr/bin/env Rscript
# ===========================================================================
# Airway Morphometry Analysis Pipeline (R version)
# ===========================================================================
#
# R replication of the statistical analyses underpinning Paper 1's tables and
# figures (Python pipeline in ../03_analyze.py, ../08_severity_cohort_compare.py,
# ../13_inter_rater.py, ../14_descriptive_prespecified.py).
#
# Mirrors the abscess paper's r/ structure
# (the parent study's R pipeline).
#
# Input:
#   r/airway_features_with_outcomes.csv  (produced by 00_prepare_r_input.py)
#   r/inter_rater_n20.csv                (produced by 00_prepare_r_input.py)
#
# Output:
#   r/results_report.txt                 (full results report)
#   r/cross_validation.txt               (Python vs R cross-check)
#
# Required packages:
#   install.packages(c("dplyr", "pROC", "boot", "irr", "tidyr"))
#
# Usage:
#   cd final_analysis/r/
#   Rscript analysis_pipeline.R
#
# Authors: see the README citation.
# Date: 2026-06
# ===========================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(pROC)
  library(boot)
  library(irr)
  library(tidyr)
})

set.seed(42)

# ---------------------------------------------------------------------------
# Paths and data loading.
# Use --args to pass an absolute path to the r/ folder; otherwise assume cwd.
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else getwd()

df <- read.csv(file.path(data_dir, "airway_features_with_outcomes.csv"),
               stringsAsFactors = FALSE)
ir <- read.csv(file.path(data_dir, "inter_rater_n20.csv"),
               stringsAsFactors = FALSE)

# ---------------------------------------------------------------------------
# RPE presence: use the complete clinical flag (retroinf), not the volumetric
# derivation. rpe_present was defined as rpe_volume_ml > 0, which is NA for the
# 23 cases without a volumetric RPE segmentation (8 of them RPE-positive). The
# complete binary flag `retroinf` is 0/1 for all 322 patients and agrees
# perfectly with rpe_present where both are defined (144/144 neg, 155/155 pos).
# Overwriting here propagates the complete variable to the descriptive count,
# the published reference ICU model, the convergent-validity panel, and the
# abscess-negative robustness check. rpe_volume_ml is left untouched (the
# volume is only defined in the 299 volumetrically segmented cases).
stopifnot(!any(is.na(df$retroinf)))
df$rpe_present <- df$retroinf
REPORT <- file.path(data_dir, "results_report.txt")
cat(sprintf("Loaded %d patients (full cohort)\n", nrow(df)))
cat(sprintf("Loaded %d inter-rater dual-segmentations\n", nrow(ir)))

sink(REPORT, split = TRUE)

hr <- function(s) cat("\n", strrep("-", 75), "\n", s, "\n", strrep("-", 75), "\n", sep = "")

fmt_p <- function(p) {
  if (is.na(p)) return("—")
  if (p < 0.001) return(formatC(p, format = "e", digits = 1))
  return(sprintf("%.3f", p))
}

cat("===========================================================================\n")
cat(" Airway Morphometry — R analysis pipeline\n")
cat("===========================================================================\n")
cat(sprintf(" Generated: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M")))
cat(sprintf(" Cohort n = %d, ICU = %d (%.1f%%)\n",
            nrow(df), sum(df$icu_binary == 1, na.rm = TRUE),
            100 * mean(df$icu_binary == 1, na.rm = TRUE)))

# ---------------------------------------------------------------------------
# 1. Cohort description (matches Table 1)
# ---------------------------------------------------------------------------
hr("1. COHORT DESCRIPTION (Table 1)")
cat(sprintf("  Age:             median %.0f  IQR %.0f–%.0f  (range %.0f–%.0f)\n",
            median(df$age, na.rm = TRUE),
            quantile(df$age, 0.25, na.rm = TRUE),
            quantile(df$age, 0.75, na.rm = TRUE),
            min(df$age, na.rm = TRUE), max(df$age, na.rm = TRUE)))
sex_male <- sum(df$sex == 1, na.rm = TRUE)
sex_female <- sum(df$sex == 2, na.rm = TRUE)
cat(sprintf("  Sex M/F:         %d/%d (%.1f/%.1f%%)\n",
            sex_male, sex_female, 100*sex_male/nrow(df), 100*sex_female/nrow(df)))
cat(sprintf("  CRP, mg/L:       median %.0f  IQR %.0f–%.0f\n",
            median(df$crp, na.rm=TRUE), quantile(df$crp, 0.25, na.rm=TRUE),
            quantile(df$crp, 0.75, na.rm=TRUE)))
cat(sprintf("  Hospital days:   median %.0f  IQR %.0f–%.0f\n",
            median(df$hospd, na.rm=TRUE), quantile(df$hospd, 0.25, na.rm=TRUE),
            quantile(df$hospd, 0.75, na.rm=TRUE)))
n_los <- sum(!is.na(df$hospd_prolonged))
n_pro <- sum(df$hospd_prolonged == 1, na.rm = TRUE)
cat(sprintf("  Prolonged stay:  %d / %d (%.1f%%)\n",
            n_pro, n_los, 100*n_pro/n_los))
cat(sprintf("  ICU admission:   %d / %d (%.1f%%)\n",
            sum(df$icu_binary == 1, na.rm = TRUE), nrow(df),
            100*mean(df$icu_binary == 1, na.rm = TRUE)))
cat(sprintf("  Abscess+:        %d / %d (%.1f%%)\n",
            sum(df$abs_positive == 1), nrow(df),
            100*mean(df$abs_positive == 1)))
abs_pos <- df %>% filter(abs_positive == 1)
cat(sprintf("  Abs max diam,mm (abs+ only): median %.0f  IQR %.0f–%.0f\n",
            median(abs_pos$abs_diamax_mri, na.rm = TRUE),
            quantile(abs_pos$abs_diamax_mri, 0.25, na.rm = TRUE),
            quantile(abs_pos$abs_diamax_mri, 0.75, na.rm = TRUE)))
n_rpe_eval <- sum(!is.na(df$rpe_present))
cat(sprintf("  RPE present:     %d / %d (%.1f%%)\n",
            sum(df$rpe_present == 1, na.rm=TRUE), n_rpe_eval,
            100*sum(df$rpe_present == 1, na.rm=TRUE)/n_rpe_eval))
cat(sprintf("  RPE volume, mL:  median %.1f  IQR %.1f–%.1f\n",
            median(df$rpe_volume_ml, na.rm=TRUE),
            quantile(df$rpe_volume_ml, 0.25, na.rm=TRUE),
            quantile(df$rpe_volume_ml, 0.75, na.rm=TRUE)))
n_me <- sum(!is.na(df$medoed_binary))
cat(sprintf("  Mediastinal edema: %d / %d (%.1f%%)\n",
            sum(df$medoed_binary == 1, na.rm=TRUE), n_me,
            100*sum(df$medoed_binary == 1, na.rm=TRUE)/n_me))

# ---------------------------------------------------------------------------
# 2. Inter-observer agreement (matches Table 2)
# ---------------------------------------------------------------------------
hr("2. INTER-OBSERVER AGREEMENT (Table 2, n=20)")
icc_features <- c("volume_ml", "min_area_mm2", "compactness",
                  "rl_dev_max_abs_mm", "len_below_50mm2_mm")
icc_labels   <- c("Airway volume (mL)", "Min CSA (mm²)", "Compactness",
                  "Max midline shift (mm)", "Length <50 mm² (mm)")

cat(sprintf("  %-26s  %6s  %16s  %4s\n", "Feature", "ICC", "95% CI", "n"))
for (i in seq_along(icc_features)) {
  jcol <- paste0("expert_", icc_features[i])
  vcol <- paste0("reference_", icc_features[i])
  if (!(jcol %in% names(ir) && vcol %in% names(ir))) next
  pairs <- na.omit(data.frame(j = ir[[jcol]], v = ir[[vcol]]))
  out <- irr::icc(pairs, model = "twoway", type = "agreement", unit = "single")
  cat(sprintf("  %-26s  %6.3f  [%.3f, %.3f]  %4d\n",
              icc_labels[i], out$value, out$lbound, out$ubound, nrow(pairs)))
}
cat(sprintf("\n  Dice:   mean %.3f  range %.3f–%.3f  (n=%d)\n",
            mean(ir$dice), min(ir$dice), max(ir$dice), nrow(ir)))
cat(sprintf("  HD95 mm: median %.1f  IQR %.1f–%.1f  max %.1f\n",
            median(ir$hd95_mm), quantile(ir$hd95_mm, 0.25),
            quantile(ir$hd95_mm, 0.75), max(ir$hd95_mm)))

# ---------------------------------------------------------------------------
# 3. Univariate features × ICU and Prolonged LOS (matches Table 3)
# ---------------------------------------------------------------------------
hr("3. PRE-SPECIFIED FEATURES × ICU AND PROLONGED LOS (Table 3)")

univariate_block <- function(outcome, label) {
  cat(sprintf("\n  %s\n", label))
  sub <- df %>% filter(!is.na(.data[[outcome]]))
  n_pos <- sum(sub[[outcome]] == 1); n_neg <- sum(sub[[outcome]] == 0)
  cat(sprintf("  (Positive n=%d, Negative n=%d)\n", n_pos, n_neg))
  cat(sprintf("  %-26s  %20s  %20s  %10s  %5s  %5s\n",
              "Feature", "Positive med (IQR)", "Negative med (IQR)", "p", "r", "AUC"))
  for (i in seq_along(icc_features)) {
    f <- icc_features[i]
    s <- na.omit(sub[, c(outcome, f)])
    a <- s[s[[outcome]] == 1, f]; b <- s[s[[outcome]] == 0, f]
    u <- wilcox.test(a, b)
    p <- u$p.value
    rb <- 1 - 2 * u$statistic / (length(a) * length(b))
    auc <- as.numeric(pROC::roc(s[[outcome]], s[[f]],
                                 quiet = TRUE, direction = "auto")$auc)
    cat(sprintf("  %-26s  %20s  %20s  %10s  %5.2f  %5.3f\n",
                icc_labels[i],
                sprintf("%.2f (%.2f–%.2f)", median(a), quantile(a, .25), quantile(a, .75)),
                sprintf("%.2f (%.2f–%.2f)", median(b), quantile(b, .25), quantile(b, .75)),
                fmt_p(p), abs(rb), auc))
  }
}

univariate_block("icu_binary",     "ICU admission")
univariate_block("hospd_prolonged", "Prolonged hospitalization (>5 days)")

# ---------------------------------------------------------------------------
# 4. Incremental ICU models — logistic + DeLong (matches Table 4)
# ---------------------------------------------------------------------------
hr("4. INCREMENTAL ICU PREDICTION (Table 4, logistic + DeLong)")

run_incremental_block <- function(d, cohort_label) {
  req <- c("icu_binary", "crp", "rpe_present", "abs_diamax_mri",
           "min_area_mm2", "compactness", "rl_dev_max_abs_mm", "len_below_50mm2_mm")
  s <- na.omit(d[, req])
  n <- nrow(s); n_icu <- sum(s$icu_binary == 1)
  cat(sprintf("\n  %s (n=%d, ICU=%d)\n", cohort_label, n, n_icu))

  models <- list(
    "Reference model"               = c("crp", "rpe_present", "abs_diamax_mri"),
    "Reference + min CSA"         = c("crp", "rpe_present", "abs_diamax_mri", "min_area_mm2"),
    "Reference + 4 airway features" = c("crp", "rpe_present", "abs_diamax_mri",
                                     "min_area_mm2", "compactness",
                                     "rl_dev_max_abs_mm", "len_below_50mm2_mm")
  )

  s_std <- s
  num_cols <- setdiff(names(s), "icu_binary")
  for (c in num_cols) s_std[[c]] <- scale(s[[c]])[, 1]

  rocs <- list(); aucs <- list(); cis <- list()
  for (m in names(models)) {
    fmla <- as.formula(paste("icu_binary ~", paste(models[[m]], collapse = " + ")))
    fit <- glm(fmla, data = s_std, family = binomial)
    p_hat <- predict(fit, type = "response")
    r <- pROC::roc(s$icu_binary, p_hat, quiet = TRUE, direction = "auto")
    rocs[[m]] <- r
    aucs[[m]] <- as.numeric(r$auc)
    ci <- pROC::ci.auc(r, method = "delong")
    cis[[m]] <- c(ci[1], ci[3])
  }

  cat(sprintf("  %-30s  %14s  %10s  %10s\n", "Model", "AUC (95% CI)", "ΔAUC", "DeLong p"))
  ref <- "Reference model"
  for (m in names(models)) {
    auc_str <- sprintf("%.3f (%.3f–%.3f)", aucs[[m]], cis[[m]][1], cis[[m]][2])
    if (m == ref) {
      cat(sprintf("  %-30s  %14s  %10s  %10s\n", m, auc_str, "(ref)", ""))
    } else {
      d_auc <- aucs[[m]] - aucs[[ref]]
      delong <- pROC::roc.test(rocs[[ref]], rocs[[m]], method = "delong")
      cat(sprintf("  %-30s  %14s  %+10.3f  %10s\n",
                  m, auc_str, d_auc, fmt_p(delong$p.value)))
    }
  }
}

run_incremental_block(df, "Full cohort")
run_incremental_block(df %>% filter(abs_positive == 1), "Abscess-positive subset")

# ---------------------------------------------------------------------------
# 5. Convergent-validity panel (matches Figure 2 / Supp Table 2)
# ---------------------------------------------------------------------------
hr("5. CONVERGENT-VALIDITY PANEL (full cohort, 6 markers)")

BINARY     <- c("icu_binary", "hospd_prolonged", "rpe_present", "medoed_binary")
BINARY_LAB <- c("ICU", "Prolonged LOS", "RPE+", "Mediastinal edema")
CONT       <- c("crp", "abs_diamax_mri")
CONT_LAB   <- c("CRP", "Abscess diam")

cat(sprintf("  %-26s  %12s  %12s  %10s  %5s\n",
            "Feature × Marker", "marker type", "effect", "p", "n"))
for (i in seq_along(icc_features)) {
  for (j in seq_along(BINARY)) {
    s <- na.omit(df[, c(BINARY[j], icc_features[i])])
    a <- s[s[[BINARY[j]]] == 1, icc_features[i]]
    b <- s[s[[BINARY[j]]] == 0, icc_features[i]]
    if (length(a) < 2 || length(b) < 2) next
    u <- wilcox.test(a, b)
    rb <- 1 - 2 * u$statistic / (length(a) * length(b))
    cat(sprintf("  %-26s  %12s  %+12.3f  %10s  %5d\n",
                paste0(icc_labels[i], " × ", BINARY_LAB[j]),
                "binary (r_rb)", -rb, fmt_p(u$p.value), nrow(s)))
  }
  for (j in seq_along(CONT)) {
    s <- na.omit(df[, c(CONT[j], icc_features[i])])
    sp <- suppressWarnings(cor.test(s[[CONT[j]]], s[[icc_features[i]]],
                                     method = "spearman"))
    cat(sprintf("  %-26s  %12s  %+12.3f  %10s  %5d\n",
                paste0(icc_labels[i], " × ", CONT_LAB[j]),
                "Spearman ρ", sp$estimate, fmt_p(sp$p.value), nrow(s)))
  }
}

# ---------------------------------------------------------------------------
# 6. Abscess-negative robustness (midline shift × RPE present)
# ---------------------------------------------------------------------------
hr("6. abs- ROBUSTNESS (midline shift × RPE present)")
neg <- df %>% filter(abs_positive == 0, !is.na(rpe_present), !is.na(rl_dev_max_abs_mm))
a <- neg %>% filter(rpe_present == 1) %>% pull(rl_dev_max_abs_mm)
b <- neg %>% filter(rpe_present == 0) %>% pull(rl_dev_max_abs_mm)
u <- wilcox.test(a, b)
rb <- 1 - 2 * u$statistic / (length(a) * length(b))
cat(sprintf("  abs- midline shift vs RPE+: r_rb = %+.3f, p = %s (n=%d, RPE+ = %d)\n",
            -rb, fmt_p(u$p.value), nrow(neg), length(a)))

# ---------------------------------------------------------------------------
# 7. Odontogenic vs Pharyngotonsillar (Supp Table 3)
# ---------------------------------------------------------------------------
hr("7. ODONTOGENIC vs PHARYNGOTONSILLAR (Supp Table 3)")
odon <- df %>% filter(hammas == 1 & (is.na(peritonsinf) | peritonsinf == 0))
phar <- df %>% filter(peritonsinf == 1 & (is.na(hammas) | hammas == 0))
cat(sprintf("  Group sizes: odontogenic n=%d, pharyngotonsillar n=%d\n",
            nrow(odon), nrow(phar)))
cat(sprintf("  %-26s  %20s  %20s  %10s  %5s\n",
            "Feature", "Odontogenic", "Pharyngotonsillar", "p", "r"))
for (i in seq_along(icc_features)) {
  f <- icc_features[i]
  a <- na.omit(odon[[f]]); b <- na.omit(phar[[f]])
  u <- wilcox.test(a, b)
  rb <- 1 - 2 * u$statistic / (length(a) * length(b))
  cat(sprintf("  %-26s  %20s  %20s  %10s  %5.2f\n",
              icc_labels[i],
              sprintf("%.2f (%.2f–%.2f)", median(a), quantile(a, .25), quantile(a, .75)),
              sprintf("%.2f (%.2f–%.2f)", median(b), quantile(b, .25), quantile(b, .75)),
              fmt_p(u$p.value), abs(rb)))
}

icu_o <- sum(odon$icu_binary == 1); icu_p <- sum(phar$icu_binary == 1)
fe <- fisher.test(matrix(c(icu_o, nrow(odon) - icu_o,
                           icu_p, nrow(phar) - icu_p), nrow = 2))
cat(sprintf("\n  ICU rate:  odontogenic %d/%d (%.1f%%)  vs  pharyngotonsillar %d/%d (%.1f%%);  Fisher's p = %s\n",
            icu_o, nrow(odon), 100*icu_o/nrow(odon),
            icu_p, nrow(phar), 100*icu_p/nrow(phar), fmt_p(fe$p.value)))

# ---------------------------------------------------------------------------
# 8. Risk stratification by abscess size and airway compactness
# ---------------------------------------------------------------------------
hr("8. RISK STRATIFICATION BY ABSCESS SIZE AND AIRWAY COMPACTNESS")
rs <- df %>% filter(!is.na(abs_diamax_mri), !is.na(compactness), !is.na(icu_binary))
rs$large <- as.integer(rs$abs_diamax_mri >= 40)
small <- rs %>% filter(large == 0)
large <- rs %>% filter(large == 1)
median_comp <- median(large$compactness)
large_good <- large %>% filter(compactness >= median_comp)
large_bad  <- large %>% filter(compactness <  median_comp)

cat(sprintf("  Compactness median among large abscesses: %.3f\n\n", median_comp))
cat(sprintf("  %-40s  %5s  %15s  %12s\n", "Group", "n", "ICU n (%)", "RR vs small"))
ref_rate <- mean(small$icu_binary == 1)
groups <- list(list("Small (<40 mm)", small),
               list("Large (≥40 mm) + good compactness", large_good),
               list("Large (≥40 mm) + poor compactness", large_bad))
for (lg in groups) {
  lab <- lg[[1]]; g <- lg[[2]]
  n <- nrow(g); icu <- sum(g$icu_binary == 1)
  rate <- icu/n; rr <- rate/ref_rate
  cat(sprintf("  %-40s  %5d  %15s  %12.2f\n",
              lab, n, sprintf("%d (%.1f%%)", icu, 100*rate), rr))
}
fe <- fisher.test(matrix(c(sum(large_bad$icu_binary  == 1), nrow(large_bad)  - sum(large_bad$icu_binary == 1),
                           sum(large_good$icu_binary == 1), nrow(large_good) - sum(large_good$icu_binary == 1)),
                         nrow = 2))
cat(sprintf("\n  Within large abscesses, poor vs good compactness: Fisher's exact p = %s\n",
            fmt_p(fe$p.value)))

# ---------------------------------------------------------------------------
# 9. Joint 2x2: abs sphericity × airway compactness (within large abscesses)
# ---------------------------------------------------------------------------
hr("9. JOINT 2x2: abscess sphericity × airway compactness (large abscesses)")
js <- df %>% filter(abs_positive == 1,
                    !is.na(abs_diamax_mri), !is.na(abs_sphericity_radiomics),
                    !is.na(compactness), !is.na(icu_binary))
js$large <- as.integer(js$abs_diamax_mri >= 40)
jl <- js %>% filter(large == 1)
sph_med <- median(jl$abs_sphericity_radiomics)
air_med <- median(jl$compactness)
cat(sprintf("  n large = %d; sphericity median %.3f; airway compactness median %.3f\n\n",
            nrow(jl), sph_med, air_med))

rho_full  <- suppressWarnings(cor.test(js$abs_sphericity_radiomics, js$compactness, method = "spearman"))
rho_large <- suppressWarnings(cor.test(jl$abs_sphericity_radiomics, jl$compactness, method = "spearman"))
cat(sprintf("  Spearman ρ (abs+ cohort n=%d):    ρ = %+.3f, p = %s\n",
            nrow(js), rho_full$estimate, fmt_p(rho_full$p.value)))
cat(sprintf("  Spearman ρ (large only n=%d):    ρ = %+.3f, p = %s\n\n",
            nrow(jl), rho_large$estimate, fmt_p(rho_large$p.value)))

cat(sprintf("  %-40s  %4s  %15s\n", "Cell", "n", "ICU n (%)"))
combos <- list(
  list("Round + OK airway",     jl$abs_sphericity_radiomics >= sph_med & jl$compactness >= air_med),
  list("Round + BAD airway",    jl$abs_sphericity_radiomics >= sph_med & jl$compactness <  air_med),
  list("Irregular + OK airway", jl$abs_sphericity_radiomics <  sph_med & jl$compactness >= air_med),
  list("Irregular + BAD airway",jl$abs_sphericity_radiomics <  sph_med & jl$compactness <  air_med)
)
for (c in combos) {
  g <- jl[c[[2]], ]; n <- nrow(g); icu <- sum(g$icu_binary == 1)
  cat(sprintf("  %-40s  %4d  %15s\n",
              c[[1]], n, sprintf("%d (%.1f%%)", icu, 100*icu/n)))
}

cat("\n===========================================================================\n")
cat(" R analysis complete\n")
cat("===========================================================================\n")

sink()
cat(sprintf("\nReport written to %s\n", REPORT))
