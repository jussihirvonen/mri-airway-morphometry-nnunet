#!/usr/bin/env Rscript
# Revision analysis (EJR-D-26-04240, reviewer point 5): incremental value of
# airway features over the PUBLISHED reference risk score, applied with its
# published coefficients and cut-offs (no refitting in this cohort).
# Vierula et al. 2025 (Eur J Radiol Open 14:100648): risk score =
#   3 x RPE present + 2 x (CRP > 171.5 mg/L) + 2 x (max abscess diameter > 39.5 mm), range 0-7.
# Models: (a) fixed score alone (AUC of the integer score, no fitting);
#         (b) Firth logistic: score + minimum CSA; (c) score + four airway features
#         (score entered as a single covariate so its internal weights stay fixed).
# Discrimination: apparent AUC with DeLong CI, bootstrap optimism-corrected AUC
# (B=500), DeLong test vs the fixed score, penalized LRT.
# Output: 36_fixed_score_reference.csv / .txt
suppressPackageStartupMessages({library(logistf); library(pROC)})
root <- Sys.getenv("AIRWAY_ROOT", unset = normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "..")))
fa <- file.path(root, "final_analysis")
d <- read.csv(file.path(fa, "airway_morphology_merged.csv"))
d$rpe_present <- d$retroinf
air4 <- c("min_area_mm2", "compactness", "rl_dev_max_abs_mm", "len_below_50mm2_mm")
d <- d[complete.cases(d[, c("icu_binary", "crp", "rpe_present", "abs_diamax_mri", air4)]), ]
d$score <- 3 * d$rpe_present + 2 * as.numeric(d$crp > 171.5) + 2 * as.numeric(d$abs_diamax_mri > 39.5)
set.seed(42); B <- 500
txt <- c("Fixed published risk score (Vierula 2025) as reference — no refitting", "")
out <- list()
for (cohort in c("Full cohort", "Abscess-positive subset")) {
  s <- if (cohort == "Full cohort") d else d[d$abs_positive == 1, ]
  X <- as.data.frame(scale(s[, air4])); X$score <- s$score; X$y <- s$icu_binary
  n <- nrow(X); ev <- sum(X$y)
  txt <- c(txt, sprintf("%s: n=%d, ICU=%d; score distribution: %s", cohort, n, ev, paste(names(table(X$score)), table(X$score), sep = ":", collapse = " ")))
  r0 <- roc(X$y, X$score, quiet = TRUE); a0 <- as.numeric(auc(r0)); c0 <- as.numeric(ci.auc(r0, method = "delong"))
  txt <- c(txt, sprintf("  Fixed score alone: AUC %.3f (%.3f-%.3f)  [published derivation AUC 0.82 in n=535]", a0, c0[1], c0[3]))
  txt <- c(txt, sprintf("  Score > 3 (published cut-off): ICU %d/%d (%.1f%%) vs <=3: %d/%d (%.1f%%)", sum(X$y[X$score > 3]), sum(X$score > 3), 100 * mean(X$y[X$score > 3]), sum(X$y[X$score <= 3]), sum(X$score <= 3), 100 * mean(X$y[X$score <= 3])))
  out[[length(out) + 1]] <- data.frame(cohort = cohort, n = n, events = ev, model = "Fixed score alone", AUC_apparent = a0, CI_lo = c0[1], CI_hi = c0[3], optimism = 0, AUC_corrected = a0, p_delong_vs_score = NA, p_LRT_vs_score = NA)
  f_score <- logistf(y ~ score, data = X)
  for (nm in c("Score + minimum CSA", "Score + four airway features")) {
    preds <- if (nm == "Score + minimum CSA") c("score", "min_area_mm2") else c("score", air4)
    f <- as.formula(paste("y ~", paste(preds, collapse = " + ")))
    fit <- logistf(f, data = X, control = logistf.control(maxit = 200), plcontrol = logistpl.control(maxit = 200))
    r1 <- roc(X$y, fit$predict, quiet = TRUE); a1 <- as.numeric(auc(r1)); c1 <- as.numeric(ci.auc(r1, method = "delong"))
    pd <- roc.test(r0, r1, method = "delong")$p.value
    lrt <- anova(f_score, fit)
    opt <- numeric(B)
    for (b in 1:B) {
      idx <- sample(n, n, replace = TRUE); Xb <- X[idx, ]
      fb <- tryCatch(logistf(f, data = Xb, control = logistf.control(maxit = 200), plcontrol = logistpl.control(maxit = 200)), error = function(e) NULL)
      if (is.null(fb)) { opt[b] <- NA; next }
      lp <- as.numeric(cbind(1, as.matrix(X[, preds, drop = FALSE])) %*% coef(fb))
      opt[b] <- as.numeric(auc(roc(Xb$y, fb$predict, quiet = TRUE))) - as.numeric(auc(roc(X$y, lp, quiet = TRUE)))
    }
    o <- mean(opt, na.rm = TRUE)
    txt <- c(txt, sprintf("  %-30s apparent AUC %.3f (%.3f-%.3f)  dAUC %+.3f  optimism %.3f  corrected %.3f  DeLong p=%.3f  penalized LRT p=%.4f",
                          nm, a1, c1[1], c1[3], a1 - a0, o, a1 - o, pd, lrt$pval))
    co <- coef(fit); lo <- fit$ci.lower; hi <- fit$ci.upper; pv <- fit$prob
    for (i in seq_along(co)) if (names(co)[i] != "(Intercept)")
      txt <- c(txt, sprintf("      %-20s OR %.2f (%.2f-%.2f) p=%.4f%s", names(co)[i], exp(co[i]), exp(lo[i]), exp(hi[i]), pv[i], if (names(co)[i] == "score") " (per point)" else " (per SD)"))
    out[[length(out) + 1]] <- data.frame(cohort = cohort, n = n, events = ev, model = nm, AUC_apparent = a1, CI_lo = c1[1], CI_hi = c1[3], optimism = o, AUC_corrected = a1 - o, p_delong_vs_score = pd, p_LRT_vs_score = lrt$pval)
  }
  txt <- c(txt, "")
}
write.csv(do.call(rbind, out), file.path(fa, "36_fixed_score_reference.csv"), row.names = FALSE)
writeLines(txt, file.path(fa, "36_fixed_score_reference.txt")); cat(paste(txt, collapse = "\n"), "\n")
