#!/usr/bin/env Rscript
# Revision R1, second-opinion point 3: the four-feature model of Table 4 comprises the
# features that were univariately associated with ICU admission (Table 3), i.e. an
# outcome-based selection. Here the bootstrap optimism correction REPEATS that selection
# in every resample: among the five pre-specified features, those with Mann-Whitney
# p < 0.05 on the resample are added to the reference model, the Firth model is fitted on
# the resample, and optimism = AUC(resample) - AUC(original) of that selected model.
# Also reports how often each feature was selected. B = 500, seed 42.
# Output: 40_selection_aware_validation.txt / .csv
suppressPackageStartupMessages({library(logistf); library(pROC)})
root <- Sys.getenv("AIRWAY_ROOT", unset = normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "..")))
fa <- file.path(root, "final_analysis")
d <- read.csv(file.path(fa, "airway_morphology_merged.csv")); d$rpe_present <- d$retroinf
air5 <- c("volume_ml", "min_area_mm2", "compactness", "rl_dev_max_abs_mm", "len_below_50mm2_mm"); ref <- c("crp", "rpe_present", "abs_diamax_mri")
d <- d[complete.cases(d[, c("icu_binary", ref, air5)]), ]
set.seed(42); B <- 500; out <- c(); res <- list()
for (cohort in c("Full cohort", "Abscess-positive subset")) {
  s <- if (cohort == "Full cohort") d else d[d$abs_positive == 1, ]
  X <- as.data.frame(scale(s[, c(ref, air5)])); X$y <- s$icu_binary; n <- nrow(X)
  select <- function(Z) air5[sapply(air5, function(v) wilcox.test(Z[[v]] ~ Z$y)$p.value < 0.05)]
  sel0 <- select(X)
  fit0 <- logistf(as.formula(paste("y ~", paste(c(ref, sel0), collapse = "+"))), data = X, control = logistf.control(maxit = 200), plcontrol = logistpl.control(maxit = 200))
  auc0 <- as.numeric(auc(roc(X$y, fit0$predict, quiet = TRUE)))
  fref <- logistf(y ~ crp + rpe_present + abs_diamax_mri, data = X); aucref <- as.numeric(auc(roc(X$y, fref$predict, quiet = TRUE)))
  opt <- numeric(B); optref <- numeric(B); selcount <- setNames(numeric(5), air5); nsel <- numeric(B)
  for (b in 1:B) {
    idx <- sample(n, n, replace = TRUE); Xb <- X[idx, ]
    sb <- select(Xb); selcount[sb] <- selcount[sb] + 1; nsel[b] <- length(sb)
    f <- as.formula(paste("y ~", paste(c(ref, sb), collapse = "+")))
    fb <- tryCatch(logistf(f, data = Xb, control = logistf.control(maxit = 200), plcontrol = logistpl.control(maxit = 200)), error = function(e) NULL)
    if (is.null(fb)) { opt[b] <- NA; optref[b] <- NA; next }
    lp <- as.numeric(cbind(1, as.matrix(X[, c(ref, sb), drop = FALSE])) %*% coef(fb))
    opt[b] <- as.numeric(auc(roc(Xb$y, fb$predict, quiet = TRUE))) - as.numeric(auc(roc(X$y, lp, quiet = TRUE)))
    frb <- logistf(y ~ crp + rpe_present + abs_diamax_mri, data = Xb)
    lpr <- as.numeric(cbind(1, as.matrix(X[, ref])) %*% coef(frb))
    optref[b] <- as.numeric(auc(roc(Xb$y, frb$predict, quiet = TRUE))) - as.numeric(auc(roc(X$y, lpr, quiet = TRUE)))
  }
  o <- mean(opt, na.rm = TRUE); oref <- mean(optref, na.rm = TRUE)
  out <- c(out, sprintf("%s: n=%d, ICU=%d; features selected on the full data: %s", cohort, n, sum(X$y), paste(sel0, collapse = ", ")),
           sprintf("  selected-feature model: apparent AUC %.3f, selection-aware optimism %.3f, corrected AUC %.3f (reference: apparent %.3f, corrected %.3f) -> corrected gain %+.3f", auc0, o, auc0 - o, aucref, aucref - oref, (auc0 - o) - (aucref - oref)),
           sprintf("  number of features selected per resample: median %d (range %d-%d); selection frequency: %s", median(nsel), min(nsel), max(nsel), paste(sprintf("%s %.0f%%", air5, 100 * selcount / B), collapse = ", ")))
  res[[length(res) + 1]] <- data.frame(cohort = cohort, n = n, events = sum(X$y), AUC_apparent = auc0, optimism_selection_aware = o, AUC_corrected = auc0 - o, ref_AUC_apparent = aucref, ref_AUC_corrected = aucref - oref)
}
write.csv(do.call(rbind, res), file.path(fa, "40_selection_aware_validation.csv"), row.names = FALSE)
writeLines(out, file.path(fa, "40_selection_aware_validation.txt")); cat(paste(out, collapse = "\n"), "\n")
