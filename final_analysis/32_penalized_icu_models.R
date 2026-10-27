#!/usr/bin/env Rscript
# Revision analysis (EJR-D-26-04240, reviewer points 4 and 5): penalized
# logistic regression for the incremental ICU models of Table 4.
#
# With 33 ICU events, the four-feature-plus-reference model has 7 predictors
# (EPV ~ 4.7). Standard ML logistic regression is replaced by Firth's
# bias-reduced penalized likelihood (logistf), and model discrimination is
# reported as apparent AUC and bootstrap optimism-corrected AUC (Harrell,
# B = 500). Nested models are compared by penalized likelihood-ratio tests
# (anova.logistf). A LASSO (glmnet, 10-fold CV repeated 20x) is run as a
# data-driven check on whether any airway feature is retained beside the
# reference predictors.
#
# Reference model = CRP + RPE presence (retroinf) + max abscess diameter
# (zero in abscess-negative patients), refitted in this cohort as in Table 4.
#
# Output: 32_penalized_icu_models.csv / .txt
suppressPackageStartupMessages({library(logistf); library(pROC); library(glmnet)})
root <- Sys.getenv("AIRWAY_ROOT", unset = normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "..")))
fa <- file.path(root, "final_analysis")
d <- read.csv(file.path(fa, "airway_morphology_merged.csv"))
d$rpe_present <- d$retroinf
air4 <- c("min_area_mm2", "compactness", "rl_dev_max_abs_mm", "len_below_50mm2_mm")
ref  <- c("crp", "rpe_present", "abs_diamax_mri")
keep <- complete.cases(d[, c("icu_binary", ref, air4, "age", "sex")])
d <- d[keep, ]
models <- list("Reference (Turku ICU-risk model)" = ref,
               "Reference + minimum CSA" = c(ref, "min_area_mm2"),
               "Reference + four airway features" = c(ref, air4),
               "Reference + age + sex" = c(ref, "age", "sex"),
               "Reference + age + sex + minimum CSA" = c(ref, "age", "sex", "min_area_mm2"))
set.seed(42); B <- 500
out <- list(); txt <- c()
for (cohort in c("Full cohort", "Abscess-positive subset")) {
  s <- if (cohort == "Full cohort") d else d[d$abs_positive == 1, ]
  X <- s[, c(ref, air4, "age")]; X <- as.data.frame(scale(X)); X$sex <- as.numeric(s$sex == 1); X$y <- s$icu_binary
  n <- nrow(X); ev <- sum(X$y)
  txt <- c(txt, sprintf("\n%s: n = %d, ICU events = %d", cohort, n, ev))
  fits <- list(); aucs <- list()
  for (nm in names(models)) {
    f <- as.formula(paste("y ~", paste(models[[nm]], collapse = " + ")))
    fit <- logistf(f, data = X, control = logistf.control(maxit = 200), plcontrol = logistpl.control(maxit = 200))
    fits[[nm]] <- fit
    p_app <- fit$predict
    r_app <- roc(X$y, p_app, quiet = TRUE); auc_app <- as.numeric(auc(r_app)); ci_app <- as.numeric(ci.auc(r_app, method = "delong"))
    # bootstrap optimism
    lp_app <- as.numeric(cbind(1, as.matrix(X[, models[[nm]], drop = FALSE])) %*% coef(fit))
    slope_app <- as.numeric(coef(glm(X$y ~ lp_app, family = binomial))[2])
    opt <- numeric(B); slopes <- numeric(B)
    for (b in 1:B) {
      idx <- sample(n, n, replace = TRUE); Xb <- X[idx, ]
      fb <- tryCatch(logistf(f, data = Xb, control = logistf.control(maxit = 200), plcontrol = logistpl.control(maxit = 200)), error = function(e) NULL)
      if (is.null(fb)) { opt[b] <- NA; next }
      pb_boot <- fb$predict
      lp_orig <- as.numeric(cbind(1, as.matrix(X[, models[[nm]], drop = FALSE])) %*% coef(fb))
      a_boot <- as.numeric(auc(roc(Xb$y, pb_boot, quiet = TRUE)))
      a_orig <- as.numeric(auc(roc(X$y, lp_orig, quiet = TRUE)))
      opt[b] <- a_boot - a_orig
      slopes[b] <- tryCatch(as.numeric(coef(glm(X$y ~ lp_orig, family = binomial))[2]), error = function(e) NA)
    }
    optim <- mean(opt, na.rm = TRUE); auc_corr <- auc_app - optim; slope_corr <- mean(slopes, na.rm = TRUE)
    aucs[[nm]] <- list(app = auc_app, lo = ci_app[1], hi = ci_app[3], corr = auc_corr, opt = optim)
    k <- length(models[[nm]])
    txt <- c(txt, sprintf("  %-36s k=%d EPV=%.1f  apparent AUC %.3f (%.3f-%.3f)  optimism %.3f  corrected AUC %.3f  calibration slope apparent %.2f, bootstrap-validated %.2f",
                          nm, k, ev / k, auc_app, ci_app[1], ci_app[3], optim, auc_corr, slope_app, slope_corr))
    co <- data.frame(term = names(coef(fit)), beta = coef(fit), lo = fit$ci.lower, hi = fit$ci.upper, p = fit$prob)
    for (i in seq_len(nrow(co))) if (co$term[i] != "(Intercept)")
      txt <- c(txt, sprintf("      %-20s OR per SD %.2f (%.2f-%.2f)  p=%.4f", co$term[i], exp(co$beta[i]), exp(co$lo[i]), exp(co$hi[i]), co$p[i]))
    out[[length(out) + 1]] <- data.frame(cohort = cohort, n = n, events = ev, model = nm, k = k, EPV = ev / k,
                                         AUC_apparent = auc_app, CI_lo = ci_app[1], CI_hi = ci_app[3], optimism = optim, AUC_corrected = auc_corr,
                                         cal_slope_apparent = slope_app, cal_slope_validated = slope_corr)
  }
  # penalized LRTs vs reference
  a2 <- anova(fits[["Reference + age + sex"]], fits[["Reference + age + sex + minimum CSA"]])
  txt <- c(txt, sprintf("  Age/sex-adjusted: minimum CSA added to reference + age + sex: penalized LRT chi2=%.2f p=%.4f; AUC %.3f -> %.3f (corrected %.3f -> %.3f)", a2$chisq, a2$pval,
                        aucs[["Reference + age + sex"]]$app, aucs[["Reference + age + sex + minimum CSA"]]$app, aucs[["Reference + age + sex"]]$corr, aucs[["Reference + age + sex + minimum CSA"]]$corr))
  for (nm in names(models)[2:3]) {
    a <- anova(fits[[1]], fits[[nm]])
    txt <- c(txt, sprintf("  Penalized LRT: %s vs reference: chi2=%.2f df=%d p=%.4f  | dAUC apparent %+.3f, dAUC optimism-corrected %+.3f",
                          nm, a$chisq, a$df, a$pval, aucs[[nm]]$app - aucs[[1]]$app, aucs[[nm]]$corr - aucs[[1]]$corr))
  }
  # DeLong on Firth predictions
  for (nm in names(models)[2:3]) {
    rt <- roc.test(roc(X$y, fits[[1]]$predict, quiet = TRUE), roc(X$y, fits[[nm]]$predict, quiet = TRUE), method = "delong")
    txt <- c(txt, sprintf("  DeLong (Firth-fitted probabilities): %s vs reference p=%.3f", nm, rt$p.value))
  }
  # LASSO
  xm <- as.matrix(X[, c(ref, air4)]); sel <- matrix(0, 20, ncol(xm), dimnames = list(NULL, colnames(xm))); sel1 <- sel
  for (r in 1:20) {
    cv <- cv.glmnet(xm, X$y, family = "binomial", alpha = 1, nfolds = 10, type.measure = "deviance")
    sel[r, ] <- as.numeric(coef(cv, s = "lambda.min")[-1] != 0)
    sel1[r, ] <- as.numeric(coef(cv, s = "lambda.1se")[-1] != 0)
  }
  txt <- c(txt, "  LASSO selection frequency over 20 repeated 10-fold CVs (lambda.min | lambda.1se):")
  for (j in colnames(xm)) txt <- c(txt, sprintf("      %-20s %.2f | %.2f", j, mean(sel[, j]), mean(sel1[, j])))
}
res <- do.call(rbind, out)
write.csv(res, file.path(fa, "32_penalized_icu_models.csv"), row.names = FALSE)
writeLines(c("Penalized (Firth) logistic regression — incremental ICU models (Table 4 re-analysis)", txt), file.path(fa, "32_penalized_icu_models.txt"))
cat(paste(c("Penalized (Firth) logistic regression — incremental ICU models (Table 4 re-analysis)", txt), collapse = "\n"), "\n")
