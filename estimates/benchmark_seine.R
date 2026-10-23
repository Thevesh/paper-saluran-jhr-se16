# Covariate-adjusted semiparametric ecological inference (McCartan & Kuriwaki 2025+), via
# the `seine` package, as a frontier comparator for the paper's constant-rate joint model.
#
#   Rscript benchmark_seine.R
#
# WHAT IT DOES. For GE-15 and SE-16, fit seine's debiased/semiparametric estimator of group
# rates -- predictors = the four ethnic shares, outcomes = every choice as a share of the
# electorate (BN, PH, PN, [BERSAMA], OTHER, REJECTED, NOTRETURNED, NONVOTE), which sum to one
# -- under conditional ignorability GIVEN area-level covariates: log polling-district
# electorate, log saluran electorate, the district's lagged outcome shares at the previous
# election, and the saluran's age composition (counted on the roll). A natural-spline basis
# of the covariates is the `preproc`. Three outputs per election:
#
#   1. DML estimates of each group's rate for each outcome, with SEs, compared with the
#      paper's joint-model marginals (joint_marginals.csv) and its held-out scores;
#   2. robustness values (Chernozhukov et al. 2024 style): the minimum strength of an
#      omitted variable -- as a partial R^2 with the outcome, and the analogous share of the
#      Riesz representer's variation, taken equal -- that would move an estimate by (a) five
#      points, (b) two standard errors, and (c) the paper's own held-out discrepancy score for
#      that group and estimand (Malay/Chinese only). These are sensitivity parameters, not
#      partial R^2s with the group share in any literal sense;
#   3. the CAR specification test (ei_test_car): whether the partially linear form of the
#      conditional expectation function IMPLIED by conditional ignorability fits as well as an
#      unrestricted nonparametric one. Rejection means this specification conflicts with that
#      implication; it does not by itself disprove conditional ignorability under some other
#      covariate set.
#
# ENVIRONMENT. seine is a development package: installed with
#   remotes::install_github("CoryMcCartan/seine", ref = "92e60fc867d90eb42ef2a28eb4d01b56a616c5a6")
# (version 0.1.2.9999). On macOS with R's CRAN build, Fortran linking may need ~/.R/Makevars
# pointing FLIBS at a Homebrew gfortran. Inputs come from seine_input.py.
#
# WHY 4 ROWS AND NOT 24. seine estimates a conditional-mean function per predictor group;
# with 24 joint cells the design faces a positivity problem (most cells are near-zero in most
# streams), and the review that prompted this benchmark said so. Age enters as covariates.
#
# Outputs: output/benchmark_seine.csv, output/benchmark_seine_rv.csv, output/benchmark_seine_car.csv

suppressPackageStartupMessages({ library(seine); library(splines) })
OUT <- file.path(if (basename(getwd()) == "estimates") getwd() else file.path(getwd(), "estimates"), "output")

ETH <- c("Malay", "Chinese", "Indian", "Other")
marg <- read.csv(file.path(OUT, "joint_marginals.csv"))
verr <- read.csv(file.path(OUT, "validation_errors.csv"))

res <- list(); rvs <- list(); cars <- list()
for (el in c("ge15", "jhr_se16")) {
  d <- read.csv(file.path(OUT, sprintf("seine_input_%s.csv", el)))
  ycols <- grep("^y_", names(d), value = TRUE); ycols <- setdiff(ycols, "y_turnout")
  xcols <- paste0("x_", ETH)
  # guard the sum-to-one constraints against floating error
  d[xcols] <- d[xcols] / rowSums(d[xcols]); d[ycols] <- d[ycols] / rowSums(d[ycols])
  covs <- c("log_dm_size", "log_size", "lag_BN", "lag_PH", "lag_PN", "lag_NONVOTE",
            paste0("age_", c("18.29", "30.39", "40.49", "50.59", "60.69", "70.")))
  covs <- intersect(covs, names(d))
  preproc <- function(z) {
    z <- as.data.frame(z)
    do.call(cbind, lapply(names(z), function(v) {
      x <- z[[v]]
      if (length(unique(x)) < 6) return(matrix(x, ncol = 1, dimnames = list(NULL, v)))
      k <- unique(quantile(x, c(0.25, 0.5, 0.75)))
      k <- k[k > min(x) & k < max(x)]
      # age shares are zero in most streams (the sorting rule), so quantile knots can
      # collapse onto the boundary; fall back to a quadratic when they do
      m <- if (length(k) >= 1) splines::ns(x, knots = k) else poly(x, 2)
      colnames(m) <- paste0(v, "_b", seq_len(ncol(m))); m
    }))
  }
  spec <- ei_spec(d, predictors = tidyselect::all_of(xcols), outcome = tidyselect::all_of(ycols), total = V,
                  covariates = tidyselect::all_of(covs), preproc = preproc)
  cat(sprintf("\n== %s: %d salurans, %d covariates -> %d basis columns\n", el, nrow(d), length(covs),
              ncol(preproc(d[covs]))))

  # NB: bounds = c(0, 1) turns the ridge into a per-unit constrained QP that scales badly
  # (27 s for 800 streams); left unbounded here, since the comparison is of statewide
  # marginals and the paper's own raking step imposes the bounds on anything it reports.
  m  <- ei_ridge(spec)
  rr <- ei_riesz(spec, penalty = m$penalty)
  est <- ei_est(regr = m, riesz = rr, data = spec, conf_level = 0.95)
  e <- as.data.frame(est)
  e$election <- el
  # turnout = 1 - NONVOTE
  nv <- e[e$outcome == "y_NONVOTE", ]
  tn <- nv; tn$outcome <- "y_turnout"; tn$estimate <- 1 - nv$estimate
  tn$conf.low <- 1 - nv$conf.high; tn$conf.high <- 1 - nv$conf.low
  e <- rbind(e, tn)
  e$group <- sub("^x_", "", e$predictor); e$estimand <- sub("^y_", "", e$outcome)
  e <- merge(e, marg[marg$election == el, c("group", "estimand", "rate")], by = c("group", "estimand"), all.x = TRUE)
  names(e)[names(e) == "rate"] <- "paper_rate"
  e <- merge(e, verr[verr$election == el, c("group", "estimand", "err")], by = c("group", "estimand"), all.x = TRUE)
  names(e)[names(e) == "err"] <- "paper_score"
  e$diff_pp <- 100 * (e$estimate - e$paper_rate)
  res[[el]] <- e
  cat("\nseine DML vs paper joint model (pp), validated groups:\n")
  v <- e[e$group %in% c("Malay", "Chinese") & e$estimand %in% c("turnout", "BN", "PH", "PN"), ]
  v <- v[order(v$group, v$estimand), ]
  print(data.frame(group = v$group, estimand = v$estimand, seine = round(100 * v$estimate, 1),
                   se = round(100 * v$std.error, 2), paper = round(100 * v$paper_rate, 1),
                   diff = round(v$diff_pp, 2), score = round(100 * v$paper_score, 2)), row.names = FALSE)

  # robustness values: confounding needed to move the estimate by the paper's held-out score,
  # and to move it by a 'substantial' 5 points
  rv1 <- as.data.frame(ei_sens_rv(est, 0.05)); rv1$bias_bound <- "5pp"
  rv2 <- as.data.frame(ei_sens_rv(est, 2 * std.error)); rv2$bias_bound <- "2SE"
  # (c) the paper's own discrepancy score as the bias bound, row by row
  sc <- merge(data.frame(predictor = est$predictor, outcome = est$outcome),
              transform(verr[verr$election == el, c("group", "estimand", "err")],
                        predictor = paste0("x_", group), outcome = paste0("y_", estimand)),
              by = c("predictor", "outcome"), all.x = TRUE, sort = FALSE)
  sc <- sc[match(paste(est$predictor, est$outcome), paste(sc$predictor, sc$outcome)), ]
  est$score <- sc$err
  rv3 <- as.data.frame(ei_sens_rv(est, score)); rv3$bias_bound <- "paper score"
  rv3 <- rv3[!is.na(rv3$rv), ]
  rv <- rbind(rv1, rv2, rv3[, names(rv1)]); rv$election <- el
  rv$group <- sub("^x_", "", rv$predictor); rv$estimand <- sub("^y_", "", rv$outcome)
  rvs[[el]] <- rv
  cat("\nrobustness values (omitted-variable strength to bias by 5pp / by the paper's score), validated groups:\n")
  w <- rv[rv$bias_bound %in% c("5pp", "paper score") & rv$group %in% c("Malay", "Chinese") & rv$estimand %in% c("BN", "PH", "PN"), ]
  print(reshape(data.frame(group = w$group, estimand = w$estimand, bias_bound = w$bias_bound, rv = round(w$rv, 3)),
                idvar = c("group", "estimand"), timevar = "bias_bound", direction = "wide"), row.names = FALSE)

  # CAR test
  ct <- tryCatch(as.data.frame(ei_test_car(spec, use_chisq = TRUE)), error = function(err) { cat("CAR test failed:", conditionMessage(err), "\n"); NULL })
  if (!is.null(ct)) { ct$election <- el; cars[[el]] <- ct; cat("\nCAR test p-values:\n"); print(ct[, intersect(c("outcome", "statistic", "p.value"), names(ct))], row.names = FALSE) }
}
R <- do.call(rbind, res); write.csv(R, file.path(OUT, "benchmark_seine.csv"), row.names = FALSE)
write.csv(do.call(rbind, rvs), file.path(OUT, "benchmark_seine_rv.csv"), row.names = FALSE)
if (length(cars)) write.csv(do.call(rbind, cars), file.path(OUT, "benchmark_seine_car.csv"), row.names = FALSE)
cat("\nwrote benchmark_seine.csv, benchmark_seine_rv.csv, benchmark_seine_car.csv\n")
