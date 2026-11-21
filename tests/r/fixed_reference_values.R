# Reference values for M1 (fixed designs).
#
# Regenerate from the repository root with:
#   Rscript tests/r/fixed_reference_values.R > tests/reference_values.json
#
# Primary values (written to the JSON and asserted by tests/test_fixed.py):
#   normal   stats::power.t.test   (two-sample, strict = FALSE)
#   binary   stats::power.prop.test (no continuity correction, strict = FALSE)
#   survival gsDesign::nEvents      (Schoenfeld 1983)
# Cross-checks with rpact are printed to stderr (not saved) and the script stops if they
# disagree with the primary value by more than the stated tolerance.

suppressPackageStartupMessages({
  library(gsDesign)
  library(rpact)
})

fmt <- function(x) formatC(x, digits = 12, format = "g")
q <- function(s) sprintf('"%s"', s)
row <- function(...) {
  a <- list(...)
  paste0("    {", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
}
# rpact calls are wrapped so an API difference between rpact versions gives a warning rather
# than stopping the run; a numerical disagreement still stops it.
rp <- function(expr) tryCatch(expr, error = function(e) {
  warning("rpact call failed: ", conditionMessage(e), call. = FALSE)
  NA_real_
})
# Fixed-design events: rpact names the field eventsFixed (kMax = 1) or maxNumberOfEvents.
rp_events <- function(x) {
  v <- tryCatch(x$eventsFixed, error = function(e) NULL)
  if (is.null(v) || all(is.na(v))) v <- x$maxNumberOfEvents
  as.numeric(v[1])
}
crosscheck <- function(label, primary, other, tol) {
  if (is.na(other)) {
    message(sprintf("%-34s primary %.6f  rpact NA (skipped)", label, primary))
    return(invisible())
  }
  diff <- abs(primary - other)
  message(sprintf("%-34s primary %.6f  rpact %.6f  |diff| %.2e", label, primary, other, diff))
  if (diff > tol) stop(sprintf("rpact cross-check failed for %s", label))
}

# Two-sided fixed design for rpact (kMax = 1).
fixed_design <- function(alpha_two_sided, power) {
  getDesignGroupSequential(kMax = 1, alpha = alpha_two_sided, beta = 1 - power, sided = 2)
}

# ---------------------------------------------------------------------------- normal
n_norm_1 <- power.t.test(delta = 0.5, sd = 1, power = 0.8)$n
n_norm_2 <- power.t.test(delta = 3, sd = 10, power = 0.9, sig.level = 0.01)$n
pw_norm <- power.t.test(n = 40, delta = 0.5, sd = 1)$power

crosscheck("normal d=0.5 (t)", n_norm_1,
  rp(getSampleSizeMeans(fixed_design(0.05, 0.8), alternative = 0.5, stDev = 1,
                        normalApproximation = FALSE)$nFixed1), tol = 0.01)
crosscheck("normal d=3 sd=10 a=.01 p=.9 (t)", n_norm_2,
  rp(getSampleSizeMeans(fixed_design(0.01, 0.9), alternative = 3, stDev = 10,
                        normalApproximation = FALSE)$nFixed1), tol = 0.01)

# ---------------------------------------------------------------------------- binary
n_bin_1 <- power.prop.test(p1 = 0.30, p2 = 0.45, power = 0.8)$n
n_bin_2 <- power.prop.test(p1 = 0.10, p2 = 0.20, power = 0.9)$n
pw_bin <- power.prop.test(n = 100, p1 = 0.30, p2 = 0.45)$power

crosscheck("binary .30 vs .45", n_bin_1,
  rp(getSampleSizeRates(fixed_design(0.05, 0.8), pi1 = 0.45, pi2 = 0.30,
                        normalApproximation = TRUE)$nFixed1), tol = 0.01)
crosscheck("binary .10 vs .20 p=.9", n_bin_2,
  rp(getSampleSizeRates(fixed_design(0.05, 0.9), pi1 = 0.20, pi2 = 0.10,
                        normalApproximation = TRUE)$nFixed1), tol = 0.01)

# Unequal allocation (n_T / n_C = 2): no base-R function, so gsDesign::nBinomial is primary.
# nBinomial returns the total n; ratio = n2 / n1 with group 1 = control here.
n_bin_2to1_total <- nBinomial(p1 = 0.30, p2 = 0.45, alpha = 0.025, beta = 0.2, sided = 1,
                              ratio = 2, outtype = 1)
n_bin_2to1 <- n_bin_2to1_total / 3
crosscheck("binary .30 vs .45, 2:1 (n_C)", n_bin_2to1,
  rp(getSampleSizeRates(fixed_design(0.05, 0.8), pi1 = 0.45, pi2 = 0.30,
                        allocationRatioPlanned = 2, normalApproximation = TRUE)$nFixed2),
  tol = 0.01)

# -------------------------------------------------------------------------- survival
ev_1 <- nEvents(hr = 0.70, alpha = 0.025, beta = 0.2, ratio = 1, sided = 1)
ev_2 <- nEvents(hr = 0.75, alpha = 0.025, beta = 0.1, ratio = 2, sided = 1)

crosscheck("survival HR 0.70 1:1", ev_1,
  rp(rp_events(getSampleSizeSurvival(fixed_design(0.05, 0.8), hazardRatio = 0.70,
                                     lambda2 = log(2) / 12, allocationRatioPlanned = 1))),
  tol = 0.01)
crosscheck("survival HR 0.75 2:1", ev_2,
  rp(rp_events(getSampleSizeSurvival(fixed_design(0.05, 0.9), hazardRatio = 0.75,
                                     lambda2 = log(2) / 12, allocationRatioPlanned = 2))),
  tol = 0.01)

# ------------------------------------------------------------------------------ JSON
rows <- c(
  row(id = q("normal_n_d0.5"), source = q("stats::power.t.test"), delta = 0.5, sd = 1,
      alpha_two_sided = 0.05, power = 0.8, n_per_arm = fmt(n_norm_1)),
  row(id = q("normal_n_d3_sd10_a01_p90"), source = q("stats::power.t.test"), delta = 3, sd = 10,
      alpha_two_sided = 0.01, power = 0.9, n_per_arm = fmt(n_norm_2)),
  row(id = q("normal_power_n40"), source = q("stats::power.t.test"), delta = 0.5, sd = 1,
      alpha_two_sided = 0.05, n_per_arm = 40, power = fmt(pw_norm)),
  row(id = q("binary_n_30_45"), source = q("stats::power.prop.test"), p_control = 0.30,
      p_treatment = 0.45, alpha_two_sided = 0.05, power = 0.8, n_per_arm = fmt(n_bin_1)),
  row(id = q("binary_n_10_20_p90"), source = q("stats::power.prop.test"), p_control = 0.10,
      p_treatment = 0.20, alpha_two_sided = 0.05, power = 0.9, n_per_arm = fmt(n_bin_2)),
  row(id = q("binary_power_n100"), source = q("stats::power.prop.test"), p_control = 0.30,
      p_treatment = 0.45, alpha_two_sided = 0.05, n_per_arm = 100, power = fmt(pw_bin)),
  row(id = q("binary_n_30_45_2to1"), source = q("gsDesign::nBinomial"), p_control = 0.30,
      p_treatment = 0.45, alpha_two_sided = 0.05, power = 0.8, allocation = 2,
      n_control = fmt(n_bin_2to1)),
  row(id = q("survival_events_hr0.7"), source = q("gsDesign::nEvents"), hazard_ratio = 0.7,
      alpha_two_sided = 0.05, power = 0.8, allocation = 1, events = fmt(ev_1)),
  row(id = q("survival_events_hr0.75_2to1"), source = q("gsDesign::nEvents"),
      hazard_ratio = 0.75, alpha_two_sided = 0.05, power = 0.9, allocation = 2,
      events = fmt(ev_2))
)

cat("{\n")
cat(sprintf('  "_generated_with": "%s; gsDesign %s; rpact %s",\n', R.version.string,
            packageVersion("gsDesign"), packageVersion("rpact")))
cat('  "fixed": [\n')
cat(paste(rows, collapse = ",\n"))
cat("\n  ]\n}\n")
