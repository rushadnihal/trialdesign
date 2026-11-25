# Writes tests/patients_endpoints_reference_values.json (M13, patient-level normal and binary
# simulation). Run from the repository root:
#
#   Rscript tests/r/patients_endpoints_reference_values.R > pe.json && mv pe.json tests/patients_endpoints_reference_values.json
#
# rpact::getSimulationMeans with normalApproximation = FALSE (estimated standard deviation)
# and rpact::getSimulationRates with normalApproximation = TRUE (the
# z-test with the Farrington-Manning variance under H0), each on a group-sequential design
# without adaptation. plannedSubjects are cumulative totals over both arms; rpact's group 1 is
# treatment and allocationRatioPlanned = n_T / n_C. Each case runs under the design
# alternative and on the H0 boundary (effect = thetaH0).
#
# The normal cases use at least 86 patients per arm per stage, where rpact agrees with
# trialdesign's default test (the t statistic turned into the z-value with the same p-value)
# within Monte Carlo error. At 30 per arm per stage they differ: rpact gives a type I error of
# about 0.0267 for three O'Brien-Fleming looks, trialdesign 0.0250, so that case is checked as
# a property in tests/test_patients_endpoints.py instead.

suppressMessages(library(rpact))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 12, format = "g")
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
obj <- function(a) paste0("{", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
n_iter <- 100000L

design <- function(K, info, type, beta_spending) {
  args <- list(kMax = K, alpha = 0.025, beta = 0.2, sided = 1, typeOfDesign = type,
    typeBetaSpending = beta_spending)
  if (!is.null(info)) args$informationRates <- info
  if (beta_spending != "none") args$bindingFutility <- FALSE
  do.call(getDesignGroupSequential, args)
}
bounds <- function(d) {
  fut <- if (d$typeBetaSpending != "none") c(d$futilityBounds, d$criticalValues[d$kMax]) else NULL
  list(efficacy = arr(d$criticalValues), futility = if (is.null(fut)) "null" else arr(fut))
}
oc <- function(sim, K, has_fut) {
  fut <- if (has_fut) c(sim$futilityPerStage[, 1], 0) else rep(0, K)
  list(n_sim = n_iter, reject_prob = fmt(sim$overallReject),
    reject_per_look = arr(sim$rejectPerStage[, 1]), futility_per_look = arr(fut[1:K]),
    expected_n_total = fmt(sim$expectedNumberOfSubjects))
}

out <- character()

means <- list(
  list(id = "means_obf3_t", K = 3, info = NULL, type = "asOF", beta = "none", delta = 0.25,
    sd = 1, theta0 = 0, alloc = 1, planned = c(172, 344, 516), seed = 11),
  list(id = "means_ni_pocock2_2to1", K = 2, info = NULL, type = "asP", beta = "none",
    delta = 0.0, sd = 1.5, theta0 = -0.3, alloc = 2, planned = c(450, 900), seed = 12)
)
for (cs in means) {
  d <- design(cs$K, cs$info, cs$type, cs$beta)
  for (alt in c(cs$delta, cs$theta0)) {
    sim <- getSimulationMeans(d, normalApproximation = FALSE, thetaH0 = cs$theta0,
      alternative = alt, stDev = cs$sd, plannedSubjects = cs$planned,
      allocationRatioPlanned = cs$alloc, maxNumberOfIterations = n_iter, seed = cs$seed)
    n_c <- cs$planned / (1 + cs$alloc)
    a <- c(list(id = sprintf('"%s_%s"', cs$id, if (alt == cs$delta) "h1" else "h0"),
      endpoint = '"normal"', delta = fmt(alt), sd = fmt(cs$sd), delta0 = fmt(cs$theta0),
      n_control = arr(n_c), n_treatment = arr(cs$planned - n_c)),
      bounds(d), oc(sim, cs$K, FALSE))
    out <- c(out, paste0("    ", obj(a)))
  }
}

rates <- list(
  list(id = "rates_obf3_futility", K = 3, info = NULL, type = "asOF", beta = "bsOF",
    pc = 0.30, pt = 0.45, theta0 = 0, alloc = 1, planned = c(120, 240, 360), seed = 21),
  list(id = "rates_ni_obf2_2to1", K = 2, info = NULL, type = "asOF", beta = "none",
    pc = 0.70, pt = 0.70, theta0 = -0.1, alloc = 2, planned = c(225, 450), seed = 22)
)
for (cs in rates) {
  d <- design(cs$K, cs$info, cs$type, cs$beta)
  for (pt in c(cs$pt, cs$pc + cs$theta0)) {
    sim <- getSimulationRates(d, normalApproximation = TRUE, thetaH0 = cs$theta0, pi1 = pt,
      pi2 = cs$pc, plannedSubjects = cs$planned, allocationRatioPlanned = cs$alloc,
      maxNumberOfIterations = n_iter, seed = cs$seed)
    n_c <- cs$planned / (1 + cs$alloc)
    a <- c(list(id = sprintf('"%s_%s"', cs$id, if (pt == cs$pt) "h1" else "h0"),
      endpoint = '"binary"', p_control = fmt(cs$pc), p_treatment = fmt(pt),
      delta0 = fmt(cs$theta0), n_control = arr(n_c), n_treatment = arr(cs$planned - n_c)),
      bounds(d), oc(sim, cs$K, cs$beta != "none"))
    out <- c(out, paste0("    ", obj(a)))
  }
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s; rpact %s",\n', R.version.string, packageVersion("rpact")))
cat('  "cases": [\n')
cat(paste(out, collapse = ",\n"))
cat("\n  ]\n}\n")
