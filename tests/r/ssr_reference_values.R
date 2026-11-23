# Writes tests/ssr_reference_values.json (M9, sample size re-estimation).
# Run from the repository root:
#
#   Rscript tests/r/ssr_reference_values.R > ssr.json && mv ssr.json tests/ssr_reference_values.json
#
# Two kinds of reference value, both from rpact with an inverse normal design (the weights
# sqrt(t1) and sqrt(1 - t1) are fixed by the planned information rates, i.e. the
# Cui-Hung-Wang test):
#   cp   rpact::getConditionalPower at stage 1 of a log-rank dataset: the stage-1 z is given
#        exactly, so these test the conditional power formula to 1e-6.
#   sim  rpact::getSimulationMeans with conditionalPower and minNumberOfSubjectsPerStage =
#        planned (re-estimate upwards whenever CP < target, capped): trialdesign's rule with
#        cp_min = 0. trialdesign computes these exactly, so the test allows 3 MC SE (plus 1
#        patient for rpact's per-stage rounding up).

suppressMessages(library(rpact))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 12, format = "g")
q <- function(s) sprintf('"%s"', s)
row <- function(...) {
  a <- list(...)
  paste0("    {", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
}
design <- function(t1, type) getDesignInverseNormal(kMax = 2, alpha = 0.025, beta = 0.2,
  informationRates = c(t1, 1), typeOfDesign = type, sided = 1)
spend_name <- c(asOF = "obf", asP = "pocock")
rows <- character()

cp_cases <- list(
  list(id = "cp_obf_t50_z1.2", t1 = 0.5, type = "asOF", z1 = 1.2, d1 = 150, n2 = 150, hr = 0.7),
  list(id = "cp_obf_t50_z0.5_more", t1 = 0.5, type = "asOF", z1 = 0.5, d1 = 150, n2 = 300, hr = 0.75),
  list(id = "cp_pocock_t40_z1.8", t1 = 0.4, type = "asP", z1 = 1.8, d1 = 120, n2 = 180, hr = 0.8),
  list(id = "cp_obf_t60_z2.2_fewer", t1 = 0.6, type = "asOF", z1 = 2.2, d1 = 200, n2 = 80, hr = 0.7),
  list(id = "cp_obf_t50_neg", t1 = 0.5, type = "asOF", z1 = -0.4, d1 = 150, n2 = 150, hr = 0.7)
)
for (cs in cp_cases) {
  d <- design(cs$t1, cs$type)
  ds <- getDataset(overallEvents = cs$d1, overallLogRanks = -cs$z1, overallAllocationRatios = 1)
  sr <- getStageResults(d, dataInput = ds, directionUpper = FALSE)
  cp <- getConditionalPower(sr, nPlanned = cs$n2, thetaH1 = cs$hr)
  rows <- c(rows, row(id = q(cs$id), kind = q("cp"), source = q("rpact::getConditionalPower"),
    interim = fmt(cs$t1), efficacy = q(spend_name[[cs$type]]), z1 = fmt(cs$z1),
    stage2_events = fmt(cs$n2), hazard_ratio = fmt(cs$hr), allocation = 1,
    c2 = fmt(d$criticalValues[2]), conditional_power = fmt(cp$conditionalPower[2])))
}

nsim <- 100000
sim_cases <- list(
  list(id = "sim_obf_t50_h0", t1 = 0.5, type = "asOF", alt = 0, n = 352, nmax2 = 352),
  list(id = "sim_obf_t50_h1", t1 = 0.5, type = "asOF", alt = 0.3, n = 352, nmax2 = 352),
  list(id = "sim_obf_t50_smaller_effect", t1 = 0.5, type = "asOF", alt = 0.22, n = 352, nmax2 = 352),
  list(id = "sim_pocock_t60_h1", t1 = 0.6, type = "asP", alt = 0.3, n = 400, nmax2 = 320)
)
for (cs in sim_cases) {
  d <- design(cs$t1, cs$type)
  n1 <- cs$t1 * cs$n
  x <- getSimulationMeans(d, groups = 2, alternative = cs$alt, stDev = 1,
    plannedSubjects = c(n1, cs$n), normalApproximation = TRUE, conditionalPower = 0.8,
    minNumberOfSubjectsPerStage = c(NA, cs$n - n1),
    maxNumberOfSubjectsPerStage = c(NA, cs$nmax2), maxNumberOfIterations = nsim, seed = 2026)
  rows <- c(rows, row(id = q(cs$id), kind = q("sim"), source = q("rpact::getSimulationMeans"),
    interim = fmt(cs$t1), efficacy = q(spend_name[[cs$type]]), delta = fmt(cs$alt), sd = 1,
    planned_n_total = fmt(cs$n), max_n_total = fmt(n1 + cs$nmax2), cp_target = 0.8,
    n_sim = nsim, seed = 2026, reject_prob = fmt(x$overallReject),
    prob_efficacy_interim = fmt(x$rejectPerStage[1]),
    expected_n_total = fmt(x$expectedNumberOfSubjects)))
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s; rpact %s",\n', R.version.string, packageVersion("rpact")))
cat('  "ssr": [\n')
cat(paste(rows, collapse = ",\n"))
cat("\n  ]\n}\n")
