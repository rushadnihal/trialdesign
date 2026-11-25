# Writes tests/ssr_survival_reference_values.json (M14, event re-estimation in survival trials).
# Run from the repository root:
#
#   Rscript tests/r/ssr_survival_reference_values.R > ssr.json && mv ssr.json tests/ssr_survival_reference_values.json
#
# Two parts, both on a two-look inverse normal design (weights sqrt(t1), sqrt(1 - t1), which
# is the CHW test of trialdesign):
#
# 1. getConditionalPower for a survival dataset: the conditional power of the final test given
#    the interim log-rank z and a planned number of stage-2 events, at an assumed hazard ratio.
#    rpact's group 1 is treatment; with directionUpper = FALSE a benefit (HR < 1) is a negative
#    log-rank statistic, so the dataset holds -z1 where trialdesign has z1.
# 2. getSimulationSurvival with event re-estimation: at the interim, the stage-2 events are set
#    to reach conditionalPower at the observed hazard ratio, between minNumberOfEventsPerStage
#    (the planned D - D1) and maxNumberOfEventsPerStage (cap * D - D1). That is the promising-
#    zone rule with cp_min = 0: every continuing trial with conditional power below the target
#    gets more events. Patients enter evenly over the accrual period and are followed until the
#    new number of events. Simulated values carry Monte Carlo error; n_sim is written beside them.

suppressMessages(library(rpact))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 12, format = "g")
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
obj <- function(a) paste0("{", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
n_iter <- 100000L

design <- function(beta_spending) {
  args <- list(kMax = 2, alpha = 0.025, beta = 0.2, sided = 1, informationRates = c(0.5, 1),
    typeOfDesign = "asOF", typeBetaSpending = beta_spending)
  if (beta_spending != "none") args$bindingFutility <- FALSE
  do.call(getDesignInverseNormal, args)
}

# ------------------------------------------------------------------ conditional power
cp_cases <- list(
  list(z1 = 1.2, d1 = 124, n2 = 150, hr = 0.7, alloc = 1),
  list(z1 = 0.5, d1 = 124, n2 = 248, hr = 0.75, alloc = 1),
  list(z1 = 2.0, d1 = 200, n2 = 120, hr = 0.8, alloc = 2),
  list(z1 = -0.3, d1 = 90, n2 = 180, hr = 0.65, alloc = 0.5)
)
cp_out <- character()
d <- design("none")
for (cs in cp_cases) {
  ds <- getDataset(overallEvents = cs$d1, overallLogRanks = -cs$z1,
    overallAllocationRatios = cs$alloc)
  sr <- getStageResults(d, ds, stage = 1, directionUpper = FALSE)
  cp <- getConditionalPower(sr, nPlanned = cs$n2, thetaH1 = cs$hr,
    allocationRatioPlanned = cs$alloc)
  a <- list(z1 = fmt(cs$z1), interim_events = cs$d1, stage2_events = cs$n2,
    hazard_ratio = fmt(cs$hr), allocation = fmt(cs$alloc),
    conditional_power = fmt(cp$conditionalPower[2]))
  cp_out <- c(cp_out, paste0("    ", obj(a)))
}

# ------------------------------------------------------------------ adaptive simulation
sim_cases <- list(
  list(id = "obf_1to1", beta_spending = "none", hr = 0.7, median = 12, alloc = c(1, 1),
    dropout = 0, accrual_time = 18, events = 248, n_total = 600, cap = 1.5, seed = 141),
  list(id = "obf_futility_2to1_dropout", beta_spending = "bsOF", hr = 0.75, median = 9,
    alloc = c(2, 1), dropout = 0.05, accrual_time = 24, events = 430, n_total = 1100,
    cap = 1.75, seed = 142)
)
sim_out <- character()
for (cs in sim_cases) {
  d <- design(cs$beta_spending)
  d1 <- round(0.5 * cs$events)
  d_max <- ceiling(cs$cap * cs$events)
  for (hr_true in c(cs$hr, 1)) {
    args <- list(design = d, directionUpper = FALSE, hazardRatio = hr_true,
      lambda2 = log(2) / cs$median, allocation1 = cs$alloc[1], allocation2 = cs$alloc[2],
      accrualTime = c(0, cs$accrual_time), maxNumberOfSubjects = cs$n_total,
      plannedEvents = c(d1, cs$events), minNumberOfEventsPerStage = c(NA, cs$events - d1),
      maxNumberOfEventsPerStage = c(NA, d_max - d1), conditionalPower = 0.8,
      maxNumberOfIterations = n_iter, seed = cs$seed, longTimeSimulationAllowed = TRUE)
    if (cs$dropout > 0) {
      args$dropoutRate1 <- cs$dropout
      args$dropoutRate2 <- cs$dropout
      args$dropoutTime <- 12
    }
    sim <- do.call(getSimulationSurvival, args)
    a <- list(
      id = sprintf('"%s_hr%s"', cs$id, format(hr_true)),
      hazard_ratio = fmt(hr_true),
      design_hazard_ratio = fmt(cs$hr),
      median_control = fmt(cs$median),
      allocation = fmt(cs$alloc[1] / cs$alloc[2]),
      dropout = fmt(cs$dropout),
      accrual_time = fmt(cs$accrual_time),
      events = cs$events,
      n_total = cs$n_total,
      max_factor = fmt(cs$cap),
      efficacy = arr(d$criticalValues),
      futility = if (cs$beta_spending == "none") "null" else arr(d$futilityBounds),
      n_sim = n_iter,
      reject_prob = fmt(sim$overallReject),
      reject_interim = fmt(sim$rejectPerStage[1, 1]),
      futility_interim = fmt(if (cs$beta_spending == "none") 0 else sim$futilityPerStage[1, 1]),
      study_duration = fmt(sim$studyDuration),
      expected_patients = fmt(sim$expectedNumberOfSubjects),
      expected_events = fmt(sim$expectedNumberOfEvents)
    )
    sim_out <- c(sim_out, paste0("    ", obj(a)))
  }
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s; rpact %s",\n', R.version.string, packageVersion("rpact")))
cat('  "conditional_power": [\n')
cat(paste(cp_out, collapse = ",\n"))
cat('\n  ],\n  "simulation": [\n')
cat(paste(sim_out, collapse = ",\n"))
cat("\n  ]\n}\n")
