# Writes tests/patients_reference_values.json (M12, patient-level survival simulation).
# Run from the repository root:
#
#   Rscript tests/r/patients_reference_values.R > pat.json && mv pat.json tests/patients_reference_values.json
#
# Each case builds an rpact group-sequential design and its survival sample size
# (getSampleSizeSurvival: Lachin-Foulkes accrual, exponential events and dropout), then
# simulates patients with getSimulationSurvival: evenly spaced entry, blocks of
# allocation1 + allocation2 patients, log-rank test at the planned cumulative events.
# Time is in months. rpact's group 1 is treatment; directionUpper = FALSE because the
# alternative is HR < 1, which makes rpact's z positive when treatment does better (the
# trialdesign convention). Simulated values carry Monte Carlo error; their SEs are written
# beside them so the Python tests can use a 2 MC SE band on the difference.

suppressMessages(library(rpact))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 12, format = "g")
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
obj <- function(a) paste0("{", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
n_iter <- 100000L

cases <- list(
  list(id = "obf3_uniform", K = 3, info = NULL, design = "asOF", beta_spending = "none",
    hr = 0.7, median = 12, alloc = c(1, 1), dropout = 0,
    accrual_time = 24, accrual_rate = NULL, followup = 12, seed = 101),
  list(id = "obf2_futility_piecewise_2to1", K = 2, info = c(0.5, 1), design = "asOF",
    beta_spending = "bsOF", hr = 0.75, median = 9, alloc = c(2, 1), dropout = 0.1,
    accrual_time = 6, accrual_rate = c(10, 25), followup = 12, seed = 202),
  list(id = "pocock4_dropout", K = 4, info = NULL, design = "asP", beta_spending = "none",
    hr = 0.65, median = 6, alloc = c(1, 1), dropout = 0.05,
    accrual_time = 18, accrual_rate = NULL, followup = 6, seed = 303)
)

out <- character()
for (cs in cases) {
  dargs <- list(kMax = cs$K, alpha = 0.025, beta = 0.2, sided = 1, typeOfDesign = cs$design,
    typeBetaSpending = cs$beta_spending)
  if (!is.null(cs$info)) dargs$informationRates <- cs$info
  if (cs$beta_spending != "none") dargs$bindingFutility <- FALSE
  d <- do.call(getDesignGroupSequential, dargs)
  alloc_ratio <- cs$alloc[1] / cs$alloc[2]
  sargs <- list(design = d, hazardRatio = cs$hr, lambda2 = log(2) / cs$median,
    allocationRatioPlanned = alloc_ratio, accrualTime = c(0, cs$accrual_time),
    followUpTime = cs$followup)
  if (!is.null(cs$accrual_rate)) sargs$accrualIntensity <- cs$accrual_rate
  if (cs$dropout > 0) {
    sargs$dropoutRate1 <- cs$dropout
    sargs$dropoutRate2 <- cs$dropout
    sargs$dropoutTime <- 12
  }
  ss <- do.call(getSampleSizeSurvival, sargs)
  d_max <- ceiling(ss$maxNumberOfEvents)
  planned <- round(d$informationRates * d_max)
  n_total <- ceiling(ss$maxNumberOfSubjects)
  acc_end <- ss$accrualTime  # without the leading zero
  if (is.null(cs$accrual_rate)) {
    sim_acc <- list(accrualTime = c(0, cs$accrual_time), maxNumberOfSubjects = n_total)
  } else {
    # stretch the last segment so the intensities enrol exactly n_total patients
    k <- length(acc_end)
    before <- sum(cs$accrual_rate[-k] * diff(c(0, acc_end[-k])))
    acc_end[k] <- acc_end[k - 1] + (n_total - before) / cs$accrual_rate[k]
    sim_acc <- list(accrualTime = c(0, acc_end), accrualIntensity = cs$accrual_rate)
  }
  futility <- if (cs$beta_spending != "none") c(d$futilityBounds, d$criticalValues[cs$K]) else NULL
  for (hr_true in c(cs$hr, 1)) {
    simargs <- c(list(design = d, hazardRatio = hr_true, lambda2 = log(2) / cs$median,
      allocation1 = cs$alloc[1], allocation2 = cs$alloc[2], plannedEvents = planned,
      directionUpper = FALSE, maxNumberOfIterations = n_iter, seed = cs$seed), sim_acc)
    if (cs$dropout > 0) {
      simargs$dropoutRate1 <- cs$dropout
      simargs$dropoutRate2 <- cs$dropout
      simargs$dropoutTime <- 12
    }
    sim <- do.call(getSimulationSurvival, simargs)
    p <- sim$overallReject
    fut <- if (is.null(futility)) rep(0, cs$K) else sim$futilityPerStage[, 1]
    fut <- c(fut, rep(0, cs$K - length(fut)))
    a <- list(
      id = sprintf('"%s_hr%s"', cs$id, format(hr_true)),
      hazard_ratio = fmt(hr_true),
      design_hazard_ratio = fmt(cs$hr),
      median_control = fmt(cs$median),
      allocation = fmt(alloc_ratio),
      dropout = fmt(cs$dropout),
      accrual_time = arr(cs$accrual_time),
      accrual_rate = if (is.null(cs$accrual_rate)) "null" else arr(cs$accrual_rate),
      followup = fmt(cs$followup),
      design_events = fmt(ss$maxNumberOfEvents),
      events = arr(planned),
      n_total = n_total,
      accrual_end = arr(acc_end),
      efficacy = arr(d$criticalValues),
      futility = if (is.null(futility)) "null" else arr(futility),
      n_sim = n_iter,
      reject_prob = fmt(p),
      reject_per_look = arr(sim$rejectPerStage[, 1]),
      futility_per_look = arr(fut),
      analysis_time = arr(sim$analysisTime[, 1]),
      study_duration = fmt(sim$studyDuration),
      expected_patients = fmt(sim$expectedNumberOfSubjects),
      expected_events = fmt(sim$expectedNumberOfEvents)
    )
    out <- c(out, paste0("    ", obj(a)))
  }
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s; rpact %s",\n', R.version.string, packageVersion("rpact")))
cat('  "survival": [\n')
cat(paste(out, collapse = ",\n"))
cat("\n  ]\n}\n")
