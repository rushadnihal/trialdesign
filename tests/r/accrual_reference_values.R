# Writes tests/accrual_reference_values.json (M8, survival accrual and dropout).
# Run from the repository root:
#
#   Rscript tests/r/accrual_reference_values.R > acc.json && mv acc.json tests/accrual_reference_values.json
#
# Primary values come from rpact::getSampleSizeSurvival (Lachin-Foulkes accrual with
# exponential events and dropout; Schoenfeld events). Time is in months, lambda2 =
# log(2) / median_control, and dropout is the probability of dropping out by dropoutTime.
# rpact's group 1 is treatment; allocationRatioPlanned = n_T / n_C.

suppressMessages(library(rpact))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 12, format = "g")
q <- function(s) sprintf('"%s"', s)
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
row <- function(...) {
  a <- list(...)
  paste0("    {", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
}
rows <- character()

# Each case: id, hr, hr0, median, alloc, dropout, and accrual given as rpact expects.
# mode A: accrualTime = c(0, R), followUpTime          -> rate (uniform)
# mode B: accrualTime = c(0, breaks), intensity, followUpTime -> end of accrual
# mode C: accrualTime = c(0, ends), intensity          -> follow-up
cases <- list(
  list(id = "acc_A_uniform_dropout", mode = "A", hr = 0.7, hr0 = 1, median = 12, alloc = 1,
    dropout = 0.05, time = 24, rate = NULL, followup = 12),
  list(id = "acc_A_ni_2to1", mode = "A", hr = 1.0, hr0 = 1.3, median = 18, alloc = 2,
    dropout = 0, time = 18, rate = NULL, followup = 12),
  list(id = "acc_B_rate20", mode = "B", hr = 0.7, hr0 = 1, median = 12, alloc = 1,
    dropout = 0, time = numeric(0), rate = 20, followup = 12),
  list(id = "acc_B_piecewise_2to1", mode = "B", hr = 0.7, hr0 = 1, median = 12, alloc = 2,
    dropout = 0.1, time = 6, rate = c(10, 25), followup = 12),
  list(id = "acc_C_rate20_24m", mode = "C", hr = 0.7, hr0 = 1, median = 12, alloc = 1,
    dropout = 0, time = 24, rate = 20, followup = NULL),
  list(id = "acc_C_piecewise", mode = "C", hr = 0.75, hr0 = 1, median = 9, alloc = 1,
    dropout = 0.05, time = c(6, 30), rate = c(8, 20), followup = NULL)
)

args_for <- function(cs) {
  a <- list(hazardRatio = cs$hr, thetaH0 = cs$hr0, lambda2 = log(2) / cs$median,
    allocationRatioPlanned = cs$alloc, accrualTime = c(0, cs$time))
  if (!is.null(cs$rate)) a$accrualIntensity <- cs$rate
  if (!is.null(cs$followup)) a$followUpTime <- cs$followup
  if (cs$dropout > 0) {
    a$dropoutRate1 <- cs$dropout
    a$dropoutRate2 <- cs$dropout
    a$dropoutTime <- 12
  }
  a
}
inputs <- function(cs) {
  c(list(hazard_ratio = fmt(cs$hr), hazard_ratio0 = fmt(cs$hr0),
    median_control = fmt(cs$median), allocation = fmt(cs$alloc), dropout = fmt(cs$dropout)),
    if (length(cs$time)) list(accrual_time = arr(cs$time)) else list(),
    if (!is.null(cs$rate)) list(accrual_rate = arr(cs$rate)) else list(),
    if (!is.null(cs$followup)) list(followup = fmt(cs$followup)) else list())
}

for (cs in cases) {
  x <- do.call(getSampleSizeSurvival,
    c(args_for(cs), list(alpha = 0.025, beta = 0.2, sided = 1)))
  times <- x$accrualTime
  rows <- c(rows, do.call(row, c(list(id = q(cs$id), kind = q("fixed"),
    source = q("rpact::getSampleSizeSurvival")), inputs(cs), list(
    alpha_one_sided = 0.025, power = 0.8, events = fmt(as.numeric(x$eventsFixed[1])),
    n_total = fmt(as.numeric(x$nFixed[1])), accrual_end = fmt(times[length(times)]),
    followup_solved = fmt(as.numeric(x$followUpTime)),
    study_duration = fmt(as.numeric(x$studyDuration[1]))))))
}

# Group-sequential: maximum events and patients, calendar time and patients at each look.
gs_cases <- list(
  list(case = cases[[1]], id = "acc_gsd_obf3_uniform", K = 3, type = "asOF", eff = "obf"),
  list(case = cases[[4]], id = "acc_gsd_pocock2_piecewise", K = 2, type = "asP", eff = "pocock")
)
for (g in gs_cases) {
  d <- getDesignGroupSequential(kMax = g$K, alpha = 0.025, beta = 0.2,
    typeOfDesign = g$type, sided = 1)
  x <- do.call(getSampleSizeSurvival, c(list(design = d), args_for(g$case)))
  rows <- c(rows, do.call(row, c(list(id = q(g$id), kind = q("gsd"),
    source = q("rpact::getSampleSizeSurvival")), inputs(g$case), list(
    alpha_one_sided = 0.025, power = 0.8, looks = g$K, efficacy = q(g$eff),
    max_events = fmt(as.numeric(x$maxNumberOfEvents)),
    max_n_total = fmt(as.numeric(x$maxNumberOfSubjects)),
    look_time = arr(as.numeric(x$analysisTime[, 1])),
    look_patients = arr(as.numeric(x$numberOfSubjects[, 1]))))))
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s; rpact %s",\n', R.version.string, packageVersion("rpact")))
cat('  "accrual": [\n')
cat(paste(rows, collapse = ",\n"))
cat("\n  ]\n}\n")
