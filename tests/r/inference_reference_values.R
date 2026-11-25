# Writes tests/inference_reference_values.json (M15, inference at the end of a group-sequential
# trial). Run from the repository root:
#
#   Rscript tests/r/inference_reference_values.R > inf.json && mv inf.json tests/inference_reference_values.json
#
# Each case is a one-sided group-sequential design (getDesignGroupSequential) and a survival
# dataset of cumulative events and log-rank statistics at the looks analysed, with the events
# on the design's information fractions. rpact's group 1 is treatment; with directionUpper =
# FALSE a benefit is a negative log-rank statistic, so the dataset holds -z where trialdesign
# has z (positive favours treatment), and rpact's estimates are hazard ratios. Reported:
# repeated confidence intervals and p-values (getRepeatedConfidenceIntervals,
# getRepeatedPValues) at every look analysed, and the stage-wise ordering p-value,
# median-unbiased estimate and final confidence interval (getFinalPValue,
# getFinalConfidenceInterval), which rpact gives only once a boundary is crossed or at the
# last look.

suppressMessages(library(rpact))
options(digits = 15)
fmt <- function(x) ifelse(is.na(x), "null", formatC(x, digits = 12, format = "g"))
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
str <- function(x) if (is.null(x)) "null" else sprintf('"%s"', x)
quiet <- function(expr) {  # rpact prints [PROGRESS] lines to stdout
  invisible(utils::capture.output(value <- suppressMessages(expr)))
  value
}
obj <- function(a) paste0("{", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")

cases <- list(
  list(id = "obf4_stop2", info = c(0.25, 0.5, 0.75, 1), design = "asOF", efficacy = "obf",
    beta_spending = "none", futility = NULL, binding = FALSE, events = 400, alloc = 1,
    z = c(1.5, 3.1)),
  list(id = "pocock3_final_not_significant", info = c(1, 2, 3) / 3, design = "asP",
    efficacy = "pocock", beta_spending = "none", futility = NULL, binding = FALSE,
    events = 300, alloc = 1, z = c(1.0, 1.8, 2.1)),
  list(id = "hsd3_nonbinding_futility_stop2", info = c(0.3, 0.7, 1), design = "asHSD",
    gamma = -4, efficacy = "hsd:-4", beta_spending = "bsOF", futility = "obf",
    binding = FALSE, events = 400, alloc = 2, z = c(1.2, 2.9)),
  list(id = "kd3_binding_futility_final", info = c(0.3, 0.7, 1), design = "asKD", gamma = 3,
    efficacy = "kd:3", beta_spending = "bsOF", futility = "obf", binding = TRUE,
    events = 400, alloc = 1, z = c(0.9, 1.6, 2.3)),
  list(id = "obf3_stop1", info = c(0.3, 0.7, 1), design = "asOF", efficacy = "obf",
    beta_spending = "none", futility = NULL, binding = FALSE, events = 400, alloc = 1,
    z = c(4.1)),
  list(id = "obf3_interim_continue", info = c(0.3, 0.7, 1), design = "asOF", efficacy = "obf",
    beta_spending = "none", futility = NULL, binding = FALSE, events = 400, alloc = 1,
    z = c(0.8, 2.2))
)

out <- character()
for (cs in cases) {
  args <- list(kMax = length(cs$info), alpha = 0.025, beta = 0.2, sided = 1,
    informationRates = cs$info, typeOfDesign = cs$design, typeBetaSpending = cs$beta_spending)
  if (!is.null(cs$gamma)) args$gammaA <- cs$gamma
  if (cs$beta_spending != "none") args$bindingFutility <- cs$binding
  d <- do.call(getDesignGroupSequential, args)
  k <- length(cs$z)
  ev <- round(cs$info[1:k] * cs$events)
  ds <- getDataset(overallEvents = ev, overallLogRanks = -cs$z,
    overallAllocationRatios = rep(cs$alloc, k))
  sr <- getStageResults(d, ds, directionUpper = FALSE)
  rci <- quiet(getRepeatedConfidenceIntervals(d, ds, directionUpper = FALSE))
  rp <- quiet(getRepeatedPValues(sr))
  fp <- quiet(getFinalPValue(sr))
  fci <- quiet(getFinalConfidenceInterval(d, ds, directionUpper = FALSE))
  final_p <- if (is.null(fp$pFinal)) NA else fp$pFinal
  mu <- if (is.null(fci$medianUnbiased)) NA else fci$medianUnbiased
  ci <- if (is.null(fci$finalConfidenceInterval)) c(NA, NA) else fci$finalConfidenceInterval
  a <- list(
    id = sprintf('"%s"', cs$id),
    info_rates = arr(cs$info),
    efficacy_spending = str(cs$efficacy),
    futility_spending = str(cs$futility),
    binding_futility = if (cs$binding) "true" else "false",
    events = cs$events,
    allocation = fmt(cs$alloc),
    z = arr(cs$z),
    efficacy = arr(d$criticalValues),
    futility = if (cs$beta_spending == "none") "null" else arr(d$futilityBounds),
    rci_lower_hr = arr(rci[1, 1:k]),
    rci_upper_hr = arr(rci[2, 1:k]),
    repeated_p = arr(rp[1:k]),
    final_p = fmt(final_p),
    median_unbiased_hr = fmt(mu),
    ci_lower_hr = fmt(ci[1]),
    ci_upper_hr = fmt(ci[2])
  )
  out <- c(out, paste0("    ", obj(a)))
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s; rpact %s",\n', R.version.string, packageVersion("rpact")))
cat('  "cases": [\n')
cat(paste(out, collapse = ",\n"))
cat("\n  ]\n}\n")
