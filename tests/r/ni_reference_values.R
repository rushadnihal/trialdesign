# Writes tests/ni_reference_values.json (M7, non-inferiority). Run from the repository root:
#
#   Rscript tests/r/ni_reference_values.R > ni.json && mv ni.json tests/ni_reference_values.json
#
# Primary values come from rpact (thetaH0 is the effect under H0, as delta0 / hazard_ratio0
# in trialdesign):
#   normal    rpact::getSampleSizeMeans(normalApproximation = TRUE)
#   binary    rpact::getSampleSizeRates   (Farrington-Manning restricted MLE under H0)
#   survival  rpact::getSampleSizeSurvival (Schoenfeld events)
# rpact's group 1 is treatment and group 2 is control; allocationRatioPlanned = n_T / n_C.
# If gsDesign is installed, nBinomial(delta0) and nEvents(hr0) are printed to stderr as a
# cross-check and the script stops if they disagree by more than 1e-6 (relative).

suppressMessages(library(rpact))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 12, format = "g")
q <- function(s) sprintf('"%s"', s)
row <- function(...) {
  a <- list(...)
  paste0("    {", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
}
has_gs <- requireNamespace("gsDesign", quietly = TRUE)
crosscheck <- function(label, primary, other) {
  message(sprintf("%-28s rpact %.8f  gsDesign %.8f", label, primary, other))
  if (abs(primary - other) > 1e-6 * primary) stop("gsDesign cross-check failed for ", label)
}

rows <- character()

# ---------------------------------------------------------------- fixed designs
means <- list(
  list(id = "ni_normal_d0_m3_sd10_p90", delta = 0, sd = 10, delta0 = -3, power = 0.9, alloc = 1),
  list(id = "ni_normal_d1_m2_sd5_2to1", delta = 1, sd = 5, delta0 = -2, power = 0.8, alloc = 2)
)
for (m in means) {
  x <- getSampleSizeMeans(groups = 2, alternative = m$delta, stDev = m$sd, thetaH0 = m$delta0,
    alpha = 0.025, beta = 1 - m$power, sided = 1, normalApproximation = TRUE,
    allocationRatioPlanned = m$alloc)
  rows <- c(rows, row(id = q(m$id), kind = q("fixed"), endpoint = q("normal"),
    source = q("rpact::getSampleSizeMeans"), delta = fmt(m$delta), sd = fmt(m$sd),
    delta0 = fmt(m$delta0), alpha_one_sided = 0.025, power = fmt(m$power),
    allocation = fmt(m$alloc), n_control = fmt(x$nFixed2)))
}

rates <- list(
  list(id = "ni_binary_85_85_m10", pc = 0.85, pt = 0.85, delta0 = -0.10, power = 0.8, alloc = 1),
  list(id = "ni_binary_80_85_m10", pc = 0.80, pt = 0.85, delta0 = -0.10, power = 0.8, alloc = 1),
  list(id = "ni_binary_85_85_m10_2to1", pc = 0.85, pt = 0.85, delta0 = -0.10, power = 0.8, alloc = 2),
  list(id = "ni_binary_lower_better_20_20_m10", pc = 0.20, pt = 0.20, delta0 = 0.10, power = 0.9, alloc = 1),
  list(id = "ni_binary_70_65_m15_1to2", pc = 0.70, pt = 0.65, delta0 = -0.15, power = 0.8, alloc = 0.5)
)
for (r in rates) {
  x <- getSampleSizeRates(groups = 2, pi1 = r$pt, pi2 = r$pc, thetaH0 = r$delta0,
    alpha = 0.025, beta = 1 - r$power, sided = 1, allocationRatioPlanned = r$alloc)
  if (has_gs) {
    g <- gsDesign::nBinomial(p1 = r$pt, p2 = r$pc, delta0 = r$delta0, alpha = 0.025,
      beta = 1 - r$power, sided = 1, ratio = 1 / r$alloc, outtype = 2)
    crosscheck(r$id, x$nFixed2, g$n2)
  }
  rows <- c(rows, row(id = q(r$id), kind = q("fixed"), endpoint = q("binary"),
    source = q("rpact::getSampleSizeRates"), p_control = fmt(r$pc), p_treatment = fmt(r$pt),
    delta0 = fmt(r$delta0), alpha_one_sided = 0.025, power = fmt(r$power),
    allocation = fmt(r$alloc), n_control = fmt(x$nFixed2)))
}

surv <- list(
  list(id = "ni_survival_hr1_hr0_1.3", hr = 1.0, hr0 = 1.3, power = 0.8, alloc = 1),
  list(id = "ni_survival_hr0.9_hr0_1.25_2to1", hr = 0.9, hr0 = 1.25, power = 0.9, alloc = 2)
)
for (s in surv) {
  x <- getSampleSizeSurvival(hazardRatio = s$hr, thetaH0 = s$hr0, alpha = 0.025,
    beta = 1 - s$power, sided = 1, allocationRatioPlanned = s$alloc,
    lambda2 = log(2) / 12, accrualTime = c(0, 24), followUpTime = 12)
  ev <- as.numeric(x$eventsFixed[1])
  if (has_gs) {
    g <- gsDesign::nEvents(hr = s$hr, hr0 = s$hr0, alpha = 0.025, beta = 1 - s$power,
      ratio = s$alloc, sided = 1)
    crosscheck(s$id, ev, g)
  }
  rows <- c(rows, row(id = q(s$id), kind = q("fixed"), endpoint = q("survival"),
    source = q("rpact::getSampleSizeSurvival"), hazard_ratio = fmt(s$hr),
    hazard_ratio0 = fmt(s$hr0), alpha_one_sided = 0.025, power = fmt(s$power),
    allocation = fmt(s$alloc), events = fmt(ev)))
}

# ------------------------------------------------------- group-sequential maxima
# Maximum sample size (or events) = fixed n x inflation factor, no futility.
gsd <- function(K, type) getDesignGroupSequential(kMax = K, alpha = 0.025, beta = 0.2,
  typeOfDesign = type, sided = 1)
d_obf3 <- gsd(3, "asOF")
d_poc2 <- gsd(2, "asP")
x <- getSampleSizeRates(d_obf3, groups = 2, pi1 = 0.85, pi2 = 0.85, thetaH0 = -0.10)
rows <- c(rows, row(id = q("ni_gsd_binary_obf3"), kind = q("gsd"), endpoint = q("binary"),
  source = q("rpact::getSampleSizeRates"), p_control = 0.85, p_treatment = 0.85,
  delta0 = -0.1, alpha_one_sided = 0.025, power = 0.8, allocation = 1, looks = 3,
  efficacy = q("obf"), max_n_total = fmt(x$maxNumberOfSubjects)))
x <- getSampleSizeMeans(d_poc2, groups = 2, alternative = 0, stDev = 10, thetaH0 = -3,
  normalApproximation = TRUE)
rows <- c(rows, row(id = q("ni_gsd_normal_pocock2"), kind = q("gsd"), endpoint = q("normal"),
  source = q("rpact::getSampleSizeMeans"), delta = 0, sd = 10, delta0 = -3,
  alpha_one_sided = 0.025, power = 0.8, allocation = 1, looks = 2, efficacy = q("pocock"),
  max_n_total = fmt(x$maxNumberOfSubjects)))
x <- getSampleSizeSurvival(d_obf3, hazardRatio = 1, thetaH0 = 1.3, lambda2 = log(2) / 12,
  accrualTime = c(0, 24), followUpTime = 12)
rows <- c(rows, row(id = q("ni_gsd_survival_obf3"), kind = q("gsd"), endpoint = q("survival"),
  source = q("rpact::getSampleSizeSurvival"), hazard_ratio = 1, hazard_ratio0 = 1.3,
  alpha_one_sided = 0.025, power = 0.8, allocation = 1, looks = 3, efficacy = q("obf"),
  max_events = fmt(as.numeric(x$maxNumberOfEvents))))

cat("{\n")
cat(sprintf('  "_generated_with": "%s; rpact %s%s",\n', R.version.string,
  packageVersion("rpact"),
  if (has_gs) paste0("; gsDesign ", packageVersion("gsDesign"), " (cross-check)") else ""))
cat('  "ni": [\n')
cat(paste(rows, collapse = ",\n"))
cat("\n  ]\n}\n")
