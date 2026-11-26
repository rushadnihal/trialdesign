# Writes tests/nph_reference_values.json (M17, non-proportional hazards and the FH test).
# Run from the repository root:
#
#   Rscript tests/r/nph_reference_values.R > nph.json && mv nph.json tests/nph_reference_values.json
#
# Three blocks:
#
# 1. "event_probabilities": rpact::getEventProbabilities, the expected share of patients with
#    an observed event by calendar time t, under piecewise-exponential hazards (delayed,
#    waning and crossing effects, with and without dropout) and Weibull event times (rpact
#    allows no dropout with Weibull). Exact, compared to 1e-6.
# 2. "simulation": rpact::getSimulationSurvival with 100,000 trials. Each design is sized
#    under proportional hazards (getSampleSizeSurvival with the design hazard ratio) and
#    then simulated under the scenario's event model, as trialdesign does. The last case
#    gives both arms the same non-exponential hazard, so its rejection rate is a type I error.
#    rpact's group 1 is treatment; directionUpper = FALSE makes z positive when treatment
#    does better (the trialdesign convention). Piecewise hazards start at
#    piecewiseSurvivalTime (first entry 0), the same convention as trialdesign.hazards.
# 3. "fh": the Fleming-Harrington FH(rho, gamma) weighted log-rank z on three simulated
#    datasets, computed here from the pooled Kaplan-Meier estimate just before each event
#    time, and cross-checked against survival::survdiff(rho = ...) for gamma = 0 (the
#    G-rho family). nphRCT is not on the build machine's R mirror, so it is not used.

suppressMessages(library(rpact))
suppressMessages(library(survival))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 15, format = "g")
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
obj <- function(a) paste0("{", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
str_q <- function(x) sprintf('"%s"', x)
n_iter <- 100000L
lam <- function(median) log(2) / median

# ---------------------------------------------------------------- 1. event probabilities

ep_cases <- list(
  list(id = "delayed4", times = c(0, 4), control = rep(lam(12), 2),
    treatment = c(lam(12), 0.7 * lam(12)), kappa = 1, accrual_end = 24, accrual_rate = NULL,
    n = 400, dropout = 0, eval = c(6, 12, 24, 36)),
  list(id = "delayed6_dropout_piecewise_accrual", times = c(0, 6),
    control = rep(lam(9), 2), treatment = c(lam(9), 0.6 * lam(9)), kappa = 1,
    accrual_end = c(6, 20), accrual_rate = c(10, 25), n = NULL, dropout = 0.1,
    eval = c(10, 20, 30)),
  list(id = "waning12", times = c(0, 12), control = rep(lam(10), 2),
    treatment = c(0.65 * lam(10), lam(10)), kappa = 1, accrual_end = 18, accrual_rate = NULL,
    n = 500, dropout = 0.05, eval = c(12, 24, 40)),
  list(id = "crossing_three_pieces", times = c(0, 3, 9), control = c(0.05, 0.08, 0.04),
    treatment = c(0.09, 0.05, 0.02), kappa = 1, accrual_end = 12, accrual_rate = NULL,
    n = 300, dropout = 0, eval = c(6, 15, 30)),
  list(id = "weibull_k1.5", times = 0, control = log(2)^(1 / 1.5) / 12,
    treatment = log(2)^(1 / 1.5) / 12 * 0.7^(1 / 1.5), kappa = 1.5, accrual_end = 24,
    accrual_rate = NULL, n = 400, dropout = 0, eval = c(12, 24, 36)),
  list(id = "weibull_k0.7", times = 0, control = log(2)^(1 / 0.7) / 18,
    treatment = log(2)^(1 / 0.7) / 18 * 0.75^(1 / 0.7), kappa = 0.7, accrual_end = 12,
    accrual_rate = NULL, n = 300, dropout = 0, eval = c(6, 18, 30))
)

ep_out <- character()
for (cs in ep_cases) {
  args <- list(time = cs$eval, kappa = cs$kappa)
  if (cs$kappa == 1) {
    args$piecewiseSurvivalTime <- cs$times
    args$lambda2 <- cs$control
    args$lambda1 <- cs$treatment
  } else {
    args$lambda2 <- cs$control
    args$lambda1 <- cs$treatment
  }
  if (is.null(cs$accrual_rate)) {
    args$accrualTime <- c(0, cs$accrual_end)
    args$maxNumberOfSubjects <- cs$n
    rates <- cs$n / cs$accrual_end
  } else {
    args$accrualTime <- c(0, cs$accrual_end)
    args$accrualIntensity <- cs$accrual_rate
    rates <- cs$accrual_rate
  }
  if (cs$dropout > 0) {
    args$dropoutRate1 <- cs$dropout
    args$dropoutRate2 <- cs$dropout
    args$dropoutTime <- 12
  }
  e <- do.call(getEventProbabilities, args)
  n_total <- sum(rates * diff(c(0, cs$accrual_end)))
  a <- list(
    id = str_q(cs$id), times = arr(cs$times), control = arr(cs$control),
    treatment = arr(cs$treatment), kappa = fmt(cs$kappa), accrual_end = arr(cs$accrual_end),
    accrual_rate = arr(rates), dropout = fmt(cs$dropout), time = arr(cs$eval),
    n_total = fmt(n_total),
    events = arr(e$cumulativeEventProbabilities * n_total)
  )
  ep_out <- c(ep_out, paste0("    ", obj(a)))
}

# ------------------------------------------------------------------------ 2. simulation

sim_cases <- list(
  list(id = "obf2_delayed4", K = 2, beta_spending = "none", hr = 0.7, median = 12,
    model = "delayed", times = c(0, 4), hr_after = 0.7, kappa = 1, dropout = 0,
    accrual_time = 24, followup = 12, seed = 1701),
  list(id = "obf3_futility_waning", K = 3, beta_spending = "bsOF", hr = 0.7, median = 12,
    model = "waning", times = c(0, 12), hr_after = 0.7, kappa = 1, dropout = 0.05,
    accrual_time = 18, followup = 12, seed = 1702),
  list(id = "obf2_weibull", K = 2, beta_spending = "none", hr = 0.7, median = 12,
    model = "weibull", times = 0, hr_after = 0.7, kappa = 1.5, dropout = 0,
    accrual_time = 24, followup = 12, seed = 1703),
  list(id = "pocock2_null_piecewise", K = 2, beta_spending = "none", hr = 0.7, median = 12,
    model = "null_piecewise", times = c(0, 6), hr_after = 1, kappa = 1, dropout = 0,
    accrual_time = 24, followup = 12, seed = 1704)
)

sim_out <- character()
for (cs in sim_cases) {
  type <- if (startsWith(cs$id, "pocock")) "asP" else "asOF"
  dargs <- list(kMax = cs$K, alpha = 0.025, beta = 0.2, sided = 1, typeOfDesign = type,
    typeBetaSpending = cs$beta_spending)
  if (cs$beta_spending != "none") dargs$bindingFutility <- FALSE
  d <- do.call(getDesignGroupSequential, dargs)
  sargs <- list(design = d, hazardRatio = cs$hr, lambda2 = lam(cs$median),
    accrualTime = c(0, cs$accrual_time), followUpTime = cs$followup)
  if (cs$dropout > 0) {
    sargs$dropoutRate1 <- cs$dropout
    sargs$dropoutRate2 <- cs$dropout
    sargs$dropoutTime <- 12
  }
  ss <- do.call(getSampleSizeSurvival, sargs)
  d_max <- ceiling(ss$maxNumberOfEvents)
  planned <- round(d$informationRates * d_max)
  n_total <- ceiling(ss$maxNumberOfSubjects)
  lc <- lam(cs$median)
  simargs <- list(design = d, accrualTime = c(0, cs$accrual_time),
    maxNumberOfSubjects = n_total, plannedEvents = planned, directionUpper = FALSE,
    maxNumberOfIterations = n_iter, seed = cs$seed)
  if (cs$model == "delayed") {
    control <- c(lc, lc); treatment <- c(lc, cs$hr_after * lc)
  } else if (cs$model == "waning") {
    control <- c(lc, lc); treatment <- c(cs$hr_after * lc, lc)
  } else if (cs$model == "null_piecewise") {
    control <- c(1.5 * lc, 0.5 * lc); treatment <- control
  }
  if (cs$model == "weibull") {
    control <- log(2)^(1 / cs$kappa) / cs$median
    treatment <- control * cs$hr_after^(1 / cs$kappa)
    simargs$kappa <- cs$kappa
    simargs$lambda2 <- control
    simargs$lambda1 <- treatment
  } else {
    simargs$piecewiseSurvivalTime <- cs$times
    simargs$lambda2 <- control
    simargs$lambda1 <- treatment
  }
  if (cs$dropout > 0) {
    simargs$dropoutRate1 <- cs$dropout
    simargs$dropoutRate2 <- cs$dropout
    simargs$dropoutTime <- 12
  }
  sim <- do.call(getSimulationSurvival, simargs)
  futility <- if (cs$beta_spending != "none") c(d$futilityBounds, d$criticalValues[cs$K]) else NULL
  fut <- if (is.null(futility)) rep(0, cs$K) else sim$futilityPerStage[, 1]
  fut <- c(fut, rep(0, cs$K - length(fut)))
  a <- list(
    id = str_q(cs$id), model = str_q(cs$model), times = arr(cs$times),
    control = arr(control), treatment = arr(treatment), kappa = fmt(cs$kappa),
    design_hazard_ratio = fmt(cs$hr), median_control = fmt(cs$median),
    dropout = fmt(cs$dropout), accrual_time = fmt(cs$accrual_time),
    followup = fmt(cs$followup), design_events = fmt(ss$maxNumberOfEvents),
    events = arr(planned), n_total = n_total, efficacy = arr(d$criticalValues),
    futility = if (is.null(futility)) "null" else arr(futility), n_sim = n_iter,
    reject_prob = fmt(sim$overallReject), reject_per_look = arr(sim$rejectPerStage[, 1]),
    futility_per_look = arr(fut), analysis_time = arr(sim$analysisTime[, 1]),
    study_duration = fmt(sim$studyDuration),
    expected_patients = fmt(sim$expectedNumberOfSubjects),
    expected_events = fmt(sim$expectedNumberOfEvents)
  )
  sim_out <- c(sim_out, paste0("    ", obj(a)))
}

# ----------------------------------------------------------------------------- 3. FH test

fh_z <- function(time, status, arm, rho, gamma) {
  # positive when treatment (arm 1) has fewer events than expected
  ev_times <- sort(unique(time[status == 1]))
  km <- 1
  num <- 0
  var <- 0
  for (tj in ev_times) {
    at_risk <- time >= tj
    n_j <- sum(at_risk)
    n_tj <- sum(at_risk & arm == 1)
    d_j <- sum(time == tj & status == 1)
    d_tj <- sum(time == tj & status == 1 & arm == 1)
    w <- km^rho * (1 - km)^gamma
    num <- num + w * (d_j * n_tj / n_j - d_tj)
    if (n_j > 1) {
      var <- var + w^2 * d_j * (n_tj / n_j) * (1 - n_tj / n_j) * (n_j - d_j) / (n_j - 1)
    }
    km <- km * (1 - d_j / n_j)
  }
  num / sqrt(var)
}

set.seed(2026)
fh_out <- character()
datasets <- list(
  list(id = "delayed_n120", n = 120, l_c = c(0.08, 0.08), l_t = c(0.08, 0.04), cut = 5, cens = 0.02),
  list(id = "ph_n80", n = 80, l_c = c(0.1, 0.1), l_t = c(0.06, 0.06), cut = 1, cens = 0.03),
  list(id = "crossing_n150", n = 150, l_c = c(0.05, 0.09), l_t = c(0.1, 0.03), cut = 4, cens = 0.01)
)
rpw <- function(n, l, cut) {
  # piecewise exponential with one change point at `cut`
  e <- rexp(n)
  ifelse(e <= l[1] * cut, e / l[1], cut + (e - l[1] * cut) / l[2])
}
for (ds in datasets) {
  arm <- rep(c(0L, 1L), length.out = ds$n)
  t_ev <- ifelse(arm == 1, rpw(ds$n, ds$l_t, ds$cut), rpw(ds$n, ds$l_c, ds$cut))
  t_c <- pmin(rexp(ds$n, ds$cens), 30)
  time <- signif(pmin(t_ev, t_c), 12)
  status <- as.integer(t_ev <= t_c)
  stopifnot(!any(duplicated(time[status == 1])))  # censorings may tie (at 30)
  zs <- list()
  for (rg in list(c(0, 0), c(0, 1), c(1, 0), c(1, 1), c(0.5, 2))) {
    z <- fh_z(time, status, arm, rg[1], rg[2])
    if (rg[2] == 0) {
      sd <- survdiff(Surv(time, status) ~ arm, rho = rg[1])
      z_sd <- (sd$exp[2] - sd$obs[2]) / sqrt(sd$var[2, 2])
      stopifnot(abs(z - z_sd) < 1e-10)
    }
    zs[[sprintf("%g,%g", rg[1], rg[2])]] <- fmt(z)
  }
  a <- list(id = str_q(ds$id), time = arr(time), event = arr(status), treat = arr(arm),
    z = paste0("{", paste(sprintf('"%s": %s', names(zs), unlist(zs)), collapse = ", "), "}"))
  fh_out <- c(fh_out, paste0("    ", obj(a)))
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s; rpact %s; survival %s",\n', R.version.string,
  packageVersion("rpact"), packageVersion("survival")))
cat('  "event_probabilities": [\n')
cat(paste(ep_out, collapse = ",\n"))
cat('\n  ],\n  "simulation": [\n')
cat(paste(sim_out, collapse = ",\n"))
cat('\n  ],\n  "fh": [\n')
cat(paste(fh_out, collapse = ",\n"))
cat("\n  ]\n}\n")
