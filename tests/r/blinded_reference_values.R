# Writes tests/blinded_reference_values.json (M19a, blinded sample size re-estimation).
# Run from the repository root:
#
#   Rscript tests/r/blinded_reference_values.R > blinded.json && mv blinded.json tests/blinded_reference_values.json
#
# No CRAN package for the internal pilot design is on the build machine's R mirror
# (blindrecalc is not), so this script simulates the design of Kieser and Friede (2003)
# patient by patient, independently of the Python code, which draws sufficient statistics:
#
# 1. Draw the pilot (n1C control, n1T treatment) from N(0, sd) and N(delta, sd).
# 2. Lumped variance var(c(pilot_c, pilot_t)); optionally subtract n1C n1T Delta^2/(n1(n1-1)).
# 3. n_C = ceiling((1 + 1/r) s^2 (qnorm(1 - alpha) + qnorm(power))^2 / Delta^2), clipped to
#    [floor, cap]; n_T = ceiling(r n_C).
# 4. Draw the remaining patients and apply the one-sided pooled t-test (stats::t.test with
#    var.equal = TRUE on a subset of runs, as a check of the hand-coded statistic).
#
# "sizes" gives the deterministic re-estimated sizes for a few lumped variances.

options(digits = 15)
fmt <- function(x) formatC(x, digits = 12, format = "g")
q <- function(s) sprintf('"%s"', s)
row <- function(...) {
  a <- list(...)
  paste0("    {", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
}

plan <- function(d) {
  r <- d$alloc
  k <- (1 + 1 / r) * (qnorm(1 - d$alpha) + qnorm(d$power))^2 / d$delta^2
  n_planned <- ceiling(k * d$sd^2 - 1e-9)
  n1c <- round(d$pilot / (1 + r))
  n1t <- d$pilot - n1c
  pilot_floor <- max(n1c, ceiling(n1t / r - 1e-9))
  lo <- if (d$floor == "pilot") pilot_floor else max(pilot_floor, n_planned)
  hi <- floor(d$max_factor * n_planned + 1e-9)
  bias <- n1c * n1t * d$delta^2 / (d$pilot * (d$pilot - 1))
  list(k = k, n_planned = n_planned, n1c = n1c, n1t = n1t, lo = lo, hi = hi, bias = bias, r = r)
}

new_n <- function(p, d, s2) {
  used <- if (d$adjusted) max(s2 - p$bias, 0) else s2
  nc <- min(max(ceiling(p$k * used - 1e-9), p$lo), p$hi)
  c(nc, ceiling(p$r * nc - 1e-9))
}

simulate <- function(d, theta, sigma, nsim) {
  p <- plan(d)
  reject <- logical(nsim)
  total <- numeric(nsim)
  crit_cache <- list()
  for (i in seq_len(nsim)) {
    yc <- rnorm(p$n1c, 0, sigma)
    yt <- rnorm(p$n1t, theta, sigma)
    n <- new_n(p, d, var(c(yc, yt)))
    if (n[1] > p$n1c) yc <- c(yc, rnorm(n[1] - p$n1c, 0, sigma))
    if (n[2] > p$n1t) yt <- c(yt, rnorm(n[2] - p$n1t, theta, sigma))
    df <- n[1] + n[2] - 2
    s2 <- (sum((yc - mean(yc))^2) + sum((yt - mean(yt))^2)) / df
    tstat <- sign(d$delta) * (mean(yt) - mean(yc)) / sqrt(s2 * (1 / n[1] + 1 / n[2]))
    if (i <= 200) {
      tt <- t.test(yt, yc, var.equal = TRUE)$statistic
      stopifnot(abs(sign(d$delta) * tt - tstat) < 1e-9)
    }
    reject[i] <- tstat >= qt(1 - d$alpha, df)
    total[i] <- n[1] + n[2]
  }
  list(rate = mean(reject), mean_n = mean(total), sd_n = sd(total))
}

designs <- list(
  kf = list(delta = 0.5, sd = 1, pilot = 60, alpha = 0.025, power = 0.8, alloc = 1,
    max_factor = 2, adjusted = FALSE, floor = "planned"),
  kf_adjusted = list(delta = 0.5, sd = 1, pilot = 60, alpha = 0.025, power = 0.8, alloc = 1,
    max_factor = 2, adjusted = TRUE, floor = "planned"),
  small_pilot = list(delta = 1, sd = 1, pilot = 10, alpha = 0.025, power = 0.8, alloc = 1,
    max_factor = 3, adjusted = FALSE, floor = "pilot"),
  two_to_one = list(delta = 0.5, sd = 1, pilot = 45, alpha = 0.025, power = 0.9, alloc = 2,
    max_factor = 2, adjusted = FALSE, floor = "planned")
)
cases <- list(
  list("kf", 0, 1), list("kf", 0.5, 1), list("kf", 0, 1.4), list("kf", 0.5, 1.4),
  list("kf_adjusted", 0, 1), list("kf_adjusted", 0.5, 1),
  list("small_pilot", 0, 1), list("small_pilot", 1, 1), list("small_pilot", 0, 1.5),
  list("two_to_one", 0, 1.2), list("two_to_one", 0.5, 1.2)
)

design_args <- function(name) {
  d <- designs[[name]]
  list(design = q(name), delta = fmt(d$delta), sd = fmt(d$sd), pilot_n = d$pilot,
    alpha_one_sided = fmt(d$alpha), power = fmt(d$power), allocation = fmt(d$alloc),
    max_factor = fmt(d$max_factor), adjusted = tolower(d$adjusted), floor = q(d$floor))
}

set.seed(2026)
nsim <- 200000
sims <- character()
for (cs in cases) {
  d <- designs[[cs[[1]]]]
  res <- simulate(d, cs[[2]], cs[[3]], nsim)
  sims <- c(sims, do.call(row, c(design_args(cs[[1]]), list(true_delta = fmt(cs[[2]]),
    true_sd = fmt(cs[[3]]), n_sim = nsim, rejection_rate = fmt(res$rate),
    mean_n_total = fmt(res$mean_n), sd_n_total = fmt(res$sd_n)))))
}

sizes <- character()
for (name in names(designs)) {
  d <- designs[[name]]
  p <- plan(d)
  for (s2 in c(0.3, 0.9, 1.0, 1.3, 2.5, 9)) {
    n <- new_n(p, d, s2)
    sizes <- c(sizes, do.call(row, c(design_args(name), list(variance_lumped = fmt(s2),
      planned_n_control = p$n_planned, n_control = n[1], n_treatment = n[2]))))
  }
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s",\n', R.version.string))
cat('  "simulations": [\n')
cat(paste(sims, collapse = ",\n"))
cat('\n  ],\n  "sizes": [\n')
cat(paste(sizes, collapse = ",\n"))
cat("\n  ]\n}\n")
