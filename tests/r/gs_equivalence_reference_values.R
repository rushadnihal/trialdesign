# Writes tests/gs_equivalence_reference_values.json (M19b, group-sequential equivalence).
# Run from the repository root:
#
#   Rscript tests/r/gs_equivalence_reference_values.R > gseq.json && mv gseq.json tests/gs_equivalence_reference_values.json
#
# Two one-sided group-sequential tests, each with the critical values of a one-sided
# Lan-DeMets design at level alpha from rpact::getDesignGroupSequential. Equivalence is
# declared at the first look k with
#   lo sqrt(n_k) + c_k s_lo < Z_k s1 < hi sqrt(n_k) - c_k s_hi,  Z_k = theta_hat_k sqrt(n_k) / s1,
# so the trial continues while Z_k <= a_k or Z_k >= b_k. P(continue through look k) is the
# sum over the 2^k choices of side of rectangle probabilities of the canonical multivariate
# normal (mvtnorm::pmvnorm, Genz-Bretz with seed 2026), independent of the Python recursive
# integration. The sample size solves power = target with uniroot.
#
# For rates, s_lo and s_hi use the Farrington-Manning restricted rates at each margin, found
# here by maximising the restricted likelihood with optimize() rather than by the closed-form
# cubic the Python code uses.

suppressMessages(library(rpact))
suppressMessages(library(mvtnorm))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 13, format = "g")
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
q <- function(s) sprintf('"%s"', s)
row <- function(...) {
  a <- list(...)
  paste0("    {", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
}

crit <- function(t, alpha, spending) {
  if (length(t) == 1) return(qnorm(1 - alpha))
  type <- c(obf = "asOF", pocock = "asP")[[spending]]
  getDesignGroupSequential(kMax = length(t), alpha = alpha, sided = 1,
    typeOfDesign = type, informationRates = t)$criticalValues
}

fm_rates <- function(pc, pt, r, d) {
  ll <- function(p2) {
    p1 <- p2 + d
    pc * log(p2) + (1 - pc) * log(1 - p2) + r * (pt * log(p1) + (1 - pt) * log(1 - p1))
  }
  lo <- max(0, -d) + 1e-12
  hi <- min(1, 1 - d) - 1e-12
  p2 <- optimize(ll, c(lo, hi), maximum = TRUE, tol = 1e-13)$maximum
  c(p2, p2 + d)
}

sds <- function(cs) {
  r <- cs$alloc
  if (cs$endpoint == "normal") {
    s <- cs$sd * sqrt(1 + 1 / r)
    return(c(s, s, s))
  }
  sd_at <- function(p) sqrt(p[1] * (1 - p[1]) + p[2] * (1 - p[2]) / r)
  c(sd_at(fm_rates(cs$pc, cs$pt, r, cs$lo)), sd_at(fm_rates(cs$pc, cs$pt, r, cs$hi)),
    sd_at(c(cs$pc, cs$pt)))
}

continue_through <- function(n, theta, cs, s, cv) {
  t <- cs$t
  K <- length(t)
  drift <- theta * sqrt(n) / s[3]
  out <- numeric(K)
  for (k in 1:K) {
    tk <- t[1:k]
    R <- outer(tk, tk, function(a, b) sqrt(pmin(a, b) / pmax(a, b)))
    mu <- drift * sqrt(tk)
    a <- (cs$lo * sqrt(tk * n) + cv[1:k] * s[1]) / s[3]
    b <- (cs$hi * sqrt(tk * n) - cv[1:k] * s[2]) / s[3]
    open <- a < b
    sides <- as.matrix(expand.grid(rep(list(1:2), k)))
    total <- 0
    for (i in seq_len(nrow(sides))) {
      if (any(sides[i, !open] == 2)) next  # no stop region: one full-line rectangle
      lower <- ifelse(!open, -Inf, ifelse(sides[i, ] == 1, -Inf, b))
      upper <- ifelse(!open, Inf, ifelse(sides[i, ] == 1, a, Inf))
      total <- total + if (k == 1) {
        pnorm(upper - mu) - pnorm(lower - mu)
      } else {
        pmvnorm(lower = lower, upper = upper, mean = mu, corr = R,
          algorithm = GenzBretz(maxpts = 2e6, abseps = 1e-10, releps = 0))[1]
      }
    }
    out[k] <- total
  }
  out
}

stop_probs <- function(n, theta, cs, s, cv) {
  cont <- continue_through(n, theta, cs, s, cv)
  -diff(c(1, cont))
}

cases <- list(
  list(id = "normal_k1", endpoint = "normal", sd = 1, lo = -0.5, hi = 0.5, theta = 0,
    alloc = 1, alpha = 0.025, power = 0.8, t = 1, spending = "obf"),
  list(id = "normal_k2_obf", endpoint = "normal", sd = 1, lo = -0.5, hi = 0.5, theta = 0,
    alloc = 1, alpha = 0.025, power = 0.8, t = c(0.5, 1), spending = "obf"),
  list(id = "normal_k3_obf", endpoint = "normal", sd = 1, lo = -0.5, hi = 0.5, theta = 0,
    alloc = 1, alpha = 0.025, power = 0.9, t = c(1, 2, 3) / 3, spending = "obf"),
  list(id = "normal_k3_pocock_alpha05", endpoint = "normal", sd = 2, lo = -1, hi = 1,
    theta = 0, alloc = 1, alpha = 0.05, power = 0.8, t = c(1, 2, 3) / 3, spending = "pocock"),
  list(id = "normal_k4_asym_2to1", endpoint = "normal", sd = 1, lo = -0.4, hi = 0.6,
    theta = 0.1, alloc = 2, alpha = 0.025, power = 0.8, t = c(0.3, 0.5, 0.8, 1),
    spending = "obf"),
  list(id = "binary_k3_obf", endpoint = "binary", pc = 0.7, pt = 0.7, lo = -0.1, hi = 0.1,
    alloc = 1, alpha = 0.025, power = 0.8, t = c(1, 2, 3) / 3, spending = "obf"),
  list(id = "binary_k2_pocock_asym", endpoint = "binary", pc = 0.6, pt = 0.65, lo = -0.1,
    hi = 0.2, alloc = 1.5, alpha = 0.05, power = 0.85, t = c(0.6, 1), spending = "pocock")
)

set.seed(2026)
out <- character()
for (cs in cases) {
  if (cs$endpoint == "binary") cs$theta <- cs$pt - cs$pc
  s <- sds(cs)
  cv <- crit(cs$t, cs$alpha, cs$spending)
  power_at <- function(n) sum(stop_probs(n, cs$theta, cs, s, cv))
  n <- uniroot(function(n) power_at(n) - cs$power, c(5, 1e5), tol = 1e-10)$root
  p_design <- stop_probs(n, cs$theta, cs, s, cv)
  p_upper <- stop_probs(n, cs$hi, cs, s, cv)
  p_lower <- stop_probs(n, cs$lo, cs, s, cv)
  e_n <- sum(p_design * cs$t * n) + (1 - sum(p_design)) * n
  inputs <- if (cs$endpoint == "normal") list(sd = fmt(cs$sd), delta = fmt(cs$theta)) else
    list(p_control = fmt(cs$pc), p_treatment = fmt(cs$pt))
  out <- c(out, do.call(row, c(list(id = q(cs$id), endpoint = q(cs$endpoint)), inputs,
    list(margin = arr(c(cs$lo, cs$hi)), allocation = fmt(cs$alloc), alpha = fmt(cs$alpha),
      power = fmt(cs$power), info_rates = arr(cs$t), spending = q(cs$spending),
      critical_values = arr(cv), sds = arr(s), n_control = fmt(n),
      stop_probs_design = arr(p_design), stop_probs_upper_margin = arr(p_upper),
      stop_probs_lower_margin = arr(p_lower), expected_n_control = fmt(e_n)))))
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s; rpact %s; mvtnorm %s",\n', R.version.string,
  packageVersion("rpact"), packageVersion("mvtnorm")))
cat('  "designs": [\n')
cat(paste(out, collapse = ",\n"))
cat("\n  ]\n}\n")
