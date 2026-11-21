# Reference values for M2 (spending functions and group-sequential boundaries). Sourced by
# tests/r/reference_values.R after fixed_reference_values.R, whose helpers it reuses; this
# file only defines `gsd_rows`.
#
# Primary values (written to the JSON and asserted by tests/test_boundaries.py):
#   spending     gsDesign::sfLDOF, gsDesign::sfLDPocock
#   efficacy     rpact::getDesignGroupSequential (typeOfDesign "asOF" / "asP", sided = 1)
#   inflation    rpact::getDesignCharacteristics()$inflationFactor
#   futility     gsDesign::gsDesign(test.type = 4): non-binding beta spending
#   max events   gsDesign::gsDesign(n.fix = <fixed events>)$n.I
#   expected n   gsDesign::gsDesign(n.fix = 1)$en under theta = 0 and the design alternative,
#                i.e. expected sample size relative to the fixed design (M3); for test.type = 4
#                gsDesign follows the futility bound when computing it
# Cross-checks: gsDesign for efficacy bounds and inflation factors (strict), rpact for
# non-binding futility bounds and expected sample sizes (warns only, in case of a convention
# difference between the two).

sf_gsd <- list(obf = sfLDOF, pocock = sfLDPocock)
as_rpact <- c(obf = "asOF", pocock = "asP")
bs_rpact <- c(obf = "bsOF", pocock = "bsP")
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
maxdiff <- function(a, b) if (anyNA(b)) NA_real_ else max(abs(a - b))
# Expected sample size relative to the fixed design, at theta = 0 and theta = delta. gsDesign
# has returned `en` both as a plain vector and as a data frame with an `en` column.
en_gsd <- function(g) {
  en <- g$en
  if (is.list(en)) en <- en$en
  as.numeric(en)
}
# rpact's average sample numbers under H0 and H1, relative to the fixed design.
asn_rpact <- function(d) {
  ch <- rp(getDesignCharacteristics(d))
  if (!(is.list(ch) || isS4(ch))) return(c(NA_real_, NA_real_))
  c(rp(as.numeric(ch$averageSampleNumber0)), rp(as.numeric(ch$averageSampleNumber1)))
}

gsd_rows <- character(0)
add <- function(...) gsd_rows <<- c(gsd_rows, row(...))

# ------------------------------------------------------------------ spending functions
t_sp <- c(0.1, 0.25, 0.5, 0.75, 1)
for (s in names(sf_gsd)) {
  add(id = q(paste0("spending_", s)), source = q(paste0("gsDesign::", c(obf = "sfLDOF",
      pocock = "sfLDPocock")[[s]])), spending = q(s), total = 0.025, info_rates = arr(t_sp),
      cumulative = arr(sf_gsd[[s]](alpha = 0.025, t = t_sp, param = 1)$spend))
}

# --------------------------------------------------------------- efficacy boundaries
eff_cases <- list(
  list(id = "obf_k3", s = "obf", t = c(1, 2, 3) / 3),
  list(id = "pocock_k3", s = "pocock", t = c(1, 2, 3) / 3),
  list(id = "obf_k2_t05", s = "obf", t = c(0.5, 1)),
  list(id = "pocock_k2_t05", s = "pocock", t = c(0.5, 1)),
  list(id = "obf_t30_70", s = "obf", t = c(0.3, 0.7, 1)),
  list(id = "pocock_t30_70", s = "pocock", t = c(0.3, 0.7, 1)),
  list(id = "obf_k5", s = "obf", t = (1:5) / 5),
  list(id = "pocock_k4_uneven", s = "pocock", t = c(0.2, 0.45, 0.8, 1))
)
for (e in eff_cases) {
  k <- length(e$t)
  d <- getDesignGroupSequential(kMax = k, alpha = 0.025, beta = 0.2, sided = 1,
                                typeOfDesign = as_rpact[[e$s]], informationRates = e$t)
  ifac <- getDesignCharacteristics(d)$inflationFactor
  g <- gsDesign(k = k, test.type = 1, alpha = 0.025, beta = 0.2, timing = e$t,
                sfu = sf_gsd[[e$s]], sfupar = 1, n.fix = 1)
  crosscheck(paste("efficacy", e$id, "max|diff| (gsDesign)"), 0, maxdiff(d$criticalValues,
             g$upper$bound), tol = 1e-4)
  crosscheck(paste("inflation", e$id, "(gsDesign)"), ifac, g$n.I[k], tol = 1e-4)
  en <- en_gsd(g)
  asn <- asn_rpact(d)
  crosscheck(paste("expected n H0", e$id, "(rpact)"), en[1], asn[1], tol = 1e-3,
             strict = FALSE)
  crosscheck(paste("expected n H1", e$id, "(rpact)"), en[2], asn[2], tol = 1e-3,
             strict = FALSE)
  add(id = q(e$id), source = q("rpact::getDesignGroupSequential"), spending = q(e$s),
      alpha_one_sided = 0.025, power = 0.8, info_rates = arr(e$t),
      efficacy = arr(d$criticalValues), inflation_factor = fmt(ifac),
      expected_n_h0 = fmt(en[1]), expected_n_h1 = fmt(en[2]))
}

# ------------------------------------------------------ non-binding futility (beta spending)
fut_cases <- list(
  list(id = "obf_obf_k3", s = "obf", f = "obf", t = c(1, 2, 3) / 3),
  list(id = "obf_pocock_k3", s = "obf", f = "pocock", t = c(1, 2, 3) / 3),
  list(id = "pocock_obf_k3", s = "pocock", f = "obf", t = c(1, 2, 3) / 3),
  list(id = "obf_obf_t30_70_p90", s = "obf", f = "obf", t = c(0.3, 0.7, 1), beta = 0.1)
)
for (e in fut_cases) {
  k <- length(e$t)
  beta <- if (is.null(e$beta)) 0.2 else e$beta
  g <- gsDesign(k = k, test.type = 4, alpha = 0.025, beta = beta, timing = e$t,
                sfu = sf_gsd[[e$s]], sfupar = 1, sfl = sf_gsd[[e$f]], sflpar = 1, n.fix = 1)
  r <- rp(getDesignGroupSequential(kMax = k, alpha = 0.025, beta = beta, sided = 1,
                                   typeOfDesign = as_rpact[[e$s]],
                                   typeBetaSpending = bs_rpact[[e$f]],
                                   bindingFutility = FALSE, informationRates = e$t))
  r_ok <- is.list(r) || isS4(r)
  r_fut <- if (r_ok) rp(r$futilityBounds) else NA_real_
  crosscheck(paste("futility", e$id, "max|diff| (rpact)"), 0,
             maxdiff(g$lower$bound[-k], r_fut), tol = 1e-3, strict = FALSE)
  en <- en_gsd(g)
  asn <- if (r_ok) asn_rpact(r) else c(NA_real_, NA_real_)
  crosscheck(paste("expected n H0", e$id, "(rpact)"), en[1], asn[1], tol = 1e-3,
             strict = FALSE)
  crosscheck(paste("expected n H1", e$id, "(rpact)"), en[2], asn[2], tol = 1e-3,
             strict = FALSE)
  add(id = q(paste0("futility_", e$id)), source = q("gsDesign::gsDesign(test.type = 4)"),
      efficacy_spending = q(e$s), futility_spending = q(e$f), alpha_one_sided = 0.025,
      power = 1 - beta, info_rates = arr(e$t), efficacy = arr(g$upper$bound),
      futility = arr(g$lower$bound), inflation_factor = fmt(g$n.I[k]),
      expected_n_h0 = fmt(en[1]), expected_n_h1 = fmt(en[2]))
}

# ------------------------------------------- maximum events = fixed events x inflation
ev_fix <- nEvents(hr = 0.70, alpha = 0.025, beta = 0.2, ratio = 1, sided = 1)
for (s in names(sf_gsd)) {
  g <- gsDesign(k = 3, test.type = 1, alpha = 0.025, beta = 0.2, sfu = sf_gsd[[s]],
                sfupar = 1, n.fix = ev_fix)
  add(id = q(paste0("max_events_hr0.7_", s, "_k3")), source = q("gsDesign::gsDesign(n.fix)"),
      spending = q(s), hazard_ratio = 0.7, alpha_one_sided = 0.025, power = 0.8,
      info_rates = arr(c(1, 2, 3) / 3), events_fixed = fmt(ev_fix),
      events_max = fmt(g$n.I[3]))
}
