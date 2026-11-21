# Writes tests/bayes_reference_values.json (M4). Base R only, so it runs anywhere R does.
# Run from the repository root:
#
#   Rscript tests/r/bayes_reference_values.R > bayes.json && mv bayes.json tests/bayes_reference_values.json
#
# This is an independent implementation of the M4 calculations, so the Python tests compare
# two codebases, not one against itself:
#   posterior probability  stats::integrate of f_C(p) * S_T(p) (adaptive quadrature; Python
#                          uses fixed Gauss-Legendre nodes on the quantile scale)
#   final success table    integrate() at every final (x_C, x_T), no monotonicity shortcut
#   beta-binomial pmf      lchoose + lbeta written out here (Python uses scipy.stats.betabinom)
#   PPoS, OCs, assurance   explicit sums over interim and future outcomes
# The script stops if a posterior probability disagrees with the exact closed form for
# integer parameters (Cook 2003; see test_bayes.py) by more than 1e-8.

options(digits = 15)
fmt <- function(x) formatC(x, digits = 12, format = "g")
q <- function(s) sprintf('"%s"', s)
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
row <- function(...) {
  a <- list(...)
  paste0("    {", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
}

# P(p_T > p_C) for independent Beta(ac, bc) and Beta(at, bt).
prob_sup <- function(ac, bc, at, bt) {
  f <- function(p) dbeta(p, ac, bc) * pbeta(p, at, bt, lower.tail = FALSE)
  integrate(f, 0, 1, rel.tol = 1e-12, abs.tol = 0, subdivisions = 1000L)$value
}
# Exact value for integer at: sum_{i=0}^{at-1} B(ac+i, bc+bt) / ((bt+i) B(1+i, bt) B(ac, bc)).
prob_sup_exact <- function(ac, bc, at, bt) {
  i <- 0:(at - 1)
  sum(exp(lbeta(ac + i, bc + bt) - log(bt + i) - lbeta(1 + i, bt) - lbeta(ac, bc)))
}
dbetabinom <- function(y, m, a, b) exp(lchoose(m, y) + lbeta(y + a, m - y + b) - lbeta(a, b))

success_table <- function(nc, nt, prior_c, prior_t, eta) {
  s <- matrix(FALSE, nc + 1, nt + 1)
  for (xc in 0:nc) for (xt in 0:nt) {
    s[xc + 1, xt + 1] <- prob_sup(prior_c[1] + xc, prior_c[2] + nc - xc,
                                  prior_t[1] + xt, prior_t[2] + nt - xt) >= eta
  }
  s
}
# P(final success | interim x) with the future responders weighted by wc(y), wt(y).
win_prob <- function(s, xc, xt, wc, wt) {
  sum(outer(wc, wt) * s[xc + seq_along(wc), xt + seq_along(wt)])
}
ppos <- function(s, xc, n1c, nc, xt, n1t, nt, prior_c, prior_t) {
  wc <- dbetabinom(0:(nc - n1c), nc - n1c, prior_c[1] + xc, prior_c[2] + n1c - xc)
  wt <- dbetabinom(0:(nt - n1t), nt - n1t, prior_t[1] + xt, prior_t[2] + n1t - xt)
  win_prob(s, xc, xt, wc, wt)
}

# Exact OCs at fixed rates (numeric law) or averaged over Beta design priors (law = c(a, b)).
ocs <- function(s, pp, n1c, nc, n1t, nt, law_c, law_t, go, nogo) {
  interim <- function(n, law) if (length(law) == 2) dbetabinom(0:n, n, law[1], law[2]) else dbinom(0:n, n, law)
  future <- function(x, n1, n, law) {
    m <- n - n1
    if (length(law) == 2) dbetabinom(0:m, m, law[1] + x, law[2] + n1 - x) else dbinom(0:m, m, law)
  }
  pc <- interim(n1c, law_c)
  pt <- interim(n1t, law_t)
  res <- c(go = 0, nogo = 0, cont = 0, final = 0)
  for (xc in 0:n1c) for (xt in 0:n1t) {
    w <- pc[xc + 1] * pt[xt + 1]
    v <- pp[xc + 1, xt + 1]
    if (v >= go) {
      res["go"] <- res["go"] + w
    } else if (v <= nogo) {
      res["nogo"] <- res["nogo"] + w
    } else {
      res["cont"] <- res["cont"] + w
      res["final"] <- res["final"] + w * win_prob(s, xc, xt, future(xc, n1c, nc, law_c),
                                                   future(xt, n1t, nt, law_t))
    }
  }
  res
}

rows <- character(0)
add <- function(...) rows <<- c(rows, row(...))

# ------------------------------------------------------------- posterior probability
post_cases <- list(
  list(id = "post_12of30_vs_6of30", xc = 6, nc = 30, xt = 12, nt = 30, pc = c(1, 1), pt = c(1, 1)),
  list(id = "post_45of100_vs_30of100", xc = 30, nc = 100, xt = 45, nt = 100, pc = c(1, 1), pt = c(1, 1)),
  list(id = "post_jeffreys_2of10_vs_7of12", xc = 2, nc = 10, xt = 7, nt = 12, pc = c(0.5, 0.5), pt = c(0.5, 0.5)),
  list(id = "post_informative_unequal", xc = 18, nc = 40, xt = 50, nt = 80, pc = c(6, 14), pt = c(2, 2)),
  list(id = "post_treatment_worse", xc = 15, nc = 25, xt = 8, nt = 25, pc = c(1, 1), pt = c(1, 1))
)
for (e in post_cases) {
  ac <- e$pc[1] + e$xc; bc <- e$pc[2] + e$nc - e$xc
  at <- e$pt[1] + e$xt; bt <- e$pt[2] + e$nt - e$xt
  p <- prob_sup(ac, bc, at, bt)
  if (at == round(at)) {
    ex <- prob_sup_exact(ac, bc, at, bt)
    message(sprintf("%-32s integrate %.12f  exact %.12f", e$id, p, ex))
    if (abs(p - ex) > 1e-8) stop("closed-form cross-check failed for ", e$id)
  }
  add(id = q(e$id), source = q("stats::integrate"), x_c = e$xc, n_c = e$nc, x_t = e$xt,
      n_t = e$nt, prior_c = arr(e$pc), prior_t = arr(e$pt), prob = fmt(p))
}

# --------------------------------------------- PPoS, operating characteristics, assurance
design_cases <- list(
  list(id = "bayes_30_60", n1c = 30, nc = 60, n1t = 30, nt = 60, pc = c(1, 1), pt = c(1, 1),
       eta = 0.95, go = 0.9, nogo = 0.1,
       interim = list(c(6, 12), c(9, 9), c(10, 15), c(5, 14)),
       truth = list(c(0.3, 0.3), c(0.3, 0.45), c(0.2, 0.5)),
       priors = list(list(c(30, 70), c(45, 55)), list(0.3, c(4.5, 5.5)))),
  list(id = "bayes_jeffreys_2to1", n1c = 12, nc = 30, n1t = 24, nt = 60, pc = c(0.5, 0.5),
       pt = c(0.5, 0.5), eta = 0.9, go = 0.85, nogo = 0.05,
       interim = list(c(3, 10), c(4, 6), c(2, 14)),
       truth = list(c(0.25, 0.25), c(0.25, 0.45)),
       priors = list(list(c(5, 15), c(9, 11))))
)
for (d in design_cases) {
  s <- success_table(d$nc, d$nt, d$pc, d$pt, d$eta)
  pp <- matrix(0, d$n1c + 1, d$n1t + 1)
  for (xc in 0:d$n1c) for (xt in 0:d$n1t) {
    pp[xc + 1, xt + 1] <- ppos(s, xc, d$n1c, d$nc, xt, d$n1t, d$nt, d$pc, d$pt)
  }
  base <- list(design = q(d$id), source = q("base R enumeration"), n_interim_control = d$n1c,
               n_final_control = d$nc, n_interim_treatment = d$n1t, n_final_treatment = d$nt,
               prior_c = arr(d$pc), prior_t = arr(d$pt), success_threshold = d$eta,
               go_threshold = d$go, nogo_threshold = d$nogo)
  for (x in d$interim) {
    rows <- c(rows, do.call(row, c(list(id = q(sprintf("%s_ppos_%d_%d", d$id, x[1], x[2]))),
      base, list(kind = q("ppos"), x_c = x[1], x_t = x[2], ppos = fmt(pp[x[1] + 1, x[2] + 1])))))
  }
  for (tr in d$truth) {
    r <- ocs(s, pp, d$n1c, d$nc, d$n1t, d$nt, tr[1], tr[2], d$go, d$nogo)
    rows <- c(rows, do.call(row, c(list(id = q(sprintf("%s_oc_%g_%g", d$id, tr[1], tr[2]))),
      base, list(kind = q("oc"), p_control = tr[1], p_treatment = tr[2],
      prob_go = fmt(r["go"]), prob_nogo = fmt(r["nogo"]), prob_continue = fmt(r["cont"]),
      prob_final_success = fmt(r["final"])))))
  }
  for (i in seq_along(d$priors)) {
    pr <- d$priors[[i]]
    r <- ocs(s, pp, d$n1c, d$nc, d$n1t, d$nt, pr[[1]], pr[[2]], d$go, d$nogo)
    rows <- c(rows, do.call(row, c(list(id = q(sprintf("%s_assurance_%d", d$id, i))),
      base, list(kind = q("assurance"), design_prior_control = arr(pr[[1]]),
      design_prior_treatment = arr(pr[[2]]), prob_go = fmt(r["go"]),
      prob_nogo = fmt(r["nogo"]), prob_continue = fmt(r["cont"]),
      prob_final_success = fmt(r["final"])))))
  }
}

cat(c("{", sprintf('  "_generated_with": "%s (base R only)",', R.version.string),
      '  "bayes": [', paste(rows, collapse = ",\n"), "  ]", "}"), sep = "\n")
