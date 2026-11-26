# Writes tests/multiplicity_reference_values.json (M18, graphical multiple testing).
# Run from the repository root:
#
#   Rscript tests/r/multiplicity_reference_values.R > mult.json && mv mult.json tests/multiplicity_reference_values.json
#
# graphicalMCP and gMCP are not on the build machine's R mirror, so this script implements
# the graphical procedure of Bretz, Maurer, Brannath and Posch (2009) independently of the
# Python code, in the closed-test form rather than the sequential one:
#
# 1. "tests": for six graphs and five p-value vectors each, the rejections and adjusted
#    p-values. A hypothesis is rejected when every intersection hypothesis H_I containing it
#    is rejected by its weighted Bonferroni test (weights from Algorithm 1, by removing the
#    hypotheses outside I); its adjusted p-value is the largest over those I of
#    min_{j in I} p_j / w_j(I) (capped at 1). For Holm graphs both are cross-checked here
#    against stats::p.adjust(method = "holm").
# 2. "power": exact power by integrating the multivariate normal (mvtnorm::pmvnorm) over the
#    cells cut by every critical value z = qnorm(1 - w_j(I) alpha). Inside a cell every
#    weighted Bonferroni test has the same outcome, so the closed test does too; the cell's
#    decision is taken at an interior point and its probability added to the rates it
#    counts towards. The global-null case gives the family-wise error rate.

suppressMessages(library(mvtnorm))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 15, format = "g")
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
mat <- function(m) paste0("[", paste(apply(m, 1, arr), collapse = ", "), "]")
obj <- function(a) paste0("{", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
str_q <- function(x) sprintf('"%s"', x)

remove_one <- function(w, G, j) {
  m <- length(w)
  w2 <- w
  G2 <- G
  for (l in seq_len(m)) {
    if (l == j) next
    w2[l] <- w[l] + w[j] * G[j, l]
    for (k in seq_len(m)) {
      if (k == j || k == l) next
      d <- 1 - G[l, j] * G[j, l]
      G2[l, k] <- if (d > 1e-12) (G[l, k] + G[l, j] * G[j, k]) / d else 0
    }
  }
  w2[j] <- 0
  G2[j, ] <- 0
  G2[, j] <- 0
  diag(G2) <- 0
  list(w = w2, G = G2)
}

intersection_weights <- function(w, G, I) {
  for (j in setdiff(seq_along(w), I)) {
    r <- remove_one(w, G, j)
    w <- r$w
    G <- r$G
  }
  w
}

all_subsets <- function(m) {
  out <- list()
  for (b in 1:(2^m - 1)) out[[b]] <- which(bitwAnd(b, 2^(0:(m - 1))) > 0)
  out
}

closed_test <- function(p, w, G, alpha, subsets, weights) {
  m <- length(p)
  adj <- rep(0, m)
  for (s in seq_along(subsets)) {
    I <- subsets[[s]]
    wi <- weights[[s]][I]
    q <- min(ifelse(wi > 0, p[I] / wi, Inf))
    q <- min(q, 1)
    for (j in I) adj[j] <- max(adj[j], q)
  }
  list(adj = adj, rejected = adj <= alpha)
}

graphs <- list(
  bonferroni3 = list(w = rep(1 / 3, 3), G = matrix(0, 3, 3)),
  holm3 = list(w = rep(1 / 3, 3), G = matrix(0.5, 3, 3) - diag(0.5, 3)),
  holm4 = list(w = rep(1 / 4, 4), G = matrix(1 / 3, 4, 4) - diag(1 / 3, 4)),
  fixed_sequence3 = list(w = c(1, 0, 0), G = rbind(c(0, 1, 0), c(0, 0, 1), c(0, 0, 0))),
  fallback3 = list(w = rep(1 / 3, 3), G = rbind(c(0, 1, 0), c(0, 0, 1), c(0, 0, 0))),
  two_doses = list(w = c(0.5, 0.5, 0, 0), G = rbind(c(0, 0.5, 0.5, 0), c(0.5, 0, 0, 0.5),
    c(0, 1, 0, 0), c(1, 0, 0, 0))),
  general4 = list(w = c(0.6, 0.4, 0, 0), G = rbind(c(0, 0.3, 0.7, 0), c(0.2, 0, 0, 0.8),
    c(0.5, 0.5, 0, 0), c(0.9, 0, 0.1, 0)))
)

set.seed(2026)
test_out <- character()
alpha <- 0.025
for (name in names(graphs)) {
  gr <- graphs[[name]]
  m <- length(gr$w)
  subsets <- all_subsets(m)
  weights <- lapply(subsets, function(I) intersection_weights(gr$w, gr$G, I))
  for (case in 1:5) {
    p <- signif(runif(m)^3 * 0.1, 10)  # mostly small, some above alpha
    ct <- closed_test(p, gr$w, gr$G, alpha, subsets, weights)
    if (startsWith(name, "holm")) stopifnot(max(abs(ct$adj - p.adjust(p, "holm"))) < 1e-12)
    a <- list(id = str_q(sprintf("%s_%d", name, case)), graph = str_q(name),
      weights = arr(gr$w), transitions = mat(gr$G), alpha_one_sided = fmt(alpha),
      p = arr(p), adjusted_p = arr(ct$adj),
      rejected = paste0("[", paste(tolower(ct$rejected), collapse = ", "), "]"))
    test_out <- c(test_out, paste0("    ", obj(a)))
  }
}

exact_power <- function(gr, mu, R, alpha) {
  m <- length(gr$w)
  subsets <- all_subsets(m)
  weights <- lapply(subsets, function(I) intersection_weights(gr$w, gr$G, I))
  cuts <- lapply(1:m, function(j) {
    wj <- unique(unlist(lapply(weights, function(w) w[j])))
    sort(unique(qnorm(1 - wj[wj > 0] * alpha)))
  })
  edges <- lapply(cuts, function(c) c(-Inf, c, Inf))
  grid <- expand.grid(lapply(edges, function(e) seq_len(length(e) - 1)))
  local <- rep(0, m)
  any_p <- 0
  all_p <- 0
  expected <- 0
  for (r in seq_len(nrow(grid))) {
    lo <- sapply(1:m, function(j) edges[[j]][grid[r, j]])
    hi <- sapply(1:m, function(j) edges[[j]][grid[r, j] + 1])
    mid <- ifelse(is.finite(lo) & is.finite(hi), (lo + hi) / 2,
      ifelse(is.finite(lo), lo + 1, hi - 1))
    rej <- closed_test(pnorm(mid, lower.tail = FALSE), gr$w, gr$G, alpha, subsets, weights)$rejected
    if (!any(rej)) next
    pr <- pmvnorm(lower = lo, upper = hi, mean = mu, corr = R,
      algorithm = GenzBretz(maxpts = 2e6, abseps = 1e-9, releps = 0))[1]
    local <- local + pr * rej
    any_p <- any_p + pr
    if (all(rej)) all_p <- all_p + pr
    expected <- expected + pr * sum(rej)
  }
  list(local = local, any = any_p, all = all_p, expected = expected)
}

r4 <- rbind(c(1, 0.5, 0.5, 0.25), c(0.5, 1, 0.25, 0.5), c(0.5, 0.25, 1, 0.5),
  c(0.25, 0.5, 0.5, 1))
power_cases <- list(
  list(id = "holm2_rho0.5", graph = list(w = c(0.5, 0.5), G = rbind(c(0, 1), c(1, 0))),
    mu = c(2.8, 2.0), R = rbind(c(1, 0.5), c(0.5, 1))),
  list(id = "fallback3_independent", graph = graphs$fallback3, mu = c(2.5, 2.0, 1.5),
    R = diag(3)),
  list(id = "two_doses", graph = graphs$two_doses, mu = c(2.8, 2.5, 2.2, 2.0), R = r4),
  list(id = "two_doses_global_null", graph = graphs$two_doses, mu = rep(0, 4), R = r4),
  list(id = "general4_negative_corr", graph = graphs$general4, mu = c(3.0, 2.2, 2.6, 1.0),
    R = rbind(c(1, -0.3, 0.6, 0), c(-0.3, 1, 0, 0.6), c(0.6, 0, 1, -0.2), c(0, 0.6, -0.2, 1)))
)

set.seed(1)
power_out <- character()
for (pc in power_cases) {
  ep <- exact_power(pc$graph, pc$mu, pc$R, alpha)
  a <- list(id = str_q(pc$id), weights = arr(pc$graph$w), transitions = mat(pc$graph$G),
    means = arr(pc$mu), correlation = mat(pc$R), alpha_one_sided = fmt(alpha),
    local_power = arr(ep$local), any_power = fmt(ep$any), all_power = fmt(ep$all),
    expected_rejections = fmt(ep$expected))
  power_out <- c(power_out, paste0("    ", obj(a)))
}

cat("{\n")
cat(sprintf('  "_generated_with": "%s; mvtnorm %s",\n', R.version.string, packageVersion("mvtnorm")))
cat('  "tests": [\n')
cat(paste(test_out, collapse = ",\n"))
cat('\n  ],\n  "power": [\n')
cat(paste(power_out, collapse = ",\n"))
cat("\n  ]\n}\n")
