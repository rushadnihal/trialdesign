# Reference values for M1 (fixed designs). Run from the repository root:
#   Rscript tests/r/fixed_reference_values.R > tests/reference_values_fixed.json
# Uses base R only (stats::power.t.test, stats::power.prop.test, qnorm), so it runs on any R.
# The "schoenfeld" rows evaluate Schoenfeld (1983) directly in R; gsDesign::nEvents gives the
# same number (it implements this formula) and can replace them once gsDesign is available.

fmt <- function(x) formatC(x, digits = 12, format = "g")
row <- function(...) {
  a <- list(...)
  paste0("    {", paste(sprintf('"%s": %s', names(a), unlist(a)), collapse = ", "), "}")
}
q <- function(s) sprintf('"%s"', s)

rows <- c(
  # power.t.test: two-sample, strict = FALSE (one rejection tail)
  row(id = q("normal_n_d0.5"), source = q("stats::power.t.test"), delta = 0.5, sd = 1,
      alpha_two_sided = 0.05, power = 0.8,
      n_per_arm = fmt(power.t.test(delta = 0.5, sd = 1, power = 0.8)$n)),
  row(id = q("normal_n_d3_sd10_a01_p90"), source = q("stats::power.t.test"), delta = 3, sd = 10,
      alpha_two_sided = 0.01, power = 0.9,
      n_per_arm = fmt(power.t.test(delta = 3, sd = 10, power = 0.9, sig.level = 0.01)$n)),
  row(id = q("normal_power_n40"), source = q("stats::power.t.test"), delta = 0.5, sd = 1,
      alpha_two_sided = 0.05, n_per_arm = 40,
      power = fmt(power.t.test(n = 40, delta = 0.5, sd = 1)$power)),
  # power.prop.test: no continuity correction, strict = FALSE
  row(id = q("binary_n_30_45"), source = q("stats::power.prop.test"), p_control = 0.30,
      p_treatment = 0.45, alpha_two_sided = 0.05, power = 0.8,
      n_per_arm = fmt(power.prop.test(p1 = 0.30, p2 = 0.45, power = 0.8)$n)),
  row(id = q("binary_n_10_20_p90"), source = q("stats::power.prop.test"), p_control = 0.10,
      p_treatment = 0.20, alpha_two_sided = 0.05, power = 0.9,
      n_per_arm = fmt(power.prop.test(p1 = 0.10, p2 = 0.20, power = 0.9)$n)),
  row(id = q("binary_power_n100"), source = q("stats::power.prop.test"), p_control = 0.30,
      p_treatment = 0.45, alpha_two_sided = 0.05, n_per_arm = 100,
      power = fmt(power.prop.test(n = 100, p1 = 0.30, p2 = 0.45)$power)),
  # Schoenfeld (1983) events, evaluated in R
  row(id = q("survival_events_hr0.7"), source = q("Schoenfeld (1983) in R"), hazard_ratio = 0.7,
      alpha_two_sided = 0.05, power = 0.8, allocation = 1,
      events = fmt((qnorm(0.975) + qnorm(0.8))^2 / (0.25 * log(0.7)^2))),
  row(id = q("survival_events_hr0.75_2to1"), source = q("Schoenfeld (1983) in R"),
      hazard_ratio = 0.75, alpha_two_sided = 0.05, power = 0.9, allocation = 2,
      events = fmt((qnorm(0.975) + qnorm(0.9))^2 / ((2 / 3) * (1 / 3) * log(0.75)^2)))
)

cat("{\n")
cat(sprintf('  "_generated_with": "%s",\n', R.version.string))
cat('  "fixed": [\n')
cat(paste(rows, collapse = ",\n"))
cat("\n  ]\n}\n")
