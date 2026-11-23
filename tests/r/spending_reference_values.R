# Writes tests/spending_reference_values.json (M10, spending options and binding futility). Run
# from the repository root:
#
#   Rscript tests/r/spending_reference_values.R > sp.json && mv sp.json tests/spending_reference_values.json
#
# All values come from rpact::getDesignGroupSequential (sided = 1) and
# rpact::getDesignCharacteristics:
#   hsd:<gamma>   typeOfDesign "asHSD" / typeBetaSpending "bsHSD", gammaA / gammaB = gamma
#   kd:<rho>      typeOfDesign "asKD"  / typeBetaSpending "bsKD",  gammaA / gammaB = rho
#   user:<f>      typeOfDesign "asUser" / typeBetaSpending "bsUser", with
#                 userAlphaSpending = f * alpha and userBetaSpending = f * beta
#   obf, pocock   "asOF" / "bsOF", "asP" / "bsP"
#   binding       bindingFutility = TRUE (gsDesign test.type = 3); FALSE otherwise
# Recorded: cumulative alpha spent, efficacy critical values, futility bounds at looks
# 1..K-1, the inflation factor and the expected sample size relative to the fixed design under
# H0 and H1 (averageSampleNumber0 / averageSampleNumber1, futility followed).
# gsDesign is not installed in the environment that produced the committed JSON (CRAN was not
# reachable); rpact was installed from the Ubuntu r-cran-rpact package (version recorded).

suppressMessages(library(rpact))
options(digits = 15)
fmt <- function(x) formatC(x, digits = 12, format = "g")
arr <- function(x) paste0("[", paste(fmt(x), collapse = ", "), "]")
q <- function(s) sprintf('"%s"', s)

spec_args <- function(spec, total, kind) {
  parts <- strsplit(spec, ":", fixed = TRUE)[[1]]
  fam <- parts[1]
  prefix <- if (kind == "alpha") "as" else "bs"
  out <- list()
  type <- switch(fam, obf = "OF", pocock = "P", hsd = "HSD", kd = "KD", user = "User")
  out[[if (kind == "alpha") "typeOfDesign" else "typeBetaSpending"]] <- paste0(prefix, type)
  if (fam %in% c("hsd", "kd")) {
    out[[if (kind == "alpha") "gammaA" else "gammaB"]] <- as.numeric(parts[2])
  }
  if (fam == "user") {
    f <- as.numeric(strsplit(parts[2], ",", fixed = TRUE)[[1]])
    out[[if (kind == "alpha") "userAlphaSpending" else "userBetaSpending"]] <- f * total
  }
  out
}

cases <- list(
  list(id = "hsd_m4_3", looks = c(1, 2, 3) / 3, eff = "hsd:-4", fut = NA, binding = FALSE, power = 0.9),
  list(id = "hsd_1_uneq", looks = c(0.3, 0.7, 1), eff = "hsd:1", fut = NA, binding = FALSE, power = 0.8),
  list(id = "hsd_m2_4", looks = c(1, 2, 3, 4) / 4, eff = "hsd:-2", fut = NA, binding = FALSE, power = 0.8),
  list(id = "kd_3_3", looks = c(1, 2, 3) / 3, eff = "kd:3", fut = NA, binding = FALSE, power = 0.9),
  list(id = "kd_1_4", looks = c(1, 2, 3, 4) / 4, eff = "kd:1", fut = NA, binding = FALSE, power = 0.8),
  list(id = "user_3", looks = c(1, 2, 3) / 3, eff = "user:0.1,0.4,1", fut = NA, binding = FALSE, power = 0.8),
  list(id = "hsd_hsd_nb", looks = c(1, 2, 3) / 3, eff = "hsd:-4", fut = "hsd:-2", binding = FALSE, power = 0.9),
  list(id = "kd_kd_nb", looks = c(0.4, 0.7, 1), eff = "kd:3", fut = "kd:2", binding = FALSE, power = 0.8),
  list(id = "user_user_nb", looks = c(1, 2, 3) / 3, eff = "user:0.2,0.6,1", fut = "user:0.25,0.5,1", binding = FALSE, power = 0.8),
  list(id = "obf_obf_b", looks = c(1, 2, 3) / 3, eff = "obf", fut = "obf", binding = TRUE, power = 0.8),
  list(id = "pocock_pocock_b", looks = c(0.5, 1), eff = "pocock", fut = "pocock", binding = TRUE, power = 0.9),
  list(id = "hsd_hsd_b", looks = c(1, 2, 3) / 3, eff = "hsd:-4", fut = "hsd:-2", binding = TRUE, power = 0.8),
  list(id = "obf_kd_b_4", looks = c(1, 2, 3, 4) / 4, eff = "obf", fut = "kd:2", binding = TRUE, power = 0.9),
  list(id = "user_hsd_b", looks = c(0.3, 0.6, 1), eff = "user:0.05,0.3,1", fut = "hsd:1", binding = TRUE, power = 0.8)
)

alpha <- 0.025
rows <- character()
for (cs in cases) {
  k <- length(cs$looks)
  beta <- 1 - cs$power
  args <- c(list(kMax = k, alpha = alpha, beta = beta, sided = 1,
    informationRates = cs$looks), spec_args(cs$eff, alpha, "alpha"))
  if (!is.na(cs$fut)) {
    args <- c(args, spec_args(cs$fut, beta, "beta"), list(bindingFutility = cs$binding))
  }
  d <- do.call(getDesignGroupSequential, args)
  ch <- getDesignCharacteristics(d)
  fut <- if (is.na(cs$fut)) "null" else arr(d$futilityBounds)
  rows <- c(rows, paste0("    {",
    paste(c(
      sprintf('"id": %s', q(cs$id)),
      sprintf('"looks": %s', arr(cs$looks)),
      sprintf('"efficacy_spending": %s', q(cs$eff)),
      sprintf('"futility_spending": %s', if (is.na(cs$fut)) "null" else q(cs$fut)),
      sprintf('"binding_futility": %s', if (cs$binding) "true" else "false"),
      sprintf('"alpha_one_sided": %s', fmt(alpha)),
      sprintf('"power": %s', fmt(cs$power)),
      sprintf('"alpha_spent": %s', arr(d$alphaSpent)),
      sprintf('"efficacy_z": %s', arr(d$criticalValues)),
      sprintf('"futility_z": %s', fut),
      sprintf('"inflation_factor": %s', fmt(ch$inflationFactor)),
      sprintf('"asn_h0": %s', fmt(ch$averageSampleNumber0)),
      sprintf('"asn_h1": %s', fmt(ch$averageSampleNumber1))
    ), collapse = ", "), "}"))
}

cat("{\n")
cat(sprintf('  "generated_by": "tests/r/spending_reference_values.R",\n'))
cat(sprintf('  "rpact_version": "%s",\n', as.character(packageVersion("rpact"))))
cat(sprintf('  "r_version": "%s",\n', paste(R.version$major, R.version$minor, sep = ".")))
cat('  "cases": [\n')
cat(paste(rows, collapse = ",\n"))
cat("\n  ]\n}\n")
