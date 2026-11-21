# Writes tests/reference_values.json. Run from the repository root:
#
#   Rscript tests/r/reference_values.R > ref.json && mv ref.json tests/reference_values.json
#
# Needs gsDesign and rpact. Cross-check messages go to stderr. The script stops if a strict
# cross-check fails; writing to ref.json first keeps the old JSON in that case.

suppressPackageStartupMessages({
  library(gsDesign)
  library(rpact)
})

source("tests/r/fixed_reference_values.R")  # defines fixed_rows and the helpers
source("tests/r/gsd_reference_values.R")    # defines gsd_rows

out <- c(
  "{",
  sprintf('  "_generated_with": "%s; gsDesign %s; rpact %s",', R.version.string,
          packageVersion("gsDesign"), packageVersion("rpact")),
  '  "fixed": [', paste(fixed_rows, collapse = ",\n"), "  ],",
  '  "boundaries": [', paste(gsd_rows, collapse = ",\n"), "  ]",
  "}"
)
cat(out, sep = "\n")
