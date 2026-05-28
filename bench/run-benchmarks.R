#!/usr/bin/env Rscript
# Run the benchmark grid and render the Quarto report.
#
# Usage (from package root):
#   Rscript bench/run-benchmarks.R

set.seed(1)
source("bench/benchmark-core.R")

results <- run_grid()

dir.create("bench/results", recursive = TRUE, showWarnings = FALSE)
saveRDS(
  list(
    results = results,
    generated = Sys.time(),
    r_version = getRversion(),
    pkg_flexmix = packageVersion("flexmix"),
    pkg_extensions = packageVersion("flexmix.extensions")
  ),
  "bench/results/fast-multinom.rds"
)

message("Rendering report ...")
quarto::quarto_render(
  input = "bench/benchmark-fast-multinom.qmd",
  output_format = "gfm"
)

# Move output into results/ (Quarto writes next to the input).
file.rename(
  "bench/benchmark-fast-multinom.md",
  "bench/results/fast-multinom.md"
)
if (dir.exists("bench/benchmark-fast-multinom_files")) {
  unlink("bench/results/fast-multinom_files", recursive = TRUE)
  file.rename(
    "bench/benchmark-fast-multinom_files",
    "bench/results/fast-multinom_files"
  )
}

message("Wrote bench/results/fast-multinom.md")
