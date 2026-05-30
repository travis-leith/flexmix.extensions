#!/usr/bin/env Rscript
# Run the benchmark grid and render the Quarto report.
#
# Usage (from package root):
#   Rscript bench/run-benchmarks.R
#   Rscript bench/run-benchmarks.R --render-only
library(tidyr)
args <- commandArgs(trailingOnly = TRUE)

render_only <- "--render-only" %in% args

unknown_args <- setdiff(args, "--render-only")
if (length(unknown_args) > 0) {
  stop(
    "Unknown argument(s): ",
    paste(unknown_args, collapse = ", "),
    call. = FALSE
  )
}

rds_path <- "bench/results/fast-multinom.rds"

dir.create("bench/results", recursive = TRUE, showWarnings = FALSE)

if (render_only) {
  if (!file.exists(rds_path)) {
    stop(
      "Cannot render only because the benchmark RDS does not exist: ",
      rds_path,
      call. = FALSE
    )
  }

  readRDS(rds_path)
} else {
  set.seed(1)
  source("bench/benchmark-core.R")

  results <- run_grid()
  list(
    results = results,
    generated = Sys.time(),
    r_version = getRversion(),
    pkg_flexmix = packageVersion("flexmix"),
    pkg_extensions = packageVersion("flexmix.extensions")
  ) |>
    saveRDS(rds_path)
}

message("Rendering report ...")

quarto::quarto_render(
  input = "bench/benchmark-fast-multinom.qmd",
  output_format = "gfm",
  output_file = "benchmark-fast-multinom.md"
)

rendered_md <- "bench/benchmark-fast-multinom.md"
rendered_files <- "bench/benchmark-fast-multinom_files"

dest_md <- "bench/results/benchmark-fast-multinom.md"
dest_files <- "bench/results/benchmark-fast-multinom_files"

if (dir.exists(dest_files)) {
  unlink(dest_files, recursive = TRUE)
}

file.rename(rendered_md, dest_md)
file.rename(rendered_files, dest_files)

message("Wrote ", dest_md)
