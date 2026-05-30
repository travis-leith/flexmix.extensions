# Core benchmark logic. No knitr/Quarto involvement.
# Sourced by run-benchmarks.R.

suppressPackageStartupMessages({
  library(flexmix)
  library(flexmix.extensions)
  library(bench)
  library(dplyr)
  library(tidyr)
})

make_data <- function(n, n_concomitant, k) {
  Z <- matrix(runif(n * n_concomitant), n, n_concomitant)
  colnames(Z) <- paste0("zp_", seq_len(n_concomitant))
  W <- matrix(rnorm(n_concomitant * k), n_concomitant, k)
  scores <- Z %*% W + matrix(rnorm(n * k, sd = 0.1), n, k)
  cls <- max.col(scores)
  beta <- matrix(rnorm(2 * k), 2, k)
  x1 <- rnorm(n)
  y <- beta[1, cls] + beta[2, cls] * x1 + rnorm(n)
  cbind(tibble::tibble(y = y, x1 = x1), Z)
}

fit_one <- function(data, k, concomitant_fun, n_concomitant) {
  rhs <- paste(paste0("zp_", seq_len(n_concomitant)), collapse = " + ")
  conc <- concomitant_fun(as.formula(paste("~", rhs)))
  flexmix(
    y ~ x1,
    data = data,
    k = k,
    concomitant = conc,
    control = list(minprior = 0)
  )
}

default_grid <- function() {
  expand_grid(
    n = c(500L, 2000L, 8000L),
    k = c(2L, 3L, 5L),
    n_concomitant = c(2L, 5L, 10L)
  )
}

run_grid <- function(grid = default_grid(), iterations = 5L) {
  total <- nrow(grid)
  t_start <- Sys.time()

  rows <- purrr::pmap(
    mutate(grid, .idx = row_number()),
    function(n, k, n_concomitant, .idx) {
      t0 <- Sys.time()

      stringr::str_glue(
        "[{.idx}/{total}] n={n}  k={k}  n_concomitant={n_concomitant} ..."
      ) |>
        message()

      data <- make_data(n, n_concomitant, k)
      b <- bench::mark(
        ref = fit_one(data, k, flexmix::FLXPmultinom, n_concomitant),
        fast = fit_one(data, k, FastFLXPmultinom, n_concomitant),
        iterations = iterations,
        check = FALSE,
        filter_gc = FALSE
      )
      dt <- as.numeric(Sys.time() - t0, units = "secs")
      stringr::str_glue("         done in {round(dt, 1)}s") |> message()

      tibble(
        n = n,
        k = k,
        n_concomitant = n_concomitant,
        impl = as.character(b$expression),
        median = as.numeric(b$median)
      )
    }
  )

  total_time <- as.numeric(Sys.time() - t_start, units = "secs")
  stringr::str_glue(
    "Total elapsed: {round(total_time, 1)}s"
  ) |>
    message()

  dplyr::bind_rows(rows) |>
    pivot_wider(
      names_from = impl,
      values_from = median,
      names_prefix = "median_"
    ) |>
    mutate(speedup = median_ref / median_fast)
}
