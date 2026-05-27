test_that("FastFLXPmultinom matches FLXPmultinom on noisy 3-class data", {
  skip_if_not_installed("dplyr")
  skip_if_not_installed("purrr")

  set.seed(42)
  n <- 10000

  data <- tibble::tibble(
    zp_1 = runif(n),
    zp_2 = runif(n),
    x1 = rnorm(n),
    x2 = rnorm(n),
    s_A = 0.5 * zp_1 - zp_2 + 0.5 + rnorm(n, sd = 0.2),
    s_B = -2 * zp_1 - zp_2 + 1.5 + rnorm(n, sd = 0.2)
  ) |>
    dplyr::mutate(
      true_latent_class = dplyr::case_when(
        s_A < 0 ~ "A",
        s_B > 0 ~ "B",
        TRUE ~ "C"
      ),
      y = dplyr::case_when(
        true_latent_class == "A" ~ 1 + 0.5 * x1 + 0.2 * x2 + rnorm(n),
        true_latent_class == "B" ~ -1 + 0.3 * x1 - 0.4 * x2 + rnorm(n),
        TRUE ~ 0.1 * x1 + 0.5 * x2 + rnorm(n)
      )
    ) |>
    dplyr::select(-s_A, -s_B)

  # Shared initial cluster assignment removes one source of EM non-determinism.
  init_cluster <- sample.int(3, n, replace = TRUE)

  set.seed(1)
  m_ref <- flexmix(
    y ~ x1 + x2,
    data = data,
    cluster = init_cluster,
    concomitant = FLXPmultinom(~ zp_1 + zp_2)
  )

  set.seed(1)
  m_fast <- flexmix(
    y ~ x1 + x2,
    data = data,
    cluster = init_cluster,
    concomitant = FastFLXPmultinom(~ zp_1 + zp_2)
  )

  expect_equal(m_ref@k, 3L)
  expect_equal(m_fast@k, 3L)

  # Component regression coefficients, one numeric vector per component.
  comp_coefs <- function(m) {
    m@components[[1]] |> purrr::map(\(cc) cc@parameters$coef)
  }
  C_ref <- comp_coefs(m_ref)
  C_fast <- comp_coefs(m_fast)
  k <- length(C_ref)

  # Align components across fits by minimum pairwise coef distance (greedy).
  perm <- purrr::reduce(
    seq_len(k),
    \(acc, i) {
      remaining <- setdiff(seq_len(k), acc)
      dists <- purrr::map_dbl(remaining, \(j) {
        sqrt(sum((C_ref[[i]] - C_fast[[j]])^2))
      })
      c(acc, remaining[which.min(dists)])
    },
    .init = integer()
  )

  purrr::walk(seq_len(k), \(i) {
    expect_equal(C_ref[[i]], C_fast[[perm[i]]], tolerance = 0.05)
  })

  # Priors on the same data, with columns aligned to the reference model.
  P_ref <- prior(m_ref, newdata = data)
  P_fast <- prior(m_fast, newdata = data)[, perm, drop = FALSE]

  expect_lt(mean(abs(P_ref - P_fast)), 0.02)
})
