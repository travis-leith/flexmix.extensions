test_that("FastFLXPmultinom matches FLXPmultinom on noisy 3-class data", {
  skip_if_not_installed("dplyr")
  skip_if_not_installed("purrr")

  set.seed(42)
  n <- 1000

  data <- tibble::tibble(
    zp_1 = runif(n),
    zp_2 = runif(n),
    x1 = rnorm(n),
    x2 = rnorm(n),
    s_A = 0.5 * zp_1 - zp_2 + 0.5 + rnorm(n, sd = 0.1),
    s_B = -2 * zp_1 - zp_2 + 1.5 + rnorm(n, sd = 0.1)
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

  # data |>
  #   ggplot2::ggplot() +
  #   ggplot2::aes(x = zp_1, y = zp_2, color = true_latent_class) +
  #   ggplot2::geom_point() +
  #   ggplot2::theme_minimal()

  set.seed(1)
  m_ref <- flexmix::flexmix(
    y ~ x1 + x2,
    data = data,
    k = 3,
    concomitant = flexmix::FLXPmultinom(~ zp_1 + zp_2)
  )

  set.seed(1)
  m_fast <- flexmix::flexmix(
    y ~ x1 + x2,
    data = data,
    k = 3,
    concomitant = FastFLXPmultinom(~ zp_1 + zp_2)
  )

  # Component regression coefficients, one numeric vector per component.
  comp_coefs <- function(m) {
    m@components |> sapply(\(cc) cc[[1]]@parameters$coef)
  }

  sort_by_intercept <- function(coefs) {
    i_order <- order(coefs[1, ])
    sorted_coefs <- coefs[, i_order]
    list(
      i_order = i_order,
      sorted_coefs = sorted_coefs
    )
  }

  C_ref <- comp_coefs(m_ref) |> sort_by_intercept()
  C_fast <- comp_coefs(m_fast) |> sort_by_intercept()

  expect_equal(C_ref$sorted_coefs, C_fast$sorted_coefs, tolerance = 0.05)

  # Priors on the same data, with columns aligned to the reference model.
  P_ref <- flexmix::prior(m_ref, newdata = data)[, C_ref$i_order]
  P_fast <- flexmix::prior(m_fast, newdata = data)[, C_fast$i_order]

  expect_lt(mean(abs(P_ref - P_fast)), 0.02)
})
