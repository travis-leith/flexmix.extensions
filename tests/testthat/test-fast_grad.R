# tests/testthat/test-fast_grad.R

# Oracle: the original FLXP method, fetched directly from flexmix.
# Using selectMethod() bypasses our FLXPmultinom override.
parent_flxp_method <- function() {
  selectMethod("FLXgradlogLikfun", "FLXP")
}

ref_flxp_scores <- function(X, fitted, weights) {
  obj <- new("FLXP", x = X)
  parent_flxp_method()(obj, fitted, weights)
}

mk_inputs <- function(N, r, K, seed = 1L) {
  set.seed(seed)
  X <- cbind(1, matrix(rnorm(N * (r - 1)), N, r - 1))
  colnames(X) <- c("(Intercept)", paste0("z", seq_len(r - 1)))
  rs <- function() {
    M <- matrix(runif(N * K), N, K)
    M / rowSums(M)
  }
  list(X = X, fitted = rs(), weights = rs())
}

mk_multinom_obj <- function(X) {
  obj <- new("FLXPmultinom", name = "FLXPmultinom", formula = ~1)
  obj@x <- X
  obj
}

expect_scores_equal <- function(N, r, K, tol = 1e-12) {
  d <- mk_inputs(N, r, K)
  obj <- mk_multinom_obj(d$X)

  # Fast path: via S4 dispatch (exercises the installed method).
  got <- flexmix::FLXgradlogLikfun(obj, d$fitted, d$weights)
  # Oracle: original FLXP method invoked directly on the same object.
  ref <- parent_flxp_method()(obj, d$fitted, d$weights)

  expect_type(got, "list")
  expect_equal(length(got), K - 1)
  for (k in seq_along(ref)) {
    expect_equal(
      dim(got[[k]]),
      dim(ref[[k]]),
      info = sprintf("dim mismatch k=%d (N=%d,r=%d,K=%d)", k, N, r, K)
    )
    expect_equal(
      unname(got[[k]]),
      unname(ref[[k]]),
      tolerance = tol,
      info = sprintf("values differ k=%d (N=%d,r=%d,K=%d)", k, N, r, K)
    )
  }
}

test_that("fast method is actually installed for FLXPmultinom", {
  m_fast <- selectMethod("FLXgradlogLikfun", "FLXPmultinom")
  m_parent <- parent_flxp_method()
  expect_false(identical(body(m_fast), body(m_parent)))
})

test_that("matches inherited FLXP method across shapes", {
  expect_scores_equal(50, 3, 2) # K = 2 edge case
  expect_scores_equal(200, 5, 3)
  expect_scores_equal(1000, 4, 5)
  expect_scores_equal(5000, 10, 4)
})

test_that("K = 2 returns a length-1 list with correct values", {
  d <- mk_inputs(100, 3, 2)
  obj <- mk_multinom_obj(d$X)
  got <- flexmix::FLXgradlogLikfun(obj, d$fitted, d$weights)
  expect_length(got, 1L)
  manual <- d$X * (d$weights[, 2] - d$fitted[, 2])
  expect_equal(unname(got[[1]]), unname(manual))
})

test_that("ordering: list element k corresponds to class k+1", {
  d <- mk_inputs(100, 3, 4)
  obj <- mk_multinom_obj(d$X)
  got <- flexmix::FLXgradlogLikfun(obj, d$fitted, d$weights)
  manual <- d$X * (d$weights[, 3] - d$fitted[, 3]) # class 3 -> index 2
  expect_equal(unname(got[[2]]), unname(manual))
})

test_that("zero residual => zero scores", {
  d <- mk_inputs(50, 4, 3)
  d$weights <- d$fitted
  obj <- mk_multinom_obj(d$X)
  got <- flexmix::FLXgradlogLikfun(obj, d$fitted, d$weights)
  for (M in got) {
    expect_true(all(M == 0))
  }
})

test_that("mismatched dims raise an error", {
  d <- mk_inputs(50, 3, 3)
  bad_weights <- d$weights[, -1, drop = FALSE]
  expect_error(cpp_multinom_scores(d$X, d$fitted, bad_weights))
  bad_fitted <- d$fitted[-1, , drop = FALSE]
  expect_error(cpp_multinom_scores(d$X, bad_fitted, d$weights))
})

test_that("refit() agrees with parent FLXP method", {
  skip_on_cran()
  set.seed(42)
  n <- 1000
  dd <- data.frame(
    x = rnorm(n),
    z1 = rnorm(n),
    z2 = rnorm(n)
  )
  dd$y <- 2 + 1.5 * dd$x + rnorm(n)

  m <- flexmix::flexmix(
    y ~ x,
    data = dd,
    k = 2,
    concomitant = flexmix::FLXPmultinom(~ z1 + z2),
    control = list(iter.max = 50)
  )

  r_fast <- flexmix::refit(m)

  # Temporarily shadow with the parent method to get the "slow" reference,
  # then restore the fast method afterwards.
  fast_method <- selectMethod("FLXgradlogLikfun", "FLXPmultinom")
  parent_fun <- parent_flxp_method()
  setMethod("FLXgradlogLikfun", "FLXPmultinom", parent_fun)
  on.exit(
    setMethod("FLXgradlogLikfun", "FLXPmultinom", fast_method),
    add = TRUE
  )

  r_slow <- flexmix::refit(m)

  expect_equal(r_fast@coef, r_slow@coef, tolerance = 1e-6)
  expect_equal(r_fast@vcov, r_slow@vcov, tolerance = 1e-6)
})

test_that("benchmark sanity (informational)", {
  skip_on_cran()
  d <- mk_inputs(50000, 6, 5)
  obj <- mk_multinom_obj(d$X)
  parent_fun <- parent_flxp_method()
  t_ref <- system.time(parent_fun(obj, d$fitted, d$weights))[["elapsed"]]
  t_cpp <- system.time(flexmix::FLXgradlogLikfun(obj, d$fitted, d$weights))[[
    "elapsed"
  ]]
  message(sprintf(
    "ref=%.3fs  cpp=%.3fs  speedup=%.1fx",
    t_ref,
    t_cpp,
    t_ref / t_cpp
  ))
  expect_true(t_cpp <= t_ref * 2)
})
