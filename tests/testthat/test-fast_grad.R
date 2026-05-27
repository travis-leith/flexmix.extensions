# library(flexmix)
# library(Rcpp)
# library(RcppArmadillo)
# library(testthat)

# --- Reference implementation: exact copy of the original FLXP method ------
ref_flxp_scores <- function(X, fitted, weights) {
  Pi <- lapply(seq_len(ncol(fitted))[-1], function(i) {
    -fitted[, i] + weights[, i]
  })
  lapply(Pi, function(p) apply(X, 2, "*", p))
}

mk_inputs <- function(N, r, K, seed = 1L) {
  set.seed(seed)
  X <- cbind(1, matrix(rnorm(N * (r - 1)), N, r - 1))
  colnames(X) <- c("(Intercept)", paste0("z", seq_len(r - 1)))
  # random row-stochastic matrices
  rs <- function() {
    M <- matrix(runif(N * K), N, K)
    M / rowSums(M)
  }
  list(X = X, fitted = rs(), weights = rs())
}

expect_scores_equal <- function(N, r, K, tol = 1e-12) {
  d <- mk_inputs(N, r, K)
  ref <- ref_flxp_scores(d$X, d$fitted, d$weights)
  got <- cpp_multinom_scores(d$X, d$fitted, d$weights)

  expect_type(got, "list")
  expect_equal(length(got), K - 1)
  for (k in seq_along(ref)) {
    expect_equal(
      dim(got[[k]]),
      dim(ref[[k]]),
      info = sprintf("dim mismatch at k=%d (N=%d,r=%d,K=%d)", k, N, r, K)
    )
    expect_equal(
      unname(got[[k]]),
      unname(ref[[k]]),
      tolerance = tol,
      info = sprintf("values differ at k=%d (N=%d,r=%d,K=%d)", k, N, r, K)
    )
  }
}

test_that("matches reference across shapes", {
  expect_scores_equal(50, 3, 2)
  expect_scores_equal(200, 5, 3)
  expect_scores_equal(1000, 4, 5)
  expect_scores_equal(5000, 10, 4)
})

test_that("ordering: list element k corresponds to class k+1 (reference = class 1)", {
  d <- mk_inputs(100, 3, 4)
  got <- cpp_multinom_scores(d$X, d$fitted, d$weights)
  # Hand-compute class 3 (=> list index 2)
  p <- d$weights[, 3] - d$fitted[, 3]
  manual <- d$X * p # recycling: p applied down each column
  expect_equal(unname(got[[2]]), unname(manual))
})

test_that("zero residual => zero scores", {
  d <- mk_inputs(50, 4, 3)
  d$weights <- d$fitted
  got <- cpp_multinom_scores(d$X, d$fitted, d$weights)
  for (M in got) {
    expect_true(all(M == 0))
  }
})

# test_that("end-to-end: dispatch via FLXgradlogLikfun on FLXPmultinom", {
#   # Register the fast method
#   setMethod("FLXgradlogLikfun", signature(object = "FLXPmultinom"),
#     function(object, fitted, weights, ...) {
#       cpp_multinom_scores(object@x, as.matrix(fitted), as.matrix(weights))
#     })

#   d <- mk_inputs(300, 4, 3)
#   obj <- new("FLXPmultinom", name = "FLXPmultinom", formula = ~1)
#   obj@x <- d$X
#   obj@coef <- matrix(0, nrow = ncol(d$X), ncol = 3)  # not used by the gradient

#   fast <- FLXgradlogLikfun(obj, d$fitted, d$weights)
#   ref  <- ref_flxp_scores(d$X, d$fitted, d$weights)
#   for (k in seq_along(ref))
#     expect_equal(unname(fast[[k]]), unname(ref[[k]]), tolerance = 1e-12)

#   # Restore original behavior so other tests/sessions aren't affected
#   removeMethod("FLXgradlogLikfun", signature(object = "FLXPmultinom"))
# })

test_that("benchmark sanity (informational, not asserted)", {
  skip_on_cran()
  d <- mk_inputs(50000, 6, 5)
  t_ref <- system.time(ref_flxp_scores(d$X, d$fitted, d$weights))[["elapsed"]]
  t_cpp <- system.time(cpp_multinom_scores(d$X, d$fitted, d$weights))[[
    "elapsed"
  ]]
  message(sprintf(
    "ref=%.3fs  cpp=%.3fs  speedup=%.1fx",
    t_ref,
    t_cpp,
    t_ref / t_cpp
  ))
  expect_true(t_cpp <= t_ref * 2) # extremely lax sanity bound
})

# test_that("refit() gives same coef/vcov with fast gradient", {
#   set.seed(42)
#   n <- 2000
#   d <- data.frame(x = rnorm(n), z1 = rnorm(n), z2 = rnorm(n))
#   d$y <- 2 + 1.5 * d$x + rnorm(n)

#   m <- flexmix(y ~ x, data = d, k = 2,
#                concomitant = FLXPmultinom(~ z1 + z2),
#                control = list(iter.max = 50))

#   r_slow <- refit(m)
#   setMethod("FLXgradlogLikfun", signature(object = "FLXPmultinom"),
#     function(object, fitted, weights, ...)
#       cpp_multinom_scores(object@x, as.matrix(fitted), as.matrix(weights)))
#   r_fast <- refit(m)
#   removeMethod("FLXgradlogLikfun", signature(object = "FLXPmultinom"))

#   expect_equal(r_slow@coef, r_fast@coef, tolerance = 1e-6)
#   expect_equal(r_slow@vcov, r_fast@vcov, tolerance = 1e-6)
# })
