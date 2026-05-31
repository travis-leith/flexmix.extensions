#' @useDynLib flexmix.extensions, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

## Drop-in replacement for flexmix::FLXPmultinom using a C++ Newton-Raphson solver.
## Source this file AFTER loading flexmix and Rcpp/RcppArmadillo.
##
## Usage:
##   library(flexmix)
##   library(Rcpp)
##   library(RcppArmadillo)
##   Rcpp::sourceCpp("fast_multinom/fast_multinom.cpp")
##   source("fast_multinom/fast_multinom.R")
##   # Now use FastFLXPmultinom() exactly like FLXPmultinom().
##   flexmix(y ~ x, data = d, k = 3, concomitant = FastFLXPmultinom(~ z))

stopifnot(requireNamespace("flexmix", quietly = TRUE))
stopifnot(exists("fast_multinom_fit_cpp"))

## Build a multinom-compatible object so summary()/vcov()/refit(method="optim")
## continue to work without any other code changes.
.make_multinom_like <- function(B, X, Y, w, fit) {
  ## B: r x (K-1); expose coefficients as (K-1) x r, like nnet::multinom.
  r <- ncol(X)
  K <- ncol(Y)
  Km1 <- K - 1L
  coefs <- t(B)
  rownames(coefs) <- as.character(seq_len(Km1) + 1L)
  colnames(coefs) <- colnames(X)

  ## Mimic nnet.default structure: inputs = r, hidden = 0, outputs = K, skip-layer.
  ## Each output unit connects to a bias + all r inputs => (r+1) connections per output.
  n_inputs <- r
  n_hidden <- 0L
  n_outputs <- K
  nunits <- 1L + n_inputs + n_hidden + n_outputs # bias + inputs + outputs
  conn_per_out <- n_inputs + 1L # bias + inputs (skip layer)
  total_conn <- conn_per_out * n_outputs

  ## nconn: cumulative pointer, length nunits + 1. Only output units have incoming conns.
  nconn <- integer(nunits + 1L)
  out_start <- 1L + n_inputs + n_hidden + 1L # first output unit index (1-based)
  for (u in seq_len(nunits + 1L)) {
    if (u <= out_start) {
      nconn[u] <- 0L
    } else {
      nconn[u] <- (u - out_start) * conn_per_out
    }
  }
  nconn[nunits + 1L] <- total_conn

  ## conn: which source units feed each output. nnet uses 0-based with 0 = bias.
  conn <- as.integer(rep(c(0L, seq_len(n_inputs)), n_outputs))

  ## wts: full weight vector. Build so that reference class (k=1) is all zeros and
  ## classes 2..K hold [intercept=0, B[, k-1]]. We don't have a separate intercept
  ## column in X here — flexmix's design X already contains the intercept column —
  ## but to keep length consistent with nnet's layout we still emit (r+1) per output.
  ## (This object is only used for summary/printing through our overrides; the real
  ## coefficients live in $coefficients and $Hessian.)
  wts_mat <- matrix(0, nrow = conn_per_out, ncol = n_outputs)
  ## Put coefs (r x (K-1)) into columns 2..K, rows 2..(r+1). Bias row stays 0.
  if (Km1 > 0) {
    wts_mat[2:(r + 1L), 2:n_outputs] <- B
  }
  wts <- as.vector(wts_mat)

  obj <- list()
  obj$n <- c(n_inputs, n_hidden, n_outputs)
  obj$nunits <- nunits
  obj$nconn <- nconn
  obj$conn <- conn
  obj$nsunits <- nunits
  obj$decay <- 0
  obj$entropy <- TRUE
  obj$softmax <- TRUE
  obj$censored <- FALSE
  obj$value <- -fit$loglik
  obj$wts <- wts
  obj$convergence <- if (fit$converged) 0L else 1L
  obj$fitted.values <- fit$fitted
  obj$residuals <- Y - fit$fitted
  obj$lev <- colnames(Y)
  obj$call <- match.call()
  obj$terms <- NULL
  obj$coefnames <- colnames(X)
  obj$vcoefnames <- colnames(X)
  obj$Hessian <- fit$hessian
  dn <- paste(rep(seq_len(Km1) + 1L, each = r), colnames(X), sep = ":")
  dimnames(obj$Hessian) <- list(dn, dn)
  obj$edf <- length(B)
  obj$AIC <- 2 * obj$value + 2 * obj$edf
  obj$weights <- w
  obj$deviance <- 2 * obj$value
  obj$rank <- length(B)
  obj$coefficients <- coefs
  class(obj) <- c("multinom", "nnet")
  obj
}

## We define a coef.multinom shim only if nnet isn't loaded; otherwise rely on nnet's.
## (nnet::coef.multinom returns the same (K-1) x r matrix.)

#' Fast multinomial concomitant model for flexmix
#'
#' Concomitant model driver that estimates component priors using a multinomial
#' logit model. It serves as a high-performance drop-in replacement for
#' [flexmix::FLXPmultinom()], leveraging a C++ Newton-Raphson solver for
#' accelerated parameter estimation.
#'
#' Fits multinomial logit models for concomitant variables in mixture-of-experts
#' models. Because it constructs a `multinom`-compatible S4 object, downstream
#' methods like `summary()`, `vcov()`, and `refit(method = "optim")` continue
#' work seamlessly.
#'
#' @param formula A one-sided formula of concomitant regressors.
#'   Defaults to `~1`.
#' @param max_iter Integer specifying the maximum number of Newton-Raphson
#'   iterations. Defaults to `100L`.
#' @param tol Numeric scalar defining convergence tolerance. Defaults to `1e-8`.
#' @param ridge Numeric scalar defining a ridge regularization penalty added to the
#'   Hessian diagonal to prevent numerical instability or separation issues.
#'   Defaults to `1e-6`.
#' @param nthreads Integer indicating the number of threads for parallel computation.
#'   Defaults to `0L` (sequential execution).
#'
#' @return An object of class `FLXPmultinom` (subclass of `FLXP`) to be passed
#'   as the `concomitant` argument to [flexmix::flexmix()].
#'
#' @examples
#' set.seed(1)
#' n <- 500
#' z <- rnorm(n)
#' y <- rnorm(n, mean = ifelse(z > 0, 2, -2))
#' df <- data.frame(y = y, z = z)
#' flexmix::flexmix(
#'   y ~ 1,
#'   data = df, k = 2,
#'   concomitant = FastFLXPmultinom(~z)
#' )
#'
#' @seealso [flexmix::FLXPmultinom()]
#' @import flexmix
#' @export
FastFLXPmultinom <- function(
  formula = ~1,
  max_iter = 100L,
  tol = 1e-8,
  ridge = 1e-6,
  nthreads = 0L
) {
  z <- new("FLXPmultinom", name = "FLXPmultinom", formula = formula)

  z@fit <- function(x, y, w, ...) {
    if (missing(w) || is.null(w)) {
      w <- rep(1, nrow(y))
    }
    x <- as.matrix(x)
    y <- as.matrix(y)
    ## Cold start every call. EM iterations move responsibilities around;
    ## warm-starting from stale B can blow up the Newton step.
    fit <- fast_multinom_fit_cpp(
      x,
      y,
      as.numeric(w),
      max_iter = max_iter,
      tol = tol,
      ridge = ridge,
      nthreads = nthreads,
      B_init = NULL
    )
    fit$fitted
  }

  z@refit <- function(x, y, w, ...) {
    if (missing(w) || is.null(w)) {
      w <- rep(1, nrow(y))
    }
    x <- as.matrix(x)
    y <- as.matrix(y)
    rownames(y) <- rownames(x) <- NULL
    if (is.null(colnames(x))) {
      colnames(x) <- paste0("x", seq_len(ncol(x)))
    }
    if (is.null(colnames(y))) {
      colnames(y) <- as.character(seq_len(ncol(y)))
    }
    fit <- fast_multinom_fit_cpp(
      x,
      y,
      as.numeric(w),
      max_iter = max_iter,
      tol = tol,
      ridge = ridge,
      nthreads = nthreads,
      B_init = NULL
    )
    .make_multinom_like(fit$coef, x, y, w, fit)
  }
  z
}
