// src/fast_grad.cpp
//
// Score matrices for the FLXPmultinom concomitant in flexmix's
// FLXgradlogLikfun. Replaces this R loop (flexmix/R/refit.R):
//
//   Pi <- lapply(seq_len(ncol(fitted))[-1],
//                function(i) -fitted[, i] + weights[, i])
//   lapply(Pi, function(p) apply(X, 2, "*", p))
//
// Returns a list of (K-1) N x r matrices with
//   out[[k]][, j] = X[, j] * (weights[, k+1] - fitted[, k+1]).

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List cpp_multinom_scores(const arma::mat& X,
                         const arma::mat& fitted,
                         const arma::mat& weights,
                         int nthreads = 0) {
  const uword N = X.n_rows;
  const uword r = X.n_cols;
  const uword K = fitted.n_cols;
  if (weights.n_rows != N || weights.n_cols != K)
    stop("cpp_multinom_scores: weights must match fitted dims");
  if (fitted.n_rows != N)
    stop("cpp_multinom_scores: fitted rows must match X rows");
  if (K < 2) stop("cpp_multinom_scores: need at least 2 classes");

#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads(nthreads);
#endif

  const uword Km1 = K - 1;
  std::vector<arma::mat> S(Km1);

  // Parallelise over class index; the inner column-wise expression is
  // already BLAS-friendly and avoids the row-major access pattern that
  // fought Armadillo's column-major layout in the previous version.
#pragma omp parallel for schedule(static) if(Km1 > 1 && N * r > 100000)
  for (uword k = 1; k < K; ++k) {
    const vec d = weights.col(k) - fitted.col(k);   // N x 1
    mat& M = S[k - 1];
    M.set_size(N, r);
    for (uword j = 0; j < r; ++j) M.col(j) = X.col(j) % d;
  }

  List out(Km1);
  for (uword k = 0; k < Km1; ++k) out[k] = wrap(std::move(S[k]));
  return out;
}