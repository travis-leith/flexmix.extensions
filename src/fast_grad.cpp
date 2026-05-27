// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::plugins(cpp14)]]
//
// Score matrices for the FLXPmultinom concomitant model, used inside
// FLXgradlogLikfun on signature("FLXPmultinom").
//
// Reference (flexmix/R/refit.R, FLXP method):
//   Pi <- lapply(seq_len(ncol(fitted))[-1],
//                function(i) -fitted[, i] + weights[, i])
//   lapply(Pi, function(p) apply(X, 2, "*", p))
//
// Returns list of (K-1) N x r matrices:
//   out[[k]][i, j] = X[i, j] * (weights[i, k+1] - fitted[i, k+1])

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
  
  // Allocate the K-1 output matrices up front; write directly into them.
  std::vector<arma::mat> S(K - 1);
  for (uword k = 0; k < K - 1; ++k) S[k].set_size(N, r);
  
  // Parallelise over rows; only kick in OpenMP for large N to avoid
  // fork/join overhead dominating on tiny problems.
#pragma omp parallel for schedule(static) if(N > 10000)
  for (uword i = 0; i < N; ++i) {
    for (uword k = 1; k < K; ++k) {
      const double p = weights(i, k) - fitted(i, k);
      arma::mat& M = S[k - 1];
      for (uword j = 0; j < r; ++j) {
        M(i, j) = X(i, j) * p;
      }
    }
  }
  
  List out(K - 1);
  for (uword k = 0; k < K - 1; ++k) out[k] = wrap(S[k]);
  return out;
}