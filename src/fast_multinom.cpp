// src/fast_multinom.cpp
//
// Newton-Raphson fitter for a multinomial logit with reference class 0.
// Used as a drop-in replacement for nnet::multinom inside flexmix's
// FLXPmultinom concomitant model. Returns coefficients, fitted probabilities,
// observed-information Hessian, log-likelihood, and convergence info.
//
// Parameterisation: B is (r x (K-1)); class 0 is the reference with eta_0 = 0.
// Penalty: Tikhonov ridge on non-intercept coefficients, scaled to mean(diag(H)).

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

namespace {

constexpr double kLogFloor      = 1e-300;  // clamp for log()
constexpr double kMaxStepInf    = 5.0;     // cap on ||step||_inf
constexpr int    kMaxLineSearch = 40;
constexpr double kInterceptTol  = 1e-12;   // range tolerance for constant col
constexpr double kInterceptRidge = 1e-10;

// Row-wise softmax in place, numerically stable.
inline void softmax_inplace(mat& Eta) {
  const uword n = Eta.n_rows;
  const uword K = Eta.n_cols;
#pragma omp parallel for schedule(static) if(n > 1024)
  for (uword i = 0; i < n; ++i) {
    double m = Eta(i, 0);
    for (uword k = 1; k < K; ++k) if (Eta(i, k) > m) m = Eta(i, k);
    double s = 0.0;
    for (uword k = 0; k < K; ++k) { Eta(i, k) = std::exp(Eta(i, k) - m); s += Eta(i, k); }
    const double inv = 1.0 / s;
    for (uword k = 0; k < K; ++k) Eta(i, k) *= inv;
  }
}

// Weighted multinomial log-likelihood: sum_i w_i * sum_k Y_ik * log(P_ik).
inline double weighted_loglik(const mat& P, const mat& Y, const vec& wv) {
  const uword n = P.n_rows;
  const uword K = P.n_cols;
  double ll = 0.0;
#pragma omp parallel for reduction(+:ll) schedule(static) if(n > 1024)
  for (uword i = 0; i < n; ++i) {
    double s = 0.0;
    for (uword k = 0; k < K; ++k) {
      if (Y(i, k) > 0.0) s += Y(i, k) * std::log(std::max(P(i, k), kLogFloor));
    }
    ll += wv(i) * s;
  }
  return ll;
}

// Linear predictor -> softmax probabilities, with optional clipping for stability.
inline mat compute_P(const mat& X, const mat& B, double eta_clip) {
  const uword n = X.n_rows;
  const uword K = B.n_cols + 1;
  mat P(n, K, fill::zeros);
  if (B.n_cols > 0) {
    P.cols(1, K - 1) = clamp(X * B, -eta_clip, eta_clip);
  }
  softmax_inplace(P);
  return P;
}

// Build the Fisher-information Hessian into H (parallelised over class pairs).
// H is (r*(K-1)) x (r*(K-1)). Blocks: H_{k,k} = X' diag(w * p_k (1-p_k)) X,
//                                    H_{k,j} = -X' diag(w * p_k p_j) X.
void build_hessian(const mat& X, const mat& P, const vec& wv, mat& H) {
  const uword r = X.n_cols;
  const uword K = P.n_cols;
  const uword Km1 = K - 1;
  H.zeros();

  // Pre-build diagonal blocks, then off-diagonals, both parallelisable.
#pragma omp parallel for schedule(dynamic)
  for (uword k = 1; k < K; ++k) {
    vec dkk = wv % P.col(k) % (1.0 - P.col(k));
    mat Xk  = X.each_col() % dkk;
    H.submat((k - 1) * r, (k - 1) * r, k * r - 1, k * r - 1) = X.t() * Xk;
  }

  // Off-diagonals: enumerate unordered (k, j) pairs as a flat index for OpenMP.
  const uword n_pairs = Km1 * (Km1 - 1) / 2;
#pragma omp parallel for schedule(dynamic)
  for (uword idx = 0; idx < n_pairs; ++idx) {
    // Recover (k, j) with 1 <= k < j <= Km1.
    uword k = 1, rem = idx;
    while (rem >= Km1 - k) { rem -= (Km1 - k); ++k; }
    uword j = k + 1 + rem;
    vec djk = -(wv % P.col(j) % P.col(k));
    mat Xjk = X.each_col() % djk;
    mat block = X.t() * Xjk;
    H.submat((j - 1) * r, (k - 1) * r, j * r - 1, k * r - 1) = block;
    H.submat((k - 1) * r, (j - 1) * r, k * r - 1, j * r - 1) = block.t();
  }
}

// Mark columns of X that are (numerically) constant — treated as intercepts.
uvec detect_intercepts(const mat& X) {
  const uword r = X.n_cols;
  uvec is_intercept(r, fill::zeros);
  for (uword j = 0; j < r; ++j) {
    if (std::abs(X.col(j).max() - X.col(j).min()) < kInterceptTol) is_intercept(j) = 1;
  }
  return is_intercept;
}

} // namespace

// [[Rcpp::export]]
List fast_multinom_fit_cpp(const arma::mat& X,
                           const arma::mat& Y,
                           const arma::vec& w,
                           int max_iter = 100,
                           double tol = 1e-8,
                           double ridge = 1e-6,
                           int nthreads = 0,
                           Rcpp::Nullable<Rcpp::NumericMatrix> B_init = R_NilValue,
                           double eta_clip = 30.0) {
#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads(nthreads);
#endif
  const uword n   = X.n_rows;
  const uword r   = X.n_cols;
  const uword K   = Y.n_cols;
  if (K < 2) stop("Multinom requires at least two components.");
  const uword Km1 = K - 1;
  const uword p   = r * Km1;
  if (w.n_elem != n) stop("weight length mismatch");

  // Warm start, accepted only if finite and within the clip range.
  mat B(r, Km1, fill::zeros);
  if (B_init.isNotNull()) {
    NumericMatrix Bi(B_init);
    if ((uword)Bi.nrow() == r && (uword)Bi.ncol() == Km1 && Bi.size() > 0) {
      mat Bcand = as<mat>(Bi);
      if (Bcand.is_finite() && Bcand.max() < eta_clip && Bcand.min() > -eta_clip) {
        B = Bcand;
      }
    }
  }

  const uvec is_intercept = detect_intercepts(X);
  const vec wv = w;

  mat P = compute_P(X, B, eta_clip);
  double cur_ll = weighted_loglik(P, Y, wv);
  bool converged = false;
  int iter = 0;

  mat H(p, p, fill::zeros);
  vec g(p, fill::zeros);

  for (iter = 0; iter < max_iter; ++iter) {
    // Gradient: g_k = X' (w * (Y_k - P_k)).
    g.zeros();
    for (uword k = 1; k < K; ++k) {
      g.subvec((k - 1) * r, k * r - 1) = X.t() * (wv % (Y.col(k) - P.col(k)));
    }

    // Hessian + adaptive Tikhonov ridge.
    build_hessian(X, P, wv, H);
    const double diag_mean = mean(H.diag());
    const double eff_ridge = std::max(ridge, ridge * diag_mean);
    for (uword k = 0; k < Km1; ++k) {
      for (uword j = 0; j < r; ++j) {
        const double add = is_intercept(j) ? kInterceptRidge * (diag_mean + 1.0)
                                           : eff_ridge;
        H(k * r + j, k * r + j) += add;
      }
    }

    // Newton step with pseudo-inverse fallback.
    vec step;
    bool ok = solve(step, H, g, solve_opts::likely_sympd + solve_opts::no_approx);
    if (!ok) {
      mat Hinv;
      if (!pinv(Hinv, H)) break;
      step = Hinv * g;
    }
    if (!step.is_finite()) break;

    const double snorm = norm(step, "inf");
    if (snorm > kMaxStepInf) step *= (kMaxStepInf / snorm);

    // Backtracking line search on the log-likelihood.
    double alpha = 1.0;
    bool improved = false;
    double new_ll = cur_ll;
    mat B_trial = B;
    for (int ls = 0; ls < kMaxLineSearch; ++ls) {
      mat trial = B + alpha * reshape(step, r, Km1);
      mat Pt    = compute_P(X, trial, eta_clip);
      double tll = weighted_loglik(Pt, Y, wv);
      if (std::isfinite(tll) && tll >= cur_ll - 1e-12) {
        B_trial = trial; new_ll = tll; P = Pt; improved = true; break;
      }
      alpha *= 0.5;
    }
    if (!improved) { converged = true; break; }
    B = B_trial;

    if (std::abs(new_ll - cur_ll) < tol * (std::abs(cur_ll) + tol)) {
      cur_ll = new_ll; converged = true; ++iter; break;
    }
    cur_ll = new_ll;
  }

  // Final fitted values without clipping, and matching Hessian.
  mat P_out = compute_P(X, B, datum::inf);
  build_hessian(X, P_out, wv, H);

  return List::create(
    _["coef"]      = B,
    _["fitted"]    = P_out,
    _["hessian"]   = H,
    _["loglik"]    = cur_ll,
    _["iter"]      = iter,
    _["converged"] = converged
  );
}