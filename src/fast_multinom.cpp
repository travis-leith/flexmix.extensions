// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::plugins(cpp14)]]

#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace arma;

// Numerically stable row-wise softmax (in place).
static inline void softmax_inplace(mat& Eta) {
  const uword n = Eta.n_rows;
#pragma omp parallel for schedule(static)
  for (uword i = 0; i < n; ++i) {
    double m = Eta.row(i).max();
    rowvec e = exp(Eta.row(i) - m);
    Eta.row(i) = e / accu(e);
  }
}

// Weighted log-likelihood given softmax probs P and target Y, case weights wv.
static inline double weighted_loglik(const mat& P, const mat& Y, const vec& wv) {
  const uword n = P.n_rows;
  const uword K = P.n_cols;
  double ll = 0.0;
#pragma omp parallel for reduction(+:ll) schedule(static)
  for (uword i = 0; i < n; ++i) {
    double s = 0.0;
    for (uword k = 0; k < K; ++k) {
      if (Y(i, k) > 0.0) s += Y(i, k) * std::log(std::max(P(i, k), 1e-300));
    }
    ll += wv(i) * s;
  }
  return ll;
}

// Compute P from B (with reference col 0), clamping linear predictors for stability.
static inline mat compute_P(const mat& X, const mat& B, double eta_clip = 30.0) {
  const uword n = X.n_rows;
  const uword K = B.n_cols + 1;
  mat Eta(n, K, fill::zeros);
  if (B.n_cols > 0) {
    Eta.cols(1, K - 1) = X * B;
    Eta.cols(1, K - 1) = clamp(Eta.cols(1, K - 1), -eta_clip, eta_clip);
  }
  mat P = Eta;
  softmax_inplace(P);
  return P;
}

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
  const uword n = X.n_rows;
  const uword r = X.n_cols;
  const uword K = Y.n_cols;
  if (K < 2) stop("Multinom requires at least two components.");
  const uword Km1 = K - 1;
  const uword p = r * Km1;
  
  // Warm start (or zeros)
  mat B(r, Km1, fill::zeros);
  if (B_init.isNotNull()) {
    NumericMatrix Bi(B_init);
    if ((uword)Bi.nrow() == r && (uword)Bi.ncol() == Km1 && Bi.size() > 0) {
      mat Bcandidate = as<mat>(Bi);
      // Only accept warm start if it's finite and not crazy
      if (Bcandidate.is_finite() && Bcandidate.max() < eta_clip && Bcandidate.min() > -eta_clip) {
        B = Bcandidate;
      }
    }
  }
  
  // Detect constant (intercept-like) columns -> don't penalize
  uvec is_intercept(r, fill::zeros);
  for (uword j = 0; j < r; ++j) {
    if (std::abs(X.col(j).max() - X.col(j).min()) < 1e-12) is_intercept(j) = 1;
  }
  
  vec wv = w;
  if (wv.n_elem != n) stop("weight length mismatch");
  
  mat P = compute_P(X, B, eta_clip);
  double cur_ll = weighted_loglik(P, Y, wv);
  double prev_ll = -datum::inf;
  bool converged = false;
  int iter = 0;
  
  mat H(p, p, fill::zeros);
  vec g(p, fill::zeros);
  
  for (iter = 0; iter < max_iter; ++iter) {
    // Gradient
    g.zeros();
    for (uword k = 1; k < K; ++k) {
      vec resid = wv % (Y.col(k) - P.col(k));
      g.subvec((k - 1) * r, k * r - 1) = X.t() * resid;
    }
    
    // Hessian (Fisher info)
    H.zeros();
    for (uword k = 1; k < K; ++k) {
      vec dkk = wv % P.col(k) % (1.0 - P.col(k));
      mat Xk = X.each_col() % dkk;
      H.submat((k - 1) * r, (k - 1) * r, k * r - 1, k * r - 1) = X.t() * Xk;
      for (uword j = k + 1; j < K; ++j) {
        vec djk = -(wv % P.col(j) % P.col(k));
        mat Xjk = X.each_col() % djk;
        mat block = X.t() * Xjk;
        H.submat((j - 1) * r, (k - 1) * r, j * r - 1, k * r - 1) = block;
        H.submat((k - 1) * r, (j - 1) * r, k * r - 1, j * r - 1) = block.t();
      }
    }
    // Tikhonov ridge on non-intercept rows (adaptive: scale with diag)
    double diag_mean = mean(H.diag());
    double effective_ridge = std::max(ridge, ridge * diag_mean);
    for (uword k = 0; k < Km1; ++k) {
      for (uword j = 0; j < r; ++j) {
        if (!is_intercept(j)) {
          H(k * r + j, k * r + j) += effective_ridge;
        }
      }
      // Always add tiny ridge to intercept too, to guarantee invertibility
      for (uword j = 0; j < r; ++j) {
        if (is_intercept(j)) H(k * r + j, k * r + j) += 1e-10 * (diag_mean + 1.0);
      }
    }
    
    // Solve H * step = g
    vec step;
    bool ok = solve(step, H, g, solve_opts::likely_sympd + solve_opts::no_approx);
    if (!ok) {
      // Try pseudo-inverse fallback silently
      mat Hinv;
      if (!pinv(Hinv, H)) break;
      step = Hinv * g;
    }
    if (!step.is_finite()) break;
    
    // Cap step norm to prevent runaway
    double step_norm = norm(step, "inf");
    double max_step = 5.0;
    if (step_norm > max_step) step *= (max_step / step_norm);
    
    // Backtracking line search
    mat B_trial = B;
    double new_ll = cur_ll;
    double alpha = 1.0;
    bool improved = false;
    for (int ls = 0; ls < 40; ++ls) {
      mat trial = B + alpha * reshape(step, r, Km1);
      mat Pt = compute_P(X, trial, eta_clip);
      double tll = weighted_loglik(Pt, Y, wv);
      if (std::isfinite(tll) && tll >= cur_ll - 1e-12) {
        B_trial = trial;
        new_ll = tll;
        P = Pt;
        improved = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!improved) {
      converged = true;  // can't improve -> we're at a (local) optimum
      break;
    }
    B = B_trial;
    
    if (std::abs(new_ll - cur_ll) < tol * (std::abs(cur_ll) + tol)) {
      cur_ll = new_ll;
      converged = true;
      ++iter;
      break;
    }
    prev_ll = cur_ll;
    cur_ll = new_ll;
  }
  
  // Final fitted values (no clipping in output)
  mat Eta(n, K, fill::zeros);
  if (B.n_cols > 0) Eta.cols(1, K - 1) = X * B;
  mat P_out = Eta;
  softmax_inplace(P_out);
  
  // Final Hessian for refit summary
  H.zeros();
  for (uword k = 1; k < K; ++k) {
    vec dkk = wv % P_out.col(k) % (1.0 - P_out.col(k));
    mat Xk = X.each_col() % dkk;
    H.submat((k - 1) * r, (k - 1) * r, k * r - 1, k * r - 1) = X.t() * Xk;
    for (uword j = k + 1; j < K; ++j) {
      vec djk = -(wv % P_out.col(j) % P_out.col(k));
      mat Xjk = X.each_col() % djk;
      mat block = X.t() * Xjk;
      H.submat((j - 1) * r, (k - 1) * r, j * r - 1, k * r - 1) = block;
      H.submat((k - 1) * r, (j - 1) * r, k * r - 1, j * r - 1) = block.t();
    }
  }
  
  return List::create(
    _["coef"]      = B,
    _["fitted"]    = P_out,
    _["hessian"]   = H,
    _["loglik"]    = cur_ll,
    _["iter"]      = iter,
    _["converged"] = converged
  );
}