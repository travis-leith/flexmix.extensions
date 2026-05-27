# Faster S4 method for FLXgradlogLikfun on the multinomial concomitant.
# Source AFTER loading flexmix and AFTER sourceCpp("fast_concomitant_grad.cpp").
#
# This replaces the FLXP method's apply(X, 2, "*", p) loop with a single
# parallel C++ pass. Dispatch on signature("FLXPmultinom") shadows the
# inherited FLXP method for multinomial concomitants only.

stopifnot(requireNamespace("flexmix", quietly = TRUE))
stopifnot(exists("cpp_multinom_scores"))

setMethod(
  "FLXgradlogLikfun",
  signature(object = "FLXPmultinom"),
  function(object, fitted, weights, ...) {
    cpp_multinom_scores(object@x, as.matrix(fitted), as.matrix(weights))
  }
)
