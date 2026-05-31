#' Categorical-only concomitant model for flexmix
#'
#' Concomitant model driver that estimates component priors as
#' the mean posterior responsibility within each *cell* of a fully categorical
#' design. A cell is one distinct combination of the categorical regressor
#' values. This is the non-parametric analogue of [flexmix::FLXPmultinom()]:
#' instead of a softmax of a linear predictor, the prior for an observation is
#' the average responsibility among observations sharing its covariate cell.
#'
#' Because the model has no coefficient vector and no smooth likelihood, it
#' cannot participate in gradient-based inference. `existGradient()` returns
#' `FALSE`, so `refit(method = "optim")` is correctly refused. EM fitting,
#' prediction, simulation (`rflexmix`/`simulate`), relabelling, and
#' `refit(method = "mstep")` are all supported.
#'
#' All regressors must be categorical. Numeric predictors trigger a clear
#' error; convert them to factors or use [flexmix::FLXPmultinom()] instead.
#'
#' @param formula A one-sided formula of categorical concomitant regressors.
#'   Defaults to `~1` (a single cell, equivalent to a constant prior).
#'
#' @return An object of class `FLXPcategorical`, to be passed as the
#'   `concomitant` argument of [flexmix::flexmix()].
#'
#' @examples
#' set.seed(1)
#' n <- 300
#' g <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
#' y <- rnorm(n, mean = c(a = -2, b = 0, c = 2)[g])
#' df <- data.frame(y = y, g = g)
#' flexmix::flexmix(
#'   y ~ 1,
#'   data = df, k = 3,
#'   concomitant = FLXPcategorical(~g)
#' )
#'
#' @seealso [flexmix::FLXPmultinom()]
#' @import flexmix
#' @import methods
#' @export
FLXPcategorical <- function(formula = ~1) {
  z <- new("FLXPcategorical", name = "FLXPcategorical", formula = formula)

  ## fit(): called every EM iteration. Returns one prior row PER OBSERVATION.
  z@fit <- function(x, y, w, ...) {
    if (missing(w) || is.null(w)) {
      w <- rep(1, nrow(y))
    }
    cell <- .flxp_cat_cells(x)
    num <- rowsum(w * y, cell)
    den <- as.vector(rowsum(w, cell))
    avg <- num / den
    avg <- avg / rowSums(avg)
    ## rowsum() labels its rows by group; index by those labels, not by
    ## levels(), so we never depend on the two orderings coinciding.
    avg[as.character(cell), , drop = FALSE]
  }

  ## refit(): used by FLXfillConcomitant to populate @coef after fitting.
  z@refit <- function(x, y, w, ...) {
    if (missing(w) || is.null(w)) {
      w <- rep(1, nrow(y))
    }
    cell <- .flxp_cat_cells(x)
    num <- rowsum(w * y, cell)
    den <- as.vector(rowsum(w, cell))
    avg <- num / den
    avg <- avg / rowSums(avg)
    ## rowsum() already names rows by cell label; keep that as the
    ## authoritative cell->row map. (Do NOT assign levels(cell) here -- that
    ## silently assumes levels() and rowsum()'s ordering agree.)
    attr(avg, "cell") <- as.character(cell)
    avg
  }
  z
}

## ---- S4 class ------------------------------------------------------------
## We subclass FLXP (the base concomitant class) rather than declaring
## ourselves FLXPmultinom: declaring FLXPmultinom would (incorrectly) opt us
## into softmax priors and multinom-shaped refit/gradient code.
## FLXP is made visible by the package-level `@import flexmix` above.
#' @noRd
setClass("FLXPcategorical", contains = "FLXP")

## Validate categorical-only design and collapse each row to a cell label.
## flexmix hands the driver a numeric model.matrix, so categorical inputs
## arrive as intercept + 0/1 indicator columns. Any other column is numeric.
#' @noRd
.flxp_cat_cells <- function(x) {
  is_intercept <- apply(x, 2, function(col) all(col == 1))
  is_indicator <- apply(x, 2, function(col) all(col %in% c(0, 1)))
  bad <- !(is_intercept | is_indicator)
  if (any(bad)) {
    stop(
      "FLXPcategorical() supports categorical regressors only.\n",
      "  The following design columns look numeric (not 0/1 indicators): ",
      paste(colnames(x)[bad], collapse = ", "),
      ".\n",
      "  Convert numeric predictors to factors, or use FLXPmultinom() instead.",
      call. = FALSE
    )
  }
  ind <- x[, !is_intercept, drop = FALSE]
  if (ncol(ind) == 0) {
    factor(rep("(all)", nrow(x)))
  } else {
    factor(apply(ind, 1, paste, collapse = ""))
  }
}

## ---- Resolve Unexported Generics locally --------------------------------
## By referencing unexported generics and setting them to local variables,
## we can declare S4 methods with plain strings, avoiding both load-time
## missing generic crashes and roxygen2 parsing errors.
getPriors <- flexmix:::getPriors
determinePrior <- flexmix:::determinePrior
dorelabel <- flexmix:::dorelabel
refit_mstep <- flexmix:::refit_mstep

## ---- Priors from stored state -------------------------------------------
## getPriors() turns stored @coef into a per-observation prior matrix inside
## the log-likelihood path. The base FLXP method does `x %*% coef` (linear);
## we override it to do a cell lookup.
#' @noRd
setMethod(
  "getPriors",
  signature(object = "FLXPcategorical"),
  function(object, group, groupfirst) {
    cell <- attr(object@coef, "cell")
    priors <- object@coef[cell, , drop = FALSE]
    flexmix:::ungroupPriors(priors, group, groupfirst)
  }
)

## ---- Priors for simulation ----------------------------------------------
## determinePrior() drives rflexmix()/simulate().
#' @noRd
setMethod(
  "determinePrior",
  signature(concomitant = "FLXPcategorical"),
  function(prior, concomitant, group) {
    cell <- attr(concomitant@coef, "cell")
    concomitant@coef[cell, , drop = FALSE]
  }
)

## ---- Disable the gradient / optim-refit path ----------------------------
## No coefficient vector to differentiate -> refit(method = "optim") refuses.
#' @noRd
setMethod(
  "existGradient",
  signature(object = "FLXPcategorical"),
  function(object) FALSE
)

## ---- Relabelling --------------------------------------------------------
## The base FLXP dorelabel permutes @coef columns but drops attributes,
## which would break the cell->row lookup in getPriors. Preserve it.
#' @noRd
setMethod(
  "dorelabel",
  signature(object = "FLXPcategorical", perm = "vector"),
  function(object, perm, ...) {
    cell <- attr(object@coef, "cell")
    object@coef <- object@coef[, perm, drop = FALSE]
    colnames(object@coef) <- sapply(
      seq_len(ncol(object@coef)),
      function(k) gsub("[0-9]+", k, colnames(object@coef)[k])
    )
    attr(object@coef, "cell") <- cell
    object
  }
)

## ---- m-step refit -------------------------------------------------------
## refit(method = "mstep") -> refit_mstep() -> a small S4 object.
#' @noRd
setClass("FLXRcategorical", representation(table = "matrix"))

#' @noRd
setMethod(
  "refit_mstep",
  signature(object = "FLXPcategorical", newdata = "missing"),
  function(object, newdata, posterior, group, w, ...) {
    groupfirst <- if (length(group)) {
      flexmix:::groupFirst(group)
    } else {
      rep(TRUE, nrow(posterior))
    }
    tab <- object@refit(object@x, posterior[groupfirst, , drop = FALSE], w)
    attr(tab, "cell") <- NULL
    colnames(tab) <- paste("Comp", seq_len(ncol(tab)), sep = ".")
    new("FLXRcategorical", table = tab)
  }
)
