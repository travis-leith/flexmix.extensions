test_that("FLXPcategorical rejects numeric regressors", {
  set.seed(1)
  n <- 200
  df <- data.frame(y = rnorm(n), z = rnorm(n))
  expect_error(
    flexmix::flexmix(
      y ~ 1,
      data = df,
      k = 2,
      concomitant = FLXPcategorical(~z)
    ),
    "categorical regressors only"
  )
})

test_that("priors equal hand-computed per-cell mean responsibilities", {
  set.seed(2)
  n <- 600
  g <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
  y <- rnorm(n, mean = c(a = -3, b = 0, c = 3)[g])
  df <- data.frame(y = y, g = g)

  m <- flexmix::flexmix(
    y ~ 1,
    data = df,
    k = 3,
    concomitant = FLXPcategorical(~g)
  )

  P <- flexmix::prior(m, newdata = df) # per-obs priors
  post <- m@posterior$scaled

  # Reconstruct: prior for an obs must equal mean posterior in its cell.
  expected <- apply(post, 2, function(col) tapply(col, g, mean)[g])
  expected <- expected / rowSums(expected)
  expect_equal(unname(P), unname(expected), tolerance = 1e-8)
})

test_that("rowsum cell ordering does not corrupt priors (labels unsorted)", {
  set.seed(3)
  n <- 600
  # Levels whose first-appearance order differs from sorted order.
  g <- factor(
    sample(c("zzz", "aaa", "mmm"), n, replace = TRUE),
    levels = c("zzz", "mmm", "aaa")
  )
  y <- rnorm(n, mean = c(zzz = 4, mmm = 0, aaa = -4)[as.character(g)])
  df <- data.frame(y = y, g = g)

  m <- flexmix::flexmix(
    y ~ 1,
    data = df,
    k = 3,
    concomitant = FLXPcategorical(~g)
  )
  P <- flexmix::prior(m, newdata = df)
  post <- m@posterior$scaled
  expected <- apply(post, 2, function(col) {
    tapply(col, g, mean)[as.character(g)]
  })
  expected <- expected / rowSums(expected)
  expect_equal(unname(P), unname(expected), tolerance = 1e-8)
})

test_that("clustering recovers structure better than constant prior", {
  set.seed(4)
  n <- 800
  g <- factor(sample(c("a", "b"), n, replace = TRUE))
  truth <- ifelse(
    g == "a",
    sample(1:2, n, TRUE, prob = c(0.9, 0.1)),
    sample(1:2, n, TRUE, prob = c(0.1, 0.9))
  )
  y <- rnorm(n, mean = c(-2, 2)[truth])
  df <- data.frame(y = y, g = g)

  m_cat <- flexmix::flexmix(
    y ~ 1,
    data = df,
    k = 2,
    concomitant = FLXPcategorical(~g)
  )
  m_con <- flexmix::flexmix(
    y ~ 1,
    data = df,
    k = 2,
    concomitant = flexmix::FLXPconstant()
  )
  expect_gte(logLik(m_cat), logLik(m_con))
})

test_that("refit(method = 'optim') is refused; 'mstep' runs", {
  set.seed(5)
  n <- 300
  g <- factor(sample(c("a", "b"), n, replace = TRUE))
  y <- rnorm(n, mean = ifelse(g == "a", -2, 2))
  df <- data.frame(y = y, g = g)
  m <- flexmix::flexmix(
    y ~ 1,
    data = df,
    k = 2,
    concomitant = FLXPcategorical(~g)
  )

  expect_false(existGradient(m@concomitant))
  # optim path produces no concomitant gradient -> NULL gradient function.
  expect_null(flexmix:::FLXgradlogLikfun(m))

  rf <- flexmix::refit(m, method = "mstep")
  expect_s4_class(rf, "FLXRmstep")
})

test_that("simulate / rflexmix produce cell-dependent class proportions", {
  set.seed(6)
  n <- 1000
  g <- factor(sample(c("a", "b"), n, replace = TRUE))
  truth <- ifelse(
    g == "a",
    sample(1:2, n, TRUE, prob = c(0.9, 0.1)),
    sample(1:2, n, TRUE, prob = c(0.1, 0.9))
  )
  y <- rnorm(n, mean = c(-3, 3)[truth])
  df <- data.frame(y = y, g = g)
  m <- flexmix::flexmix(
    y ~ 1,
    data = df,
    k = 2,
    concomitant = FLXPcategorical(~g)
  )

  set.seed(7)
  sim <- flexmix::rflexmix(m)
  # Mean simulated y should differ by cell, reflecting cell-specific priors.
  by_cell <- tapply(unlist(sim$y), df$g, mean)
  expect_gt(abs(by_cell["a"] - by_cell["b"]), 1)
})

test_that("relabel preserves cell lookup and permutes priors", {
  set.seed(8)
  n <- 400
  g <- factor(sample(c("a", "b"), n, replace = TRUE))
  y <- rnorm(n, mean = ifelse(g == "a", -2, 2))
  df <- data.frame(y = y, g = g)
  m <- flexmix::flexmix(
    y ~ 1,
    data = df,
    k = 2,
    concomitant = FLXPcategorical(~g)
  )

  P <- flexmix::prior(m, newdata = df)
  m2 <- flexmix::relabel(m, by = 2L:1L)
  P2 <- flexmix::prior(m2, newdata = df)
  # Relabelling swaps the columns but the cell->prior map must survive.
  expect_equal(unname(P[, 2:1]), unname(P2), tolerance = 1e-8)
  expect_false(is.null(attr(m2@concomitant@coef, "cell")))
})
