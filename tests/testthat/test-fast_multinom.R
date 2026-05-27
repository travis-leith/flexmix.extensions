library(conflicted)
library(tidyverse)
library(flexmix)

rm(list = ls())

Sys.setenv(PKG_CXXFLAGS = "-fopenmp -O3")
Sys.setenv(PKG_LIBS = "-fopenmp")
Rcpp::sourceCpp("fast_multinom.cpp")
source("fast_multinom.R")


data <-
  tibble(
    zp_1 = runif(n = 10000),
    zp_2 = runif(n = 10000),
  ) |>
  mutate(
    true_latent_class = case_when(
      0.5 * zp_1 - zp_2 + 0.5 < 0 ~ "A",
      -2 * zp_1 - zp_2 + 1.5 > 0 ~ "B",
      TRUE ~ "C"
    ),
    x1 = rnorm(n = 10000),
    x2 = rnorm(n = 10000),
    y = case_when(
      true_latent_class == "A" ~ 1 + 0.5 * x1 + 0.2 * x2 + rnorm(10000),
      true_latent_class == "B" ~ -1 + 0.3 * x1 - 0.4 * x2 + rnorm(10000),
      TRUE ~ 0.1 * x1 + 0.5 * x2 + rnorm(10000)
    )
  )


data |>
  ggplot() +
  aes(x = zp_1, y = zp_2, color = true_latent_class) +
  geom_point() +
  theme_minimal() +
  labs(title = "True Latent Classes")


m_flexmix <-
  flexmix(
    y ~ x1 + x2,
    data = data,
    concomitant = FastFLXPmultinom(~ zp_1 + zp_2),
    k = 3
  )

data |>
  mutate(
    prior_class = prior(m_flexmix, newdata = data) |> apply(1, which.max)
  ) |>
  ggplot() +
  aes(x = zp_1, y = zp_2, color = factor(prior_class)) +
  geom_point() +
  theme_minimal() +
  labs(title = "Prior Predicted Latent Classes", color = "Predicted Class")

m_refit <- refit(m_flexmix)
summary(m_refit)
