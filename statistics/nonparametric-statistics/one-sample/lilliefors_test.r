# Lilliefors test (normality)
# Package: nortest::lillie.test
#
# Continuous data. H0: the sample comes from some Normal(mu, sigma^2).
#
# ks.test needs a fully specified F0: you pass pnorm plus the two values
# (mean, sd). That is simple KS, not Lilliefors.
#
# If mu and sigma are unknown, estimate them from the same sample
# (mean, sd, var). Then D is still the KS distance to N(bar(x), s),
# but the usual ks.test p-value is wrong (too large). Lilliefors
# recalibrates that p-value. nortest::lillie.test does this; n >= 5.

# --- theoretical normal CDF at a few points: pnorm(x, mean, sd) ---
# Phi(x) = P(Z <= x) for Z ~ N(mean, sd^2). The two extra arguments
# are exactly the two parameters KS will use below.

mu0 <- 70
sigma0 <- 8
cat("Phi at 60, 70, 80 for N(70, 8^2):\n")
print(pnorm(c(60, 70, 80), mean = mu0, sd = sigma0))

# --- one-sample KS vs a *named* normal (parameters given, not estimated) ---
# H0: weight ~ N(70, 8^2)

weight <- c(58, 62, 65, 67, 70, 71, 73, 75, 78, 82)
ks.test(weight, pnorm, mu0, sigma0)

# --- same idea, but mu and sigma unknown: Lilliefors via nortest ---
# install.packages("nortest")

# library(nortest)
data <- c(58, 62, 65, 67, 70, 71, 73, 75, 78, 82)

mu_hat <- mean(data)
sigma_hat <- sd(data)
cat("mean =", mu_hat, " sd =", sigma_hat, " var =", var(data), "\n")

# Fitted normal used by Lilliefors: N(mean(data), sd(data)^2).
# Full KS distance on the order statistics:
#   D+ = max (i/n - F(x_(i)))
#   D- = max (F(x_(i)) - (i - 1)/n)
#   D  = max(D+, D-)
# The study guide also writes max |i/n - F(x_(i))|. That piece is not
# always equal to D. ks.test and the Lilliefors table use D.
data_sorted <- sort(data)
n_obs <- length(data_sorted)
i_obs <- seq_len(n_obs)
F_hat <- pnorm(data_sorted, mean = mu_hat, sd = sigma_hat)
d_plus <- max(i_obs / n_obs - F_hat)
d_minus <- max(F_hat - (i_obs - 1) / n_obs)
d_lillie <- max(d_plus, d_minus)
d_guide <- max(abs(i_obs / n_obs - F_hat))

cat("D+ =", d_plus, " D- =", d_minus, " D =", d_lillie, "\n")
cat("max |i/n - F| from the study guide =", d_guide, "\n")

# ks.test here has the same D and a p-value that is too large.
# Do not use that p-value. The parameters were estimated from this sample.
cat("\nks.test with estimated mean and sd (invalid p-value):\n")
print(ks.test(data, pnorm, mu_hat, sigma_hat))
cat("Decision needs the Lilliefors table, or nortest::lillie.test.\n\n")

if (requireNamespace("nortest", quietly = TRUE)) {
  print(nortest::lillie.test(data))
} else {
  cat("nortest not installed. In R: install.packages(\"nortest\")\n")
  cat("then: library(nortest); lillie.test(data)\n")
}
