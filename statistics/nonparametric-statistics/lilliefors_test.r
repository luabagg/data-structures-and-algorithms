# Lilliefors test (normality)
# Notes: obsidian-statistics-vault → Statistical Inference/Lilliefors Test
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
pnorm(c(60, 70, 80), mean = mu0, sd = sigma0)

# --- one-sample KS vs a *named* normal (parameters given, not estimated) ---
# H0: weight ~ N(70, 8^2)

weight <- c(58, 62, 65, 67, 70, 71, 73, 75, 78, 82)
ks.test(weight, pnorm, mu0, sigma0)

# --- same idea, but mu and sigma unknown: Lilliefors via nortest ---
# install.packages("nortest")

# library(nortest)
data <- c(58, 62, 65, 67, 70, 71, 73, 75, 78, 82)

mean(data)
sd(data)
var(data)

# Fitted normal used by Lilliefors: N(mean(data), sd(data)^2).
# Compare: ks.test(data, pnorm, mean(data), sd(data)) has the same D
# and a larger (invalid) p-value.

if (requireNamespace("nortest", quietly = TRUE)) {
  print(nortest::lillie.test(data))
} else {
  cat("nortest not installed. In R: install.packages(\"nortest\")\n")
  cat("then: library(nortest); lillie.test(data)\n")
}
