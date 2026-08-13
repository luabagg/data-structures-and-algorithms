# Binomial test
# Notes: obsidian-statistics-vault → Statistical Inference/Binomial Test
#
# Binary outcomes (success / failure). Parameter of interest: p = P(success).
# Under H0: p = p0. X ~ Binomial(n, p0) under H0 (iid Bernoulli trials).

# --- small exact example: n = 10, x = 1 success, p0 = 0.3, H1: p < p0 ---

n <- 10
x <- 1
p0 <- 0.3
alpha <- 0.05

# PMF: P(X = k) = C(n, k) * p0^k * (1 - p0)^(n - k)
x0 <- dbinom(0, size = n, prob = p0)
x1 <- dbinom(1, size = n, prob = p0)
p_manual <- x0 + x1
p_cdf <- pbinom(x, size = n, prob = p0) # lower-tail p-value P(X <= x | H0)

cat("P(X=0) + P(X=1) =", p_manual, "\n")
cat("pbinom(1, 10, 0.3) =", p_cdf, "\n")
cat("reject H0 at alpha =", alpha, "?", p_cdf < alpha, "\n\n")

# Exact binomial test.
# conf.level is 1 - alpha (coverage), not alpha itself.
exact <- binom.test(
  x = x,
  n = n,
  p = p0,
  alternative = "less",
  conf.level = 1 - alpha
)
print(exact)

# --- large-sample normal approximation when n is large (rule of thumb n > 30,
# and preferably np0 and n(1-p0) not tiny) ---
# Z = (X - n*p0) / sqrt(n*p0*(1-p0))  ~  N(0,1) under H0
#
# Prefer binom.test (exact) when feasible; R still uses exact methods here.

large <- binom.test(x = 725, n = 1500, p = 0.5)
print(large)

# Optional: hand-rolled z for the large sample (two-sided illustration)
n2 <- 1500
x2 <- 725
p0_2 <- 0.5
z <- (x2 - n2 * p0_2) / sqrt(n2 * p0_2 * (1 - p0_2))
p_z_two_sided <- 2 * pnorm(-abs(z))
cat("\nLarge-sample Z =", z, "\n")
cat("Approx two-sided p-value =", p_z_two_sided, "\n")
