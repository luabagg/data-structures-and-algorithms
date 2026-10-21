# Sign test
# Study guide: duas amostras relacionadas
#
# Paired numeric or ordinal measurements.
# The test keeps only the sign of each difference. It ignores the size.
#
# H0: the two treatments do not differ (median of the differences is 0).
# H1: the treatments differ.
#
# di = first - second. Keep this order for every pair.
# di > 0 gives +. di < 0 gives -. di = 0 is dropped.
# After the zeros are dropped, n = n+ + n-.
# Under H0, X = min(n+, n-) follows Binomial(n, 0.5).
# Two-sided p-value = 2 * P(X <= xmin).
# Reject H0 when p <= alpha.
#
# If n > 25 the course uses a normal approximation, with x = min(n+, n-):
#   Z = (2x - n) / sqrt(n)
#   two-sided p-value = 2 * P(Z <= z)
# That Z is the usual normal approximation to Binomial(n, 0.5).
# The class example below has n = 8, so the decision uses the binomial.

# Class example. Eight workers, pieces per day, alpha = 0.10.
# The Wilcoxon script uses the same pairs and also uses the size of di.
old <- c(18, 15, 19, 23, 12, 16, 18, 17)
new <- c(24, 14, 22, 28, 16, 20, 20, 18)
alpha <- 0.10

diff <- old - new
n_plus <- sum(diff > 0)
n_minus <- sum(diff < 0)
n <- n_plus + n_minus
x_min <- min(n_plus, n_minus)
p_manual <- 2 * pbinom(x_min, size = n, prob = 0.5)

cat("differences (old - new) =", paste(diff, collapse = " "), "\n")
cat("n+ =", n_plus, " n- =", n_minus, " n =", n, " x =", x_min, "\n")
cat("two-sided p-value =", p_manual, "\n")
cat("reject H0 at alpha =", alpha, "?", p_manual <= alpha, "\n")
cat("The same p-value does not reject at alpha = 0.05.\n\n")

exact <- binom.test(x_min, n, p = 0.5, alternative = "two.sided")
print(exact)

# Separate illustration of the n > 25 formula. These counts are not
# the worker data above.
n_large <- 40
x_large <- 12
z_large <- (2 * x_large - n_large) / sqrt(n_large)
p_large <- 2 * pnorm(z_large)
cat("\nLarge-sample illustration: n =", n_large, " x =", x_large, "\n")
cat("Z =", z_large, " two-sided p-value =", p_large, "\n")
cat("reject H0 at alpha = 0.05?", p_large <= 0.05, "\n")
