# Kolmogorov-Smirnov test
#
# Continuous data. Compare cumulative distribution functions (CDFs).
#
# One-sample: empirical CDF F_n vs a fully specified continuous CDF F0.
#   H0: F = F0
# Two-sample: empirical CDF of sample X vs empirical CDF of sample Y.
#   H0: F_X = F_Y
#
# Test statistic (one-sample):
#   D = sup_x |F_n(x) - F0(x)|
# Under continuous F0, the null distribution of D does not depend on F0
# (distribution-free). Sensitive to any departure from F0 (location, scale,
# shape) — not only mean shifts.

set.seed(42)
alpha <- 0.05

# --- one-sample: do these draws look like N(0, 1)? ---
# H0: X_i iid ~ Normal(mean = 0, sd = 1)

x <- rnorm(n = 40, mean = 0, sd = 1)
x_sorted <- sort(x)
n <- length(x_sorted)
i <- seq_len(n)
F0 <- pnorm(x_sorted, mean = 0, sd = 1)

# Exact D on the order statistics (continuous F0):
#   D+ = max_i (i/n - F0(X_(i)))
#   D- = max_i (F0(X_(i)) - (i - 1)/n)
#   D  = max(D+, D-)
d_plus <- max(i / n - F0)
d_minus <- max(F0 - (i - 1) / n)
d_manual <- max(d_plus, d_minus)

cat("One-sample KS vs N(0,1)\n")
cat("D+ =", d_plus, " D- =", d_minus, " D =", d_manual, "\n")

ks_one <- ks.test(x, "pnorm", mean = 0, sd = 1)
print(ks_one)
cat("reject H0 at alpha =", alpha, "?", ks_one$p.value < alpha, "\n\n")

# Sanity check: same draws should reject a badly wrong F0 (e.g. N(2, 1)).
ks_wrong <- ks.test(x, "pnorm", mean = 2, sd = 1)
cat("Same sample vs N(2,1): D =", ks_wrong$statistic, " p =", ks_wrong$p.value, "\n")
cat("reject H0 at alpha =", alpha, "?", ks_wrong$p.value < alpha, "\n\n")

# --- two-sample: same continuous distribution? ---
# H0: F_x = F_y (both continuous). No need to name the common law.

y <- rnorm(n = 35, mean = 0.8, sd = 1)
ks_two <- ks.test(x, y)
cat("Two-sample KS (N(0,1) draws vs N(0.8,1) draws)\n")
print(ks_two)
cat("reject H0 at alpha =", alpha, "?", ks_two$p.value < alpha, "\n\n")

# Empirical CDFs: D is the largest vertical gap between the two step functions.
ecdf_x <- ecdf(x)
ecdf_y <- ecdf(y)
grid <- sort(c(x, y))
d_emp <- max(abs(ecdf_x(grid) - ecdf_y(grid)))
cat("Largest |F_n,x - F_n,y| on pooled sample points =", d_emp, "\n")
cat("(matches ks.test D up to how ties / evaluation points are handled)\n")
