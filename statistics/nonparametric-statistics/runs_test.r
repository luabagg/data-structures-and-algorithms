# Runs test (randomness / "teste de iterações")
# Notes: obsidian-statistics-vault → Statistical Inference/Runs Test
# Common R: tseries::runs.test or randtests::runs.test
#
# Sequence data. H0: the order is random (independent of position).
# This is *not* a test of the marginal distribution (unlike KS / Lilliefors).
#
# A run (iteração) is a maximal streak of the same symbol.
#   + + + - - + - - -    has 4 runs.
#
# For a numeric sample, recode vs the median:
#   + if x_i > median,  - if x_i < median,  drop ties.
# Let n1 = # of +, n2 = # of -, n = n1 + n2, R = number of runs.
#
# Under H0 (large n1, n2):
#   E[R] = 2*n1*n2 / n + 1
#   Var[R] = 2*n1*n2*(2*n1*n2 - n1 - n2) / (n^2 * (n - 1))
#   Z = (R - E[R]) / sqrt(Var[R])  ~  N(0, 1)
# Too few runs: clustering. Too many runs: oscillation.

alpha <- 0.05

x <- c(12, 15, 14, 10, 18, 20, 9, 11, 16, 13, 19, 8)

med <- median(x)
# drop ties at the median so every leftover point is + or -
keep <- x != med
s <- ifelse(x[keep] > med, "+", "-")

n1 <- sum(s == "+")
n2 <- sum(s == "-")
n <- n1 + n2

# R = 1 + number of times the symbol changes
r <- 1 + sum(s[-1] != s[-length(s)])

e_r <- 2 * n1 * n2 / n + 1
var_r <- 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (n^2 * (n - 1))
z <- (r - e_r) / sqrt(var_r)
p_two <- 2 * pnorm(-abs(z))

cat("median =", med, "\n")
cat("signs =", paste(s, collapse = " "), "\n")
cat("n1 =", n1, " n2 =", n2, " R =", r, "\n")
cat("E[R] =", e_r, " Var[R] =", var_r, " Z =", z, "\n")
cat("two-sided p-value =", p_two, "\n")
cat("reject H0 at alpha =", alpha, "?", p_two < alpha, "\n\n")

# --- package versions, if installed ---
# install.packages("tseries")    # runs.test on a two-level factor
# install.packages("randtests")  # runs.test on the numeric vector (median cutoff)

if (requireNamespace("tseries", quietly = TRUE)) {
  print(tseries::runs.test(as.factor(s)))
}

if (requireNamespace("randtests", quietly = TRUE)) {
  print(randtests::runs.test(x, threshold = med, alternative = "two.sided"))
}
