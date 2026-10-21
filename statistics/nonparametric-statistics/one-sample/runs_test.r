# Runs test (randomness / "teste de iterações")
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
cat("two-sided p-value (normal approximation) =", p_two, "\n")
cat("reject H0 at alpha =", alpha, "?", p_two < alpha, "\n\n")

# The study guide decides with a runs table, not with Z.
# Z needs large n1 and n2. Here both are 6, so use the exact null PMF.
#   P(R = 2k) = 2 * C(n1-1, k-1) * C(n2-1, k-1) / C(n, n1)
#   P(R = 2k+1) = [C(n1-1, k) * C(n2-1, k-1) + C(n1-1, k-1) * C(n2-1, k)] / C(n, n1)
# Equal tails: reject when R <= c_low or R >= c_high, with each tail <= alpha/2.
# Guide notation: do not reject when FI <= R <= FS.

runs_pmf <- function(n1, n2) {
  total <- n1 + n2
  denom <- choose(total, n1)
  r_values <- 2:total
  probs <- vapply(r_values, function(r) {
    if (r %% 2 == 0) {
      k <- r / 2
      2 * choose(n1 - 1, k - 1) * choose(n2 - 1, k - 1) / denom
    } else {
      k <- (r - 1) / 2
      (choose(n1 - 1, k) * choose(n2 - 1, k - 1) +
        choose(n1 - 1, k - 1) * choose(n2 - 1, k)) / denom
    }
  }, numeric(1))
  data.frame(r = r_values, probability = probs)
}

pmf <- runs_pmf(n1, n2)
pmf <- pmf[pmf$probability > 0, ]
cdf <- cumsum(pmf$probability)
c_low_hits <- pmf$r[cdf <= alpha / 2]
c_low <- if (length(c_low_hits) == 0) NA_integer_ else max(c_low_hits)
upper_tail <- rev(cumsum(rev(pmf$probability)))
c_high_hits <- pmf$r[upper_tail <= alpha / 2]
c_high <- if (length(c_high_hits) == 0) NA_integer_ else min(c_high_hits)
fi <- if (is.na(c_low)) min(pmf$r) else c_low + 1
fs <- if (is.na(c_high)) max(pmf$r) else c_high - 1
reject_exact <- r < fi || r > fs

cat("Exact runs test, equal tails at alpha =", alpha, "\n")
cat("reject if R <=", c_low, "or R >=", c_high, "\n")
cat("guide form: do not reject when", fi, "<= R <=", fs, "\n")
cat("R =", r, " reject H0?", reject_exact, "\n")
cat("P(R =", r, ") =", pmf$probability[pmf$r == r], "\n\n")

# --- package versions, if installed ---
# install.packages("tseries")    # runs.test on a two-level factor
# install.packages("randtests")  # runs.test on the numeric vector (median cutoff)

if (requireNamespace("tseries", quietly = TRUE)) {
  print(tseries::runs.test(as.factor(s)))
}

if (requireNamespace("randtests", quietly = TRUE)) {
  print(randtests::runs.test(x, threshold = med, alternative = "two.sided"))
}
