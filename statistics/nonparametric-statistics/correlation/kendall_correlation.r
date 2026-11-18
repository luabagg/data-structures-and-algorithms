# Kendall rank correlation (tau)
# Study guide: correlação não paramétrica
#
# Two paired numeric or ordinal variables. Better than Spearman for
# small samples or many ties.
#
# H0: tau = 0 (no association).
# H1: tau != 0.
#
# 1. For every pair of observations (i, j), look at the signs of
#    x_i - x_j and y_i - y_j.
# 2. Same sign = concordant (C). Opposite sign = discordant (D).
# 3. Without ties: tau = (C - D) / (n * (n - 1) / 2).
# Check: C + D = n * (n - 1) / 2 when there are no ties.
#
# R reports T = C (concordant pairs) and tau. The p-value is exact for
# small n without ties.

kendall_by_hand <- function(x, y) {
  n <- length(x)
  s <- sign(outer(x, x, "-")) * sign(outer(y, y, "-"))
  concordant <- sum(s[upper.tri(s)] > 0)
  discordant <- sum(s[upper.tri(s)] < 0)
  cat("C =", concordant, " D =", discordant, " C + D =", concordant + discordant,
      " n(n-1)/2 =", n * (n - 1) / 2, "\n")
  cat("tau by formula =", (concordant - discordant) / (n * (n - 1) / 2), "\n")
}

# Example 1: four paired values.
x <- c(7, 9, 5, 4)
y <- c(8, 4, 9, 6)
kendall_by_hand(x, y)
print(cor.test(x, y, method = "kendall"))

# Example 2: two critics ranking ten items.
critico_a <- c(6, 4, 9, 1, 2, 7, 3, 8, 5, 10)
critico_b <- c(5, 6, 10, 2, 3, 8, 1, 7, 4, 9)
cat("\n")
kendall_by_hand(critico_a, critico_b)
print(cor.test(critico_a, critico_b, method = "kendall"))
