# Spearman rank correlation
# Study guide: correlação não paramétrica
#
# Two paired numeric or ordinal variables. Measures monotonic association.
#
# H0: rho = 0 (no monotonic association).
# H1: rho != 0.
#
# 1. Rank x and rank y separately. Ties get the average rank.
# 2. di = rank(x) - rank(y).
# 3. Without ties: rho = 1 - 6 * sum(di^2) / (n * (n^2 - 1)).
#    With ties, rho is the Pearson correlation of the ranks.
# Check: rho is always between -1 and 1.
#
# R reports S = sum(di^2) and rho. The p-value is exact for small n
# without ties. With ties R falls back to an approximation and warns.

spearman_by_hand <- function(x, y) {
  d <- rank(x) - rank(y)
  n <- length(x)
  cat("di =", paste(d, collapse = " "), "\n")
  cat("sum(di^2) =", sum(d^2), "\n")
  cat("rho by formula =", 1 - 6 * sum(d^2) / (n * (n^2 - 1)), "\n")
  cat("rho = Pearson of ranks =", cor(rank(x), rank(y)), "\n")
}

# Example 1: two judges scoring the same ten items (no ties).
g1 <- c(8.5, 5.8, 8.7, 5.7, 8.1, 9.3, 6.3, 7.7, 5.3, 6.7)
g2 <- c(8.8, 7.3, 9.2, 6.1, 8.5, 8.3, 7.1, 7.2, 6.2, 7.5)
spearman_by_hand(g1, g2)
print(cor.test(g1, g2, method = "spearman"))

# Example 2: grades and distance (no ties).
notas <- c(12, 7, 5, 19, 17, 14, 9, 18, 3, 8, 15, 4)
distancia <- c(75, 165, 300, 15, 180, 240, 120, 60, 230, 200, 130, 140)
cat("\n")
spearman_by_hand(notas, distancia)
print(cor.test(notas, distancia, method = "spearman"))
