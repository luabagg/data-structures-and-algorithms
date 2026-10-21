# Mann-Whitney U test
# Study guide: duas amostras independentes
#
# Two independent groups. The outcome is ordinal or numeric.
# The test uses ranks. It does not assume normality.
#
# H0: the groups have the same distribution.
# H1: the groups differ.
#
# 1. Pool the values and rank them from 1 to n1 + n2.
#    Ties get the average of the ranks they occupy.
# 2. R1 = sum of ranks in group 1. R2 = sum of ranks in group 2.
# 3. U1 = n1*n2 + n1*(n1+1)/2 - R1
#    U2 = n1*n2 + n2*(n2+1)/2 - R2
#    Ucal = min(U1, U2)
# Check: U1 + U2 = n1*n2.
# Table rule: reject H0 when Ucal <= Ucrit.
#
# R's wilcox.test reports W = R1 - n1*(n1+1)/2.
# That W is not Ucal. Course U1 = n1*n2 - W, and Ucal = min(W, n1*n2 - W).
#
# Paired data do not belong here. Use the sign test or Wilcoxon signed-rank.

alpha <- 0.05

g1 <- c(2, 5, 8)
g2 <- c(3, 6, 9)
n1 <- length(g1)
n2 <- length(g2)

# rank() keeps the original order and averages ties.
ranks <- rank(c(g1, g2))
r1 <- sum(ranks[seq_len(n1)])
r2 <- sum(ranks[(n1 + 1):(n1 + n2)])
u1 <- n1 * n2 + n1 * (n1 + 1) / 2 - r1
u2 <- n1 * n2 + n2 * (n2 + 1) / 2 - r2
u_cal <- min(u1, u2)

cat("Study guide: G1 = 2, 5, 8 and G2 = 3, 6, 9\n")
cat("ranks =", paste(ranks, collapse = " "), "\n")
cat("R1 =", r1, " R2 =", r2, "\n")
cat("U1 =", u1, " U2 =", u2, " Ucal =", u_cal, "\n")
cat("U1 + U2 =", u1 + u2, " n1*n2 =", n1 * n2, "\n")
stopifnot(abs((u1 + u2) - n1 * n2) < 1e-8)

# Exact null distribution: every choice of n1 ranks out of n1+n2
# is equally likely when there are no ties.
rank_sets <- combn(n1 + n2, n1)
u1_null <- apply(rank_sets, 2, function(idx) {
  n1 * n2 + n1 * (n1 + 1) / 2 - sum(idx)
})
u_null <- pmin(u1_null, n1 * n2 - u1_null)
p_exact <- mean(u_null <= u_cal)
cat("Exact two-sided p-value =", p_exact, "\n")
cat("reject H0 at alpha =", alpha, "?", p_exact < alpha, "\n")
cat(
  "Smallest possible Ucal is 0. Its two-sided probability is",
  mean(u_null <= 0), "\n"
)
cat("At alpha = 0.05 this n1 and n2 have an empty rejection region.\n\n")

wt <- wilcox.test(g1, g2, exact = TRUE, alternative = "two.sided")
print(wt)
w_r <- unname(wt$statistic)
cat("\nR statistic W =", w_r, "\n")
cat("W = R1 - n1*(n1+1)/2 =", r1 - n1 * (n1 + 1) / 2, "\n")
cat("min(W, n1*n2 - W) =", min(w_r, n1 * n2 - w_r), "\n")
