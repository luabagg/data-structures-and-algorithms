# Wilcoxon signed-rank test
# Study guide: duas amostras relacionadas
#
# Paired numeric or ordinal measurements.
# The sign test uses only the direction of di. This test also uses the
# rank of |di|.
#
# H0: the two treatments do not differ.
# H1: the treatments differ.
#
# 1. di = first - second. Drop di = 0.
# 2. Rank the absolute differences. Ties get the average rank.
# 3. Put the original sign back on each rank.
# 4. T+ = sum of the positive ranks. T- = sum of the absolute negative ranks.
#    T = min(T+, T-).
# Check: T+ + T- = n*(n+1)/2.
# If n <= 25, reject H0 when T <= T_table.
# The course solution for this example quotes T_table = 4 when n = 8
# and alpha = 0.05 (two-sided).
#
# The problem statement asks for alpha = 0.10. The written solution
# compares T with the alpha = 0.05 critical value. T = 1.5 is below 4,
# so both levels reject H0. The exact p-value below is the check.
#
# If n > 25 the course writes
#   Z = (T - n*(n+1)/4) / sqrt(n*(n+1)*(2*n+1)/24)
# The mean n*(n+1)/4 is the mean of T+, not the mean of min(T+, T-).
# Put T+ in that formula. For a two-sided test, double the tail.
# Do not put min(T+, T-) into Z and then double the tail.
# R reports V = T+. R does not report T.

# Same workers as sign_test.r. Differences keep old - new.
old <- c(18, 15, 19, 23, 12, 16, 18, 17)
new <- c(24, 14, 22, 28, 16, 20, 20, 18)
alpha_problem <- 0.10
alpha_solution <- 0.05
t_table_solution <- 4

diff <- old - new
d <- diff[diff != 0]
abs_d <- abs(d)
ranks <- rank(abs_d)
signed_ranks <- ranks * sign(d)
n <- length(d)
t_plus <- sum(signed_ranks[signed_ranks > 0])
t_minus <- sum(abs(signed_ranks[signed_ranks < 0]))
t_stat <- min(t_plus, t_minus)
rank_total <- n * (n + 1) / 2

cat("di =", paste(d, collapse = " "), "\n")
cat("|di| ranks =", paste(ranks, collapse = " "), "\n")
cat("signed ranks =", paste(signed_ranks, collapse = " "), "\n")
cat("T+ =", t_plus, " T- =", t_minus, " T =", t_stat, "\n")
cat("T+ + T- =", t_plus + t_minus, " n(n+1)/2 =", rank_total, "\n")
stopifnot(abs((t_plus + t_minus) - rank_total) < 1e-8)

cat(
  "Course solution: n = 8, alpha =", alpha_solution,
  ", T_table =", t_table_solution, "\n"
)
cat("reject H0 by that table?", t_stat <= t_table_solution, "\n")

# Exact two-sided p-value. Under H0 every sign pattern is equally likely.
# The midranks stay fixed. Ties are already in those midranks.
sign_patterns <- expand.grid(rep(list(c(-1, 1)), n))
t_plus_null <- apply(sign_patterns, 1, function(signs) {
  signed <- ranks * signs
  sum(signed[signed > 0])
})
t_null <- pmin(t_plus_null, rank_total - t_plus_null)
p_exact <- mean(t_null <= t_stat)
cat("Exact two-sided p-value =", p_exact, "\n")
cat("reject H0 at alpha =", alpha_problem, "?", p_exact <= alpha_problem, "\n")
cat("reject H0 at alpha =", alpha_solution, "?", p_exact <= alpha_solution, "\n\n")

# n = 8, so this Z is not the decision. It checks the formula.
# Here T+ equals T, so substituting either one gives the same Z.
# The mean is still the mean of T+.
mean_t <- n * (n + 1) / 4
sd_t <- sqrt(n * (n + 1) * (2 * n + 1) / 24)
z_t <- (t_plus - mean_t) / sd_t
cat("Normal formula with T+ (not used for the decision, n <= 25)\n")
cat("Z =", z_t, "\n")
cat("Ties make this variance approximate. The exact p-value above is the one to use.\n\n")

# correct = FALSE matches the course Z, which has no continuity correction.
# With ties, R will not compute an exact p-value.
print(wilcox.test(old, new, paired = TRUE, exact = FALSE, correct = FALSE))
cat("\nR default adds a continuity correction:\n")
print(wilcox.test(old, new, paired = TRUE, exact = FALSE, correct = TRUE))
cat("\nR statistic V =", t_plus, "(this is T+, not T)\n")
