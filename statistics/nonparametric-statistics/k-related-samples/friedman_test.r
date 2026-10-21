# Friedman test
# k amostras relacionadas
#
# k treatments measured on the same n blocks (subjects, judges, plots).
# Ordinal data, or numeric data where normality for a two-way ANOVA
# (randomized block design) is doubtful.
# With k = 2 it reduces to the sign test.
#
# H0: the k treatments have the same distribution.
# H1: at least one treatment tends to give larger values.
#
# 1. Rank the k values inside each block, from 1 to k.
#    Ties inside a block get the average rank.
# 2. Rj = sum of the ranks of treatment j over the n blocks.
# Check: sum(Rj) = n*k*(k+1)/2.
# 3. Fr = 12 / (n*k*(k+1)) * sum(Rj^2) - 3*n*(k+1)
#    Some books write Fr as chi_r^2.
# 4. For large n, Fr follows chi-squared with k - 1 df under H0.
#    Reject H0 when Fr >= chi^2 table.
#    For small n and k, books give an exact table. The exact p-value
#    below does the same job.
#
# Ties shrink the variance of the ranks. The tie-corrected statistic is
#   Fr_ties = 12 * sum((Rj - n*(k+1)/2)^2) /
#             (n*k*(k+1) - sum(t^3 - t) / (k - 1))
# where t is the size of each tie group inside a block.
# Without ties this is the same Fr. R's friedman.test uses this form.

# Illustrative data (not from the course). Six judges score three
# recipes from 0 to 10. Each judge is a block. Judge 2 ties B and C.
scores <- matrix(
  c(7, 5, 6,
    8, 6, 6,
    9, 7, 8,
    6, 5, 4,
    8, 7, 5,
    9, 6, 7),
  ncol = 3,
  byrow = TRUE,
  dimnames = list(judge = paste0("J", 1:6), recipe = c("A", "B", "C"))
)
alpha <- 0.05

n <- nrow(scores)
k <- ncol(scores)

# apply returns one column per block, so transpose back.
ranks <- t(apply(scores, 1, rank))
r_j <- colSums(ranks)
rank_total <- n * k * (k + 1) / 2

cat("Scores:\n")
print(scores)
cat("\nRanks inside each judge:\n")
print(ranks)
cat("\nRj =", paste(names(r_j), r_j, sep = ": ", collapse = "  "), "\n")
cat("sum(Rj) =", sum(r_j), " n*k*(k+1)/2 =", rank_total, "\n")
stopifnot(abs(sum(r_j) - rank_total) < 1e-8)

fr <- 12 / (n * k * (k + 1)) * sum(r_j^2) - 3 * n * (k + 1)

tie_sizes <- unlist(apply(scores, 1, function(block) as.vector(table(block))))
tie_term <- sum(tie_sizes^3 - tie_sizes)
fr_ties <- 12 * sum((r_j - n * (k + 1) / 2)^2) /
  (n * k * (k + 1) - tie_term / (k - 1))

df <- k - 1
chi_tab <- qchisq(1 - alpha, df = df)
p_chi <- pchisq(fr_ties, df = df, lower.tail = FALSE)

cat("\nFr =", fr, " Fr with tie correction =", fr_ties, "\n")
cat("df =", df, " chi^2 table (alpha =", alpha, ") =", chi_tab, "\n")
cat("reject H0 because Fr >= table?", fr_ties >= chi_tab, "\n")
cat("p-value (chi-squared approximation) =", p_chi, "\n\n")

# Exact p-value. Under H0 each block's ranks are equally likely in any
# order, and blocks are independent. That gives (k!)^n arrangements.
# Here 6^6 = 46656. The midranks stay fixed, so ties are already in them.
# The tie correction is the same constant for every arrangement, so
# ordering the arrangements by Fr or by Fr_ties gives the same p-value.
perms <- as.matrix(expand.grid(rep(list(1:k), k)))
perms <- perms[apply(perms, 1, function(p) length(unique(p)) == k), , drop = FALSE]

block_orders <- lapply(seq_len(n), function(i) {
  t(apply(perms, 1, function(p) ranks[i, p]))
})
choices <- as.matrix(expand.grid(rep(list(seq_len(nrow(perms))), n)))
r_null <- matrix(0, nrow = nrow(choices), ncol = k)
for (i in seq_len(n)) {
  r_null <- r_null + block_orders[[i]][choices[, i], , drop = FALSE]
}
fr_null <- 12 / (n * k * (k + 1)) * rowSums(r_null^2) - 3 * n * (k + 1)
p_exact <- mean(fr_null >= fr - 1e-8)

cat("Exact p-value (", nrow(choices), " arrangements) =", p_exact, "\n")
cat("reject H0 at alpha =", alpha, "?", p_exact <= alpha, "\n\n")

# friedman.test wants blocks in rows and treatments in columns.
check <- friedman.test(scores)
print(check)
stopifnot(abs(unname(check$statistic) - fr_ties) < 1e-8)
