# Kruskal-Wallis test
# k amostras independentes
#
# k treatments, each applied to its own group of experimental units.
# This is the nonparametric counterpart of the one-way ANOVA in a
# completely randomized design (delineamento inteiramente casualizado,
# see experimental-statistics/completely_randomized_design.r).
# Use it for ordinal data, or numeric data where normality or equal
# variances for the ANOVA are doubtful. Groups may have different sizes.
# With k = 2 it is the Mann-Whitney test (two-sided).
#
# H0: the k treatments have the same distribution (no difference).
# H1: at least one treatment differs from the others.
#
# 1. Pool all N = n1 + ... + nk values and rank them from 1 to N.
#    Ties get the average rank.
# 2. Rj = sum of the ranks in group j.
# Check: sum(Rj) = N*(N+1)/2.
# 3. H = 12 / (N*(N+1)) * sum(Rj^2 / nj) - 3*(N+1)
# 4. For large groups (each nj > 5 is the usual rule), H follows
#    chi-squared with k - 1 df under H0.
#    Reject H0 when H >= chi^2 table.
#    For small groups, books give an exact table. The permutation
#    p-value below does the same job.
#
# Ties shrink the variance of the ranks. The tie-corrected statistic is
#   H_ties = H / (1 - sum(t^3 - t) / (N^3 - N))
# where t is the size of each tie group in the pooled sample.
# Without ties the divisor is 1. R's kruskal.test uses this form.

# Illustrative data (not from the course). Plant height (cm) after
# three fertilizers, plots assigned at random. Group sizes differ.
# B and C share a tie at 18.
height <- list(
  A = c(12, 15, 14, 11, 13),
  B = c(18, 16, 20, 17),
  C = c(19, 22, 18, 21, 23, 24)
)
alpha <- 0.05

values <- unlist(height, use.names = FALSE)
group <- factor(rep(names(height), lengths(height)), levels = names(height))
n_j <- lengths(height)
n_total <- length(values)
k <- length(height)

ranks <- rank(values)
r_j <- tapply(ranks, group, sum)
rank_total <- n_total * (n_total + 1) / 2

cat("Values and pooled ranks:\n")
print(data.frame(group, value = values, rank = ranks))
cat("\nnj =", paste(names(n_j), n_j, sep = ": ", collapse = "  "), "\n")
cat("Rj =", paste(names(r_j), r_j, sep = ": ", collapse = "  "), "\n")
cat("mean rank =", paste(names(r_j), round(r_j / n_j, 3), sep = ": ", collapse = "  "),
    " (H0 expects", (n_total + 1) / 2, "each)\n")
cat("sum(Rj) =", sum(r_j), " N*(N+1)/2 =", rank_total, "\n")
stopifnot(abs(sum(r_j) - rank_total) < 1e-8)

h <- 12 / (n_total * (n_total + 1)) * sum(r_j^2 / n_j) - 3 * (n_total + 1)

tie_sizes <- as.vector(table(values))
tie_term <- sum(tie_sizes^3 - tie_sizes)
h_ties <- h / (1 - tie_term / (n_total^3 - n_total))

df <- k - 1
chi_tab <- qchisq(1 - alpha, df = df)
p_chi <- pchisq(h_ties, df = df, lower.tail = FALSE)

cat("\nH =", h, " H with tie correction =", h_ties, "\n")
cat("df =", df, " chi^2 table (alpha =", alpha, ") =", chi_tab, "\n")
cat("reject H0 because H >= table?", h_ties >= chi_tab, "\n")
cat("p-value (chi-squared approximation) =", p_chi, "\n\n")

# Permutation p-value. Under H0 the group labels are exchangeable: any
# split of the N ranks into groups of sizes n1, ..., nk is equally
# likely. Here there are 15! / (5! 4! 6!) = 630630 splits, too many to
# list comfortably, so draw a large random sample of them instead.
# The tie correction is the same constant for every split, so ordering
# splits by H or by H_ties gives the same p-value.
set.seed(42)
n_perm <- 100000
h_null <- replicate(n_perm, {
  r_perm <- tapply(ranks, sample(group), sum)
  12 / (n_total * (n_total + 1)) * sum(r_perm^2 / n_j) - 3 * (n_total + 1)
})
p_perm <- mean(h_null >= h - 1e-8)

cat("Permutation p-value (", n_perm, " random splits) =", p_perm, "\n")
cat("reject H0 at alpha =", alpha, "?", p_perm <= alpha, "\n\n")

# Rejecting H0 says only that some treatment differs. To find which
# pairs differ, follow up with pairwise comparisons (for example Dunn's
# test or pairwise Mann-Whitney with a Bonferroni correction), the same
# role Tukey plays after the ANOVA.
print(pairwise.wilcox.test(values, group, p.adjust.method = "bonferroni", exact = FALSE))

check <- kruskal.test(values, group)
print(check)
stopifnot(abs(unname(check$statistic) - h_ties) < 1e-8)
