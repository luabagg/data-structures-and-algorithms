# Chi-squared test
# Notes: obsidian-statistics-vault → Statistical Inference/Chi-Squared Test
#
# Categorical counts. Compare observed frequencies with expected frequencies
# under a null model.
#
# Goodness of fit: one sample vs known category probabilities.
# Independence: two-way contingency table (row variable vs column variable).
#
# Under H0 and large enough expected counts, the Pearson statistic
#   X^2 = sum( (O - E)^2 / E )
# is approximately Chi-squared with df = k - 1 - (# estimated params)
# (goodness of fit) or df = (r - 1)(c - 1) (independence).

# --- goodness of fit: fair six-sided die? ---
# Observed faces from n = 60 rolls. H0: each face has p = 1/6.

observed_die <- c(12, 8, 11, 9, 10, 10)
names(observed_die) <- 1:6
n_die <- sum(observed_die)
p0_die <- rep(1 / 6, 6)
expected_die <- n_die * p0_die
alpha <- 0.05

# Manual Pearson X^2
x2_manual <- sum((observed_die - expected_die)^2 / expected_die)
df_gof <- length(observed_die) - 1
p_manual <- pchisq(x2_manual, df = df_gof, lower.tail = FALSE)

cat("Goodness of fit (die)\n")
cat("X^2 manual =", x2_manual, " df =", df_gof, "\n")
cat("p-value manual =", p_manual, "\n")
cat("reject H0 at alpha =", alpha, "?", p_manual < alpha, "\n\n")

gof <- chisq.test(x = observed_die, p = p0_die)
print(gof)

# Rule of thumb: expected counts preferably >= 5 for the chi-squared
# approximation. Inspect expected under H0:
cat("\nExpected counts:\n")
print(gof$expected)

# --- test of independence: 2x2 contingency table ---
# Rows: treatment A / B. Columns: success / failure.
# H0: outcome independent of treatment (same success rate).

table_tx <- matrix(
  c(40, 20,
    30, 30),
  nrow = 2,
  byrow = TRUE,
  dimnames = list(
    treatment = c("A", "B"),
    outcome = c("success", "failure")
  )
)

cat("\nIndependence (treatment x outcome)\n")
print(table_tx)

# Expected under independence: E_ij = (row_i total * col_j total) / n
row_tot <- rowSums(table_tx)
col_tot <- colSums(table_tx)
n_tab <- sum(table_tx)
expected_ind <- outer(row_tot, col_tot) / n_tab
x2_ind_manual <- sum((table_tx - expected_ind)^2 / expected_ind)
df_ind <- (nrow(table_tx) - 1) * (ncol(table_tx) - 1)
p_ind_manual <- pchisq(x2_ind_manual, df = df_ind, lower.tail = FALSE)

cat("X^2 manual =", x2_ind_manual, " df =", df_ind, "\n")
cat("p-value manual =", p_ind_manual, "\n")
cat("reject H0 at alpha =", alpha, "?", p_ind_manual < alpha, "\n\n")

ind <- chisq.test(table_tx, correct = FALSE)
print(ind)
cat("\nExpected under independence:\n")
print(ind$expected)

# Yates continuity correction is the default for 2x2 in chisq.test.
# Set correct = FALSE to match the plain Pearson statistic above.
ind_yates <- chisq.test(table_tx, correct = TRUE)
cat("\nWith Yates continuity correction:\n")
print(ind_yates)
