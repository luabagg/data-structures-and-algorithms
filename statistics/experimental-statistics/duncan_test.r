# Duncan test (teste de Duncan, multiple range test)
#
# Use Duncan after a significant ANOVA F to compare all pairs of means.
# It finds more differences than Tukey, but its error rate for the whole
# set of comparisons is higher than alpha.
#
# H0: m_i = m_j.  H1: m_i != m_j (for each pair).
#
# Sort the means. Two means that are p places apart in the sorted list
# span a range of p means: p = 2 for neighbours, p = I for the extremes.
# The least significant range for p means is
#   D_p = z_p * sqrt(QMRes / r)
# z_p comes from the Duncan table with p and GLRes.
#
# Duncan sets the significance level of a range of p means to
# 1 - (1 - alpha)^(p - 1). So z_p is a studentized range quantile:
#   z_p = qtukey((1 - alpha)^(p - 1), p, GLRes)
# The table check below confirms this formula.
#
# 1. Sort the means from high to low.
# 2. Compute D_p for p = 2, ..., I.
# 3. Test the widest range first, then the narrower ranges.
#    The two means at the ends of a range differ when diff > D_p.
# 4. Protection rule: when a range is not significant, no pair inside
#    it is significant.

alpha <- 0.05

# means: named treatment means. crit: critical range for p = 2, ..., I
# (crit[1] is for p = 2). Returns one row for each pair.
range_test <- function(means, crit) {
  means <- sort(means, decreasing = TRUE)
  k <- length(means)
  not_sig <- matrix(FALSE, k, k)
  rows <- NULL
  for (p in k:2) {
    for (i in seq_len(k - p + 1)) {
      j <- i + p - 1
      diff <- unname(means[i] - means[j])
      inside_not_sig <- any(not_sig[seq_len(i), j:k])
      differ <- !inside_not_sig && diff > crit[p - 1]
      not_sig[i, j] <- !differ
      rows <- rbind(rows, data.frame(
        pair = paste(names(means)[i], "-", names(means)[j]),
        p = p, diff = diff, crit = crit[p - 1], differ = differ
      ))
    }
  }
  rows
}

# --- Example: car brands, time from 0 to 100 km/h (DIC) ---
# Five brands, four cars per brand. Which brand is the slowest
# (highest time)? completely_randomized_design.r computes this ANOVA
# by hand.
cars <- data.frame(
  brand = factor(rep(c("A", "B", "C", "D", "E"), each = 4)),
  time = c(
    12, 11, 11, 13,
    12, 10, 10, 11,
    8, 7, 8, 6,
    12, 12, 10, 11,
    13, 14, 15, 13
  )
)
r <- 4
fit <- aov(time ~ brand, data = cars)
gl_res <- df.residual(fit)
qm_res <- deviance(fit) / gl_res
means <- tapply(cars$time, cars$brand, mean)

p_values <- 2:length(means)
z <- qtukey((1 - alpha)^(p_values - 1), p_values, gl_res)
d <- z * sqrt(qm_res / r)

cat("Duncan, car brands: QMRes =", qm_res, " GLRes =", gl_res, "\n")
print(data.frame(p = p_values, z = z, D = d), row.names = FALSE, digits = 4)
cat("\n")
print(range_test(means, d), row.names = FALSE, digits = 4)
cat("\nslowest: E. It differs from every other brand.\n")

# Duncan table, 5%, GLRes = 15: z = 3.01, 3.16, 3.25, 3.31 for p = 2 to 5.
stopifnot(all(abs(z - c(3.01, 3.16, 3.25, 3.31)) < 0.01))

# For p = 2 the level is alpha itself. D_2 is the DMS (LSD) of the t test:
# t * sqrt(2 * QMRes / r). qtukey inverts the distribution numerically
# and is accurate to about 4 decimals, so compare with a tolerance.
dms <- qt(1 - alpha / 2, gl_res) * sqrt(2 * qm_res / r)
stopifnot(abs(d[1] - dms) < 1e-4)
