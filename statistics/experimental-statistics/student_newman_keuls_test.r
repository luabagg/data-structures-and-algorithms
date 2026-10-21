# Student-Newman-Keuls test (SNK, teste de Student-Newman-Keuls)
#
# SNK is a range test like Duncan. It uses Tukey's table, but with the
# number of means in the range (p) in place of the number of treatments.
#
# H0: m_i = m_j.  H1: m_i != m_j (for each pair).
#
# Critical range for a range of p sorted means:
#   SNK_p = q(p, GLRes) * sqrt(QMRes / r),  q = qtukey(1 - alpha, p, GLRes)
#
# Strictness, from most to least strict:
#   Tukey:  q(I, GLRes) for every pair.
#   SNK:    q(p, GLRes). Equals Tukey for the widest range (p = I).
#   Duncan: qtukey((1 - alpha)^(p - 1), p, GLRes). Smaller than SNK for p > 2.
#
# 1. Sort the means from high to low.
# 2. Compute SNK_p for p = 2, ..., I.
# 3. Test the widest range first, then the narrower ranges.
# 4. Protection rule: when a range is not significant, no pair inside
#    it is significant.

alpha <- 0.05

# Same procedure as in duncan_test.r. Only the critical values differ.
# crit[1] is for p = 2.
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
# The same data as tukey_test.r and duncan_test.r, so you can compare
# the three tests on one set of means.
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
n_trt <- length(means)

p_values <- 2:n_trt
snk <- qtukey(1 - alpha, p_values, gl_res) * sqrt(qm_res / r)
duncan <- qtukey((1 - alpha)^(p_values - 1), p_values, gl_res) * sqrt(qm_res / r)
tukey <- qtukey(1 - alpha, n_trt, gl_res) * sqrt(qm_res / r)

cat("Car brands: critical ranges\n")
print(data.frame(
  p = p_values, Duncan = duncan, SNK = snk, Tukey = tukey
), row.names = FALSE, digits = 4)
cat("\nSNK\n")
print(range_test(means, snk), row.names = FALSE, digits = 4)

# E - A: diff = 2.0. SNK (p = 2) and Duncan say they differ.
# Tukey (delta = 2.09) says they do not.
cat("\nE - A: SNK and Duncan differ, Tukey does not.\n")
stopifnot(
  isTRUE(all.equal(snk[n_trt - 1], tukey)),
  all(duncan[-1] < snk[-1]),
  2.0 > snk[1],
  2.0 < tukey
)
