# Completely randomized design (DIC, delineamento inteiramente casualizado)
#
# Use DIC when the experimental units are homogeneous.
# Treatments go to the units fully at random. There are no blocks.
# Principles: repetition and randomization.
#
# Model: Y_ij = m + t_i + e_ij
#   m = overall mean, t_i = effect of treatment i, e_ij = random error.
#
# H0: m_1 = m_2 = ... = m_I (all treatment means are equal).
# H1: at least one mean is different.
#
# 1. G = sum of all observations. N = I*r. C = G^2 / N.
# 2. SQTotal = sum(Y^2) - C. Square each observation, then add.
# 3. SQTrat = sum(T_i^2 / r) - C. T_i = total of treatment i.
# 4. SQRes = SQTotal - SQTrat.
# 5. GLTrat = I - 1. GLRes = N - I. GLTotal = N - 1.
# 6. QM = SQ / GL. F = QMTrat / QMRes.
# 7. Reject H0 when F > Ftab(GLTrat, GLRes, alpha).
#
# The ANOVA shows that not all means are equal. It does not show which
# means differ. The comparison scripts (tukey_test.r and others) do that.

alpha <- 0.05

# Example: time in seconds to go from 0 to 100 km/h for five car brands
# of the same category. Four cars of each brand were picked at random
# from the production line and tested on the same track. The cars are
# alike and the track is the same for all, so there is no known source
# of variation to block. That makes this a DIC.
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

y <- cars$time
n_total <- length(y)
n_trt <- nlevels(cars$brand)
r <- n_total / n_trt

g <- sum(y)
c_corr <- g^2 / n_total
trt_totals <- tapply(y, cars$brand, sum)

sq_total <- sum(y^2) - c_corr
sq_trat <- sum(trt_totals^2 / r) - c_corr
sq_res <- sq_total - sq_trat

gl_trat <- n_trt - 1
gl_res <- n_total - n_trt
gl_total <- n_total - 1

qm_trat <- sq_trat / gl_trat
qm_res <- sq_res / gl_res
f_cal <- qm_trat / qm_res
f_tab <- qf(1 - alpha, gl_trat, gl_res)
p_value <- pf(f_cal, gl_trat, gl_res, lower.tail = FALSE)

cat("DIC: 5 car brands x 4 cars, time from 0 to 100 km/h\n")
cat("G =", g, " C =", c_corr, "\n")
cat("treatment totals:", paste(names(trt_totals), trt_totals), "\n\n")
print(data.frame(
  FV = c("Treatments", "Residual", "Total"),
  GL = c(gl_trat, gl_res, gl_total),
  SQ = c(sq_trat, sq_res, sq_total),
  QM = c(qm_trat, qm_res, NA),
  F = c(f_cal, NA, NA)
), row.names = FALSE, digits = 5)
cat("\nFtab(", gl_trat, ",", gl_res, ") =", f_tab, "\n")
cat("p-value =", p_value, "\n")
cat("reject H0 at alpha =", alpha, "?", f_cal > f_tab, "\n")

# By hand: F = 22.30 / 0.91667 = 24.33. Table: Ftab(4, 15) = 3.06.
# If you round QMRes to 0.92 first, you get F = 24.24. Small differences
# like this are normal in hand calculations.
stopifnot(abs(f_cal - 24.33) < 0.01, abs(f_tab - 3.06) < 0.01)

# Built-in check. aov fits the same model.
fit <- aov(time ~ brand, data = cars)
aov_table <- summary(fit)[[1]]
print(summary(fit))
stopifnot(isTRUE(all.equal(unname(aov_table[["Sum Sq"]]), c(sq_trat, sq_res))))

# --- coefficient of variation ---
# CV = sqrt(QMRes) / overall mean * 100. It is the residual error relative
# to the mean. A common scale for field trials (Pimentel-Gomes):
# below 10% low, 10% to 20% medium, 20% to 30% high, above 30% very high.
# A low CV shows good experimental precision.
cv <- sqrt(qm_res) / mean(y) * 100
cat("\nCV =", round(cv, 2), "%\n")

# --- assumptions (pressuposicoes) ---
# Shapiro-Wilk on the residuals. H0: the errors are normal.
# Bartlett on the groups. H0: the treatment variances are equal.
# p > alpha: do not reject the assumption.
#
# This example shows two useful cases:
# - Shapiro-Wilk gives p = 0.04. The times are whole seconds, so the
#   residuals take only a few values, and the test is sensitive to that.
#   When normality fails, transform the data or use a nonparametric test.
# - Bartlett gives p = 1. Every brand has the same within-group SQ (2.75),
#   so the sample variances are identical.
sw <- shapiro.test(residuals(fit))
bt <- bartlett.test(time ~ brand, data = cars)
cat("Shapiro-Wilk p-value =", sw$p.value,
    " normal errors?", sw$p.value > alpha, "\n")
cat("Bartlett p-value =", bt$p.value,
    " equal variances?", bt$p.value > alpha, "\n")
