# Scheffe test (teste de Scheffe)
#
# Use Scheffe for any contrast, also a contrast that you choose after
# you see the data. It protects all possible contrasts at once, so it is
# the most conservative of the comparison tests.
#
# H0: Y = 0.  H1: Y != 0.
#
# 1. Y_hat = sum(c_i * mean_i), with sum(c_i) = 0.
# 2. V(Y_hat) = sum(c_i^2) * QMRes / r.
# 3. F = Ftab(I - 1, GLRes, alpha).
# 4. S = sqrt((I - 1) * F * V(Y_hat)).
# 5. Reject H0 when |Y_hat| > S.

alpha <- 0.05

# contrasts: one row for each contrast. Returns Y_hat, S and the decision.
scheffe <- function(contrasts, means, qm_res, gl_res, r) {
  stopifnot(all(rowSums(contrasts) == 0))
  n_trt <- length(means)
  y_hat <- drop(contrasts %*% means)
  v_y <- rowSums(contrasts^2) * qm_res / r
  s <- sqrt((n_trt - 1) * qf(1 - alpha, n_trt - 1, gl_res) * v_y)
  data.frame(Y = y_hat, S = s, reject_H0 = abs(y_hat) > s)
}

# --- Example: car brands grouped by price (DIC) ---
# Time from 0 to 100 km/h, four cars per brand.
# Price groups: A high, B and C medium, D and E low.
# The same contrasts as t_test_contrasts.r, so you can compare S with t.
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
fit <- aov(time ~ brand, data = cars)
cars_contrasts <- rbind(
  C1 = c(4, -1, -1, -1, -1),
  C2 = c(0, 1, 1, -1, -1),
  C3 = c(0, 1, -1, 0, 0),
  C4 = c(0, 0, 0, 1, -1)
)
cars_result <- scheffe(
  cars_contrasts,
  means = tapply(cars$time, cars$brand, mean),
  qm_res = deviance(fit) / df.residual(fit),
  gl_res = df.residual(fit),
  r = 4
)
cat("Car brands: Scheffe\n")
print(cars_result, digits = 4)
# By hand: S = 7.48, 3.35, 2.37, 2.37. Reject H0 for C2, C3 and C4.
stopifnot(
  all(abs(cars_result$S - c(7.48, 3.35, 2.37, 2.37)) < 0.01),
  cars_result$reject_H0 == c(FALSE, TRUE, TRUE, TRUE)
)

# --- Example: Scheffe and t on the same planned contrasts ---
# DBC with 4 treatments, r = 5, QMRes = 1.82, GLRes = 12.
# The contrasts are planned in advance and orthogonal, so Scheffe and
# t are both valid. They can still disagree, because Scheffe is more
# conservative. Here they disagree on Y3.
dbc_means <- c(A = 27.8, B = 26.0, C = 23.8, D = 31.4)
dbc_qm_res <- 1.82
dbc_gl_res <- 12
dbc_r <- 5

planned <- rbind(
  Y1 = c(3, -1, -1, -1),
  Y2 = c(0, 1, 1, -2),
  Y3 = c(0, 1, -1, 0)
)
planned_result <- scheffe(planned, dbc_means, dbc_qm_res, dbc_gl_res, dbc_r)
planned_t <- planned_result$Y / sqrt(rowSums(planned^2) * dbc_qm_res / dbc_r)
planned_result$t <- planned_t
planned_result$t_rejects <- abs(planned_t) > qt(1 - alpha / 2, dbc_gl_res)
cat("\nPlanned contrasts: Scheffe and t\n")
print(planned_result, digits = 4)
stopifnot(!planned_result["Y3", "reject_H0"], planned_result["Y3", "t_rejects"])

# --- Example: a contrast chosen after you see the data ---
# Same DBC. Y9 = mA + mB - mC - mD is chosen after looking at the means.
# Only Scheffe is valid for that. S = 3.90, so do not reject H0.
y9 <- rbind(Y9 = c(1, 1, -1, -1))
y9_result <- scheffe(y9, dbc_means, dbc_qm_res, dbc_gl_res, dbc_r)
cat("\nContrast chosen after the data\n")
print(y9_result, digits = 4)
stopifnot(abs(y9_result$S - 3.90) < 0.01, !y9_result$reject_H0)

# --- Example: potato cultivars, Scheffe vs Dunnett ---
# Ten cultivars in a DBC with 4 blocks, SQRes = 572.9369, GLRes = 27.
# Contrast: Matilda - Achat, the same pair as in dunnett_test.r.
# Dunnett finds a difference (d = 9.37). Scheffe does not.
potato <- c(
  Achat = 41.63, NA3 = 27.23, Apua = 38.93, Baronesa = 33.12,
  Bintjie = 28.41, Carola = 29.08, Elipsa = 36.6, Matilda = 30.73,
  Omega = 36.93, Radosa = 23.61
)
matilda_vs_achat <- rbind(
  "Matilda - Achat" = (names(potato) == "Matilda") - (names(potato) == "Achat")
)
cat("\nPotatoes: Scheffe on one pair\n")
print(scheffe(matilda_vs_achat, potato, 572.9369 / 27, 27, r = 4), digits = 4)
