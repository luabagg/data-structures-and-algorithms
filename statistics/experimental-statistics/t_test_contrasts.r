# t test for contrasts (teste t para contrastes)
#
# A contrast is a linear combination of treatment means whose
# coefficients add to zero:
#   Y = c_1 m_1 + c_2 m_2 + ... + c_I m_I,  sum(c_i) = 0
#
# Use the t test for contrasts that you choose before you see the data
# (a priori) and that are orthogonal to each other.
# Two contrasts are orthogonal when sum(c_i * c'_i) = 0 (equal r).
# With unequal r the rule is sum(c_i * c'_i / r_i) = 0.
#
# H0: Y = 0.  H1: Y != 0.
#
# 1. Y_hat = sum(c_i * mean_i).
# 2. V(Y_hat) = sum(c_i^2) * QMRes / r.  s(Y_hat) = sqrt(V(Y_hat)).
# 3. t = Y_hat / s(Y_hat). Divide by s(Y_hat), not by V(Y_hat).
# 4. ttab = t(GLRes) at alpha / 2 in each tail. In a table that gives
#    p = P[T >= tc], use the column p = 0.025 for alpha = 5%.
# 5. Reject H0 when |t| > ttab.
#
# A contrast of two means (c = 1, -1) gives the DMS (LSD) test:
#   DMS = ttab * sqrt(2 * QMRes / r). Two means differ when |diff| > DMS.

alpha <- 0.05

# --- Example: pineapple intercropping, equal r ---
# Four treatments, r = 5. Response: fruit yield (t/ha).
#   1 = pineapple 0.90 x 0.30 m, alone
#   2 = pineapple 0.80 x 0.30 m, alone
#   3 = pineapple 0.80 x 0.30 m + peanut
#   4 = pineapple 0.80 x 0.30 m + beans
# Contrasts of practical interest:
#   Y1 = m1 + m2 - m3 - m4: pineapple alone vs intercropped
#   Y2 = m1 - m2: the two spacings
#   Y3 = m3 - m4: peanut vs beans as the second crop
shared_contrasts <- rbind(
  Y1 = c(1, 1, -1, -1),
  Y2 = c(1, -1, 0, 0),
  Y3 = c(0, 0, 1, -1)
)

# With equal r, use sum(c_i * c'_i). All three pairs give zero.
pineapple <- c(53.5, 56.5, 62.0, 60.4)
equal_products <- shared_contrasts %*% t(shared_contrasts)
cat("Pineapple, equal r: Y_hat =",
    drop(shared_contrasts %*% pineapple), "\n")
print(equal_products)
stopifnot(all(equal_products[upper.tri(equal_products)] == 0))

# --- Example: the same contrasts with unequal r ---
# Some plots were lost, so r = 6, 6, 4, 5. QMRes = 0.45.
# Use the general formulas:
#   V(Y_hat) = QMRes * sum(c_i^2 / r_i)
#   orthogonal when sum(c_i * c'_i / r_i) = 0
unequal_means <- c(11.2, 10.5, 10, 21)
unequal_r <- c(6, 6, 4, 5)
unequal_qm_res <- 0.45
unequal_y <- drop(shared_contrasts %*% unequal_means)
unequal_v <- unequal_qm_res * drop(shared_contrasts^2 %*% (1 / unequal_r))
unequal_products <- shared_contrasts %*% diag(1 / unequal_r) %*% t(shared_contrasts)
cat("\nUnequal r\n")
print(data.frame(Y = unequal_y, V = unequal_v), digits = 4)
cat("\nsum(c_i * c'_i / r_i) for each pair of contrasts\n")
print(unequal_products, digits = 4)

# Y1 and Y3 are orthogonal with equal r but not here:
# (-1)(1)/4 + (-1)(-1)/5 = -0.05. Y1 gives a different weight to m3
# and m4 because they have a different number of repetitions.
stopifnot(
  abs(unequal_products["Y1", "Y2"]) < 1e-12,
  abs(unequal_products["Y2", "Y3"]) < 1e-12,
  abs(unequal_products["Y1", "Y3"] - (-0.05)) < 1e-12
)

# --- Example: car brands grouped by price (DIC) ---
# Time from 0 to 100 km/h, four cars per brand.
# completely_randomized_design.r computes this ANOVA by hand.
# Price groups: A high, B and C medium, D and E low.
# Planned contrasts:
#   C1: high price vs all other brands
#   C2: medium price vs low price
#   C3: B vs C, inside the medium group
#   C4: D vs E, inside the low group
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

# One row for each contrast, one column for each brand A to E.
contrasts <- rbind(
  C1 = c(4, -1, -1, -1, -1),
  C2 = c(0, 1, 1, -1, -1),
  C3 = c(0, 1, -1, 0, 0),
  C4 = c(0, 0, 0, 1, -1)
)
stopifnot(all(rowSums(contrasts) == 0))

# Orthogonality: every off-diagonal product sum must be zero.
products <- contrasts %*% t(contrasts)
cat("\nCar brands: sum(c_i * c'_i) for each pair of contrasts\n")
print(products)
stopifnot(all(products[upper.tri(products)] == 0))

y_hat <- drop(contrasts %*% means)
v_y <- rowSums(contrasts^2) * qm_res / r
t_cal <- y_hat / sqrt(v_y)
t_tab <- qt(1 - alpha / 2, gl_res)

cat("\nQMRes =", qm_res, " GLRes =", gl_res, " ttab =", t_tab, "\n")
print(data.frame(
  Y = y_hat, V = v_y, t = t_cal, reject_H0 = abs(t_cal) > t_tab
), digits = 4)

# By hand: Y = 4.0, -7.0, 3.5, -2.5. t = 1.87, -7.31, 5.17, -3.69.
# Table: t(15) = 2.13.
stopifnot(
  all(abs(y_hat - c(4.0, -7.0, 3.5, -2.5)) < 1e-8),
  all(abs(t_cal - c(1.87, -7.31, 5.17, -3.69)) < 0.01),
  abs(t_tab - 2.13) < 0.01
)

# DMS: C3 and C4 compare two means, so the t test is the DMS test.
dms <- t_tab * sqrt(2 * qm_res / r)
cat("\nDMS =", dms, "\n")
stopifnot((abs(y_hat[3:4]) > dms) == (abs(t_cal[3:4]) > t_tab))
