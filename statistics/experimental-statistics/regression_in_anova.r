# Regression in the ANOVA (regressao na analise de variancia)
#
# Use it when the treatments are quantitative levels, for example doses
# of a fertilizer. Do not compare doses pair by pair with Tukey. Fit a
# curve through the treatment means.
#
# The treatment sum of squares splits into polynomial parts:
#   SQTrat = SQ(linear) + SQ(quadratic) + ... + SQ(degree I - 1)
# Each part has 1 GL. For a model of degree k, the deviations
# (desvios da regressao, lack of fit) are what the model does not explain:
#   SQ(deviations) = SQTrat - SQ(linear) - ... - SQ(degree k),
#   GL(deviations) = I - 1 - k
#
# Test each part with F = QM(part) / QMRes. QMRes comes from the ANOVA
# (pure error), not from the residual of the curve fit.
#
# With I equally spaced levels and r repetitions, use the orthogonal
# polynomial coefficients c (printed table, or contr.poly(I) in R):
#   SQ(part) = (sum c_i * T_i)^2 / (r * sum c_i^2),  T_i = treatment total
#
# Choose the highest degree that is significant, if its deviations are
# not significant. Then fit the curve. The levels must be equally spaced
# for these coefficients.
#
# R^2 = SQ(regression) / SQTrat. It shows how much of the treatment
# variation the curve explains.

alpha <- 0.05

# Example (illustrative values): a DIC tests five nitrogen doses on a
# crop, with r = 4 plots for each dose. Response: yield.
# Yield rises with the dose and then levels off, so a straight line is
# not enough. The doses are quantitative and equally spaced (every
# 30 kg/ha), so fit a polynomial in place of a means test.
dose_levels <- c(0, 30, 60, 90, 120)
r <- 4
crop <- data.frame(
  dose = rep(dose_levels, each = r),
  yield = c(
    20.1, 21.4, 19.8, 22.0,
    26.3, 27.9, 25.1, 26.8,
    30.2, 31.5, 29.4, 30.9,
    31.0, 32.6, 30.1, 31.8,
    29.5, 30.8, 28.7, 30.2
  )
)
n_trt <- length(dose_levels)

# --- step 1: the usual DIC ANOVA ---
fit_anova <- aov(yield ~ factor(dose), data = crop)
anova_table <- summary(fit_anova)[[1]]
sq_trat <- anova_table[["Sum Sq"]][1]
gl_res <- df.residual(fit_anova)
qm_res <- deviance(fit_anova) / gl_res
cat("DIC ANOVA, doses as categories\n")
print(summary(fit_anova))

# --- step 2: split SQTrat with orthogonal polynomial coefficients ---
# Integer coefficients for I = 5 from the orthogonal polynomial table.
coefs <- rbind(
  linear = c(-2, -1, 0, 1, 2),
  quadratic = c(2, -1, -2, -1, 2),
  cubic = c(-1, 2, 0, -2, 1),
  quartic = c(1, -4, 6, -4, 1)
)
# They point the same way as R's contr.poly. Only the scale differs.
stopifnot(isTRUE(all.equal(
  unname(t(contr.poly(n_trt))),
  unname(coefs / sqrt(rowSums(coefs^2)))
)))

trt_totals <- tapply(crop$yield, crop$dose, sum)
sq_parts <- drop(coefs %*% trt_totals)^2 / (r * rowSums(coefs^2))
# The parts add up to SQTrat.
stopifnot(isTRUE(all.equal(sum(sq_parts), sq_trat)))

f_parts <- sq_parts / qm_res
f_tab <- qf(1 - alpha, 1, gl_res)

# Degree I - 1 leaves no deviations (GL = 0), so stop at I - 2.
degree <- seq_len(n_trt - 2)
sq_dev <- sq_trat - cumsum(sq_parts)[degree]
gl_dev <- n_trt - 1 - degree
f_dev <- (sq_dev / gl_dev) / qm_res
f_dev_tab <- qf(1 - alpha, gl_dev, gl_res)

cat("\nSQTrat =", sq_trat, " QMRes =", qm_res, " GLRes =", gl_res, "\n")
cat("Ftab(1,", gl_res, ") =", f_tab, "\n\n")
print(data.frame(
  part = names(sq_parts)[degree],
  SQ = sq_parts[degree],
  F = f_parts[degree],
  significant = f_parts[degree] > f_tab,
  SQ_dev = sq_dev,
  GL_dev = gl_dev,
  F_dev = f_dev,
  dev_significant = f_dev > f_dev_tab
), row.names = FALSE, digits = 4)

# Here the quadratic term is significant and its deviations are not.
# The cubic term is not significant. So choose degree 2.
stopifnot(
  f_parts["quadratic"] > f_tab,
  f_dev[2] < f_dev_tab[2],
  f_parts["cubic"] < f_tab
)

# Built-in check. Sequential SS of lm with dose, dose^2 and factor(dose):
# dose = linear, I(dose^2) = quadratic, factor(dose) = deviations of the
# quadratic model, Residuals = the ANOVA residual.
seq_ss <- anova(lm(yield ~ dose + I(dose^2) + factor(dose), data = crop))
stopifnot(isTRUE(all.equal(
  unname(seq_ss[["Sum Sq"]]),
  c(sq_parts["linear"], sq_parts["quadratic"], sq_dev[2], deviance(fit_anova)),
  check.attributes = FALSE
)))

# --- step 3: fit the chosen curve ---
# With equal r, the fit on all observations and the fit on the treatment
# means give the same coefficients.
curve <- lm(yield ~ dose + I(dose^2), data = crop)
b <- coef(curve)
r2 <- sum(sq_parts[1:2]) / sq_trat
cat(sprintf("\nyield = %.4f %+.5f * dose %+.7f * dose^2\n", b[1], b[2], b[3]))
cat("R^2 =", round(r2, 4), "\n")

# Dose of maximum yield: the derivative b1 + 2 * b2 * dose is zero.
dose_max <- -b[2] / (2 * b[3])
cat("dose of maximum yield =", round(dose_max, 1), "kg/ha\n")
cat("maximum yield =", round(predict(curve, data.frame(dose = dose_max)), 2), "\n")
