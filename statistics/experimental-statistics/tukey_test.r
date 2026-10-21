# Tukey test (teste de Tukey, HSD)
#
# Use Tukey after a significant ANOVA F to compare all pairs of means.
# It controls the error rate of the whole set of I(I-1)/2 comparisons.
#
# H0: m_i = m_j.  H1: m_i != m_j (for each pair).
#
# 1. Get QMRes and GLRes from the ANOVA.
# 2. Find q in the Tukey table: columns = number of treatments I,
#    rows = GLRes. In R: qtukey(1 - alpha, I, GLRes).
# 3. delta = q * sqrt(QMRes / r). r = repetitions of each mean.
# 4. For each pair, compute |mean_i - mean_j|.
# 5. The pair differs when |mean_i - mean_j| > delta.
#
# One delta serves all pairs. Duncan and SNK change the critical value
# with the number of means in the range. See duncan_test.r and
# student_newman_keuls_test.r.

alpha <- 0.05

# Print every pair from the highest mean down. Return delta.
tukey_pairs <- function(means, qm_res, gl_res, r) {
  means <- sort(means, decreasing = TRUE)
  q <- qtukey(1 - alpha, length(means), gl_res)
  delta <- q * sqrt(qm_res / r)
  pairs <- combn(length(means), 2)
  diff <- means[pairs[1, ]] - means[pairs[2, ]]
  cat("q(", length(means), ",", gl_res, ") =", round(q, 3),
      " delta =", delta, "\n")
  print(data.frame(
    pair = paste(names(means)[pairs[1, ]], "-", names(means)[pairs[2, ]]),
    diff = unname(diff),
    differ = unname(diff > delta)
  ), row.names = FALSE, digits = 4)
  invisible(delta)
}

# TukeyHSD intervals are diff +/- delta. Use this to check delta.
check_with_tukeyhsd <- function(fit, term, delta) {
  hsd <- TukeyHSD(fit, term, conf.level = 1 - alpha)[[term]]
  half_width <- unname(hsd[, "upr"] - hsd[, "diff"])
  stopifnot(isTRUE(all.equal(half_width, rep(delta, nrow(hsd)))))
}

# --- Example: tomato cultivars, only the means are known ---
# Six cultivars in a DIC with r = 5. Response: mean yield (kg).
# QMRes = 36. GLRes = N - I = 30 - 6 = 24.
tomato <- c(Cia1 = 21, Cia2 = 29, Tac1 = 10, Tac2 = 8, Br1 = 13, Br4 = 2)
cat("Tomato cultivars (DIC, r = 5)\n")
delta_tomato <- tukey_pairs(tomato, qm_res = 36, gl_res = 24, r = 5)
# Table: q(6, 24) = 4.37, so delta = 4.37 * 6 / sqrt(5) = 11.73.
# 29 vs 13: 16 > 11.73, they differ. 21 vs 10: 11 < 11.73, they do not.
stopifnot(
  abs(delta_tomato - 11.73) < 0.01,
  29 - 13 > delta_tomato,
  21 - 10 < delta_tomato
)

# --- Example: car brands, time from 0 to 100 km/h (DIC, raw data) ---
# Four cars per brand. completely_randomized_design.r computes this
# ANOVA by hand. Lower time means a faster car.
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
cars_fit <- aov(time ~ brand, data = cars)
cars_qm_res <- deviance(cars_fit) / df.residual(cars_fit)
cars_means <- tapply(cars$time, cars$brand, mean)
cat("\nCar brands, QMRes =", cars_qm_res, "\n")
delta_cars <- tukey_pairs(cars_means, cars_qm_res, df.residual(cars_fit), r = 4)
check_with_tukeyhsd(cars_fit, "brand", delta_cars)
# By hand: delta = 4.37 * sqrt(0.9167 / 4) = 2.09.
# If you round s to 0.96 first, you get 2.10.
stopifnot(abs(delta_cars - 2.09) < 0.01)
cat("fastest (lowest time): C. It differs from every other brand.\n")
cat("slowest: E and A. They do not differ from each other.\n")

# --- Example: insecticides against mealybugs (DBC, raw data) ---
# Five insecticides and a control without insecticide (F).
# Response: live mealybugs per plot (mean of 3 plants), 8 days after
# spraying. Four blocks. Rows: blocks I to IV. Columns: treatments.
bugs <- matrix(
  c(15.7, 23.5, 20.5, 25.7, 15.2, 40.4,
    10.0, 28.3, 18.3, 26.7, 17.1, 43.2,
    12.2, 23.4, 20.1, 25.4, 18.2, 45.1,
    13.2, 27.6, 22.5, 27.2, 16.6, 45.0),
  nrow = 4, byrow = TRUE,
  dimnames = list(block = c("I", "II", "III", "IV"), trt = LETTERS[1:6])
)
bugs_long <- data.frame(
  y = as.vector(bugs),
  trt = factor(rep(colnames(bugs), each = nrow(bugs))),
  block = factor(rep(rownames(bugs), times = ncol(bugs)))
)
bugs_fit <- aov(y ~ trt + block, data = bugs_long)
bugs_qm_res <- deviance(bugs_fit) / df.residual(bugs_fit)
cat("\nInsecticides (DBC, 4 blocks), QMRes =", bugs_qm_res, "\n")
delta_bugs <- tukey_pairs(colMeans(bugs), bugs_qm_res, df.residual(bugs_fit), r = 4)
check_with_tukeyhsd(bugs_fit, "trt", delta_bugs)
# By hand: QMRes = 3.7359, q(6, 15) = 4.59, delta = 4.44.
stopifnot(abs(bugs_qm_res - 3.7359) < 1e-4, abs(delta_bugs - 4.44) < 0.01)

# --- Example: DQL 5 x 5, only treatment totals and SQRes are known ---
# SQRes = 34116. Each treatment total has r = 5 values.
# GLRes = (5 - 1)(5 - 2) = 12. QMRes = 34116 / 12.
dql_totals <- c(A = 3024, B = 2549, C = 2349, D = 1970, E = 1734)
cat("\nDQL 5 x 5 from totals, QMRes = 34116 / 12\n")
tukey_pairs(dql_totals / 5, qm_res = 34116 / 12, gl_res = 12, r = 5)

# --- Example: potato cultivars, only the means are known ---
# Ten cultivars in a DBC with 4 blocks. Response: total tuber yield.
# SQRes = 572.9369. GLRes = (10 - 1)(4 - 1) = 27.
potato <- c(
  Achat = 41.63, NA3 = 27.23, Apua = 38.93, Baronesa = 33.12,
  Bintjie = 28.41, Carola = 29.08, Elipsa = 36.6, Matilda = 30.73,
  Omega = 36.93, Radosa = 23.61
)
cat("\nPotato cultivars (DBC, 4 blocks), QMRes = 572.9369 / 27\n")
tukey_pairs(potato, qm_res = 572.9369 / 27, gl_res = 27, r = 4)

# --- Example: Tukey and t can disagree ---
# DBC with 4 treatments, r = 5, QMRes = 1.82, GLRes = 12.
# Means: A = 27.8, B = 26.0, C = 23.8, D = 31.4. Contrast Y3 = mB - mC.
# One analyst plans Y3 in advance, with other orthogonal contrasts, and
# uses t. Another compares all pairs with Tukey. Both are correct.
# Tukey protects 6 comparisons at once, so its critical value is larger.
y3 <- 26.0 - 23.8
delta_y3 <- qtukey(1 - alpha, 4, 12) * sqrt(1.82 / 5)
t_cal_y3 <- y3 / sqrt(2 * 1.82 / 5)
t_tab_y3 <- qt(1 - alpha / 2, 12)
cat("\nTukey and t on Y3 = mB - mC =", y3, "\n")
cat("Tukey: delta =", delta_y3, " significant?", y3 > delta_y3, "\n")
cat("t:     t =", t_cal_y3, " ttab =", t_tab_y3,
    " significant?", abs(t_cal_y3) > t_tab_y3, "\n")
stopifnot(y3 < delta_y3, abs(t_cal_y3) > t_tab_y3)
