# Latin square design (DQL, delineamento em quadrado latino)
#
# Use DQL when two known sources of variation act on the units, for
# example soil fertility in two directions. Rows and columns are two
# blocking criteria. Each treatment appears once in each row and once in
# each column. Principles: repetition, randomization and local control
# (used twice: rows and columns).
#
# Model: Y_ijk = m + t_i + l_j + c_k + e_ijk
#   t_i = treatment, l_j = row (linha), c_k = column (coluna).
#
# An I x I square has I^2 observations. Each treatment, row and column
# total has I values.
#
# 1. C = G^2 / I^2. SQTotal = sum(Y^2) - C.
# 2. SQTrat = sum(T_i^2 / I) - C.
# 3. SQLinhas = sum(L_j^2 / I) - C. SQColunas = sum(C_k^2 / I) - C.
# 4. SQRes = SQTotal - SQTrat - SQLinhas - SQColunas.
# 5. GLTrat = GLLinhas = GLColunas = I - 1. GLRes = (I - 1)(I - 2).
# 6. F = QMsource / QMRes for each source. Reject H0 when F > Ftab.

alpha <- 0.05

# y: I x I matrix of responses. trt: I x I matrix of treatment letters.
latin_square_anova <- function(y, trt) {
  size <- nrow(y)
  # Latin square rule: each letter once per row and once per column.
  stopifnot(
    all(apply(trt, 1, anyDuplicated) == 0),
    all(apply(trt, 2, anyDuplicated) == 0)
  )
  c_corr <- sum(y)^2 / size^2
  trt_totals <- tapply(y, trt, sum)
  sq_total <- sum(y^2) - c_corr
  sq_trat <- sum(trt_totals^2) / size - c_corr
  sq_row <- sum(rowSums(y)^2) / size - c_corr
  sq_col <- sum(colSums(y)^2) / size - c_corr
  sq_res <- sq_total - sq_trat - sq_row - sq_col
  gl <- c(rep(size - 1, 3), (size - 1) * (size - 2), size^2 - 1)
  sq <- c(sq_trat, sq_row, sq_col, sq_res, sq_total)
  qm <- sq / gl
  qm_res <- qm[4]
  table <- data.frame(
    FV = c("Treatments", "Rows", "Columns", "Residual", "Total"),
    GL = gl,
    SQ = sq,
    QM = c(qm[1:4], NA),
    F = c(qm[1:3] / qm_res, NA, NA)
  )
  list(
    table = table,
    trt_totals = trt_totals,
    means = trt_totals / size,
    qm_res = qm_res,
    gl_res = gl[4],
    f_tab = qf(1 - alpha, size - 1, gl[4]),
    cv = sqrt(qm_res) / mean(y) * 100
  )
}

# Built-in check: aov with three factors gives the same sums of squares.
check_with_aov <- function(y, trt, result) {
  size <- nrow(y)
  long <- data.frame(
    y = as.vector(y),
    trt = factor(as.vector(trt)),
    row = factor(rep(seq_len(size), times = size)),
    col = factor(rep(seq_len(size), each = size))
  )
  fit <- aov(y ~ trt + row + col, data = long)
  stopifnot(isTRUE(all.equal(
    unname(summary(fit)[[1]][["Sum Sq"]]),
    result$table$SQ[1:4]
  )))
  fit
}

# Tukey after a DQL. Each treatment mean has r = I values.
# tukey_test.r explains the test in full.
print_tukey <- function(result, fit) {
  size <- length(result$means)
  q <- qtukey(1 - alpha, size, result$gl_res)
  delta <- q * sqrt(result$qm_res / size)
  sorted <- sort(result$means, decreasing = TRUE)
  cat("Tukey: q(", size, ",", result$gl_res, ") =", q, " delta =", delta, "\n")
  cat("means (high to low):", paste(names(sorted), sorted), "\n")
  same_as_top <- names(sorted)[sorted[1] - sorted <= delta]
  cat("no difference from the top mean:", same_as_top, "\n")
  # TukeyHSD intervals are diff +/- delta.
  hsd <- TukeyHSD(fit, "trt", conf.level = 1 - alpha)$trt
  stopifnot(isTRUE(all.equal(unname(hsd[, "upr"] - hsd[, "diff"]), rep(delta, nrow(hsd)))))
}

# --- Example: forage sugarcane varieties, 5 x 5 square ---
# Five varieties: A = CO290, B = CO294, C = CO297, D = CO299, E = CO295.
# Soil fertility changes in two directions across the field. Rows and
# columns of plots block the two directions.
# Response: production (kg per plot).

cane_y <- matrix(
  c(432, 518, 458, 583, 331,
    724, 478, 524, 550, 400,
    489, 384, 556, 297, 420,
    494, 500, 313, 486, 501,
    515, 660, 438, 394, 318),
  nrow = 5, byrow = TRUE
)
cane_trt <- matrix(
  c("D", "A", "B", "C", "E",
    "C", "E", "A", "B", "D",
    "E", "B", "C", "D", "A",
    "B", "D", "E", "A", "C",
    "A", "C", "D", "E", "B"),
  nrow = 5, byrow = TRUE
)
# Check the data against the known row and column totals.
stopifnot(
  rowSums(cane_y) == c(2322, 2676, 2146, 2294, 2325),
  colSums(cane_y) == c(2654, 2540, 2289, 2310, 1970)
)

cane <- latin_square_anova(cane_y, cane_trt)
cat("DQL 5 x 5: sugarcane varieties, kg per plot\n")
cat("treatment totals:", paste(names(cane$trt_totals), cane$trt_totals), "\n\n")
print(cane$table, row.names = FALSE, digits = 5)
cat("\nFtab(4, 12) =", cane$f_tab, "\n")
cat("reject H0 for varieties?", cane$table$F[1] > cane$f_tab, "\n")
cane_fit <- check_with_aov(cane_y, cane_trt, cane)
print_tukey(cane, cane_fit)
cat("CV =", round(cane$cv, 2), "%\n")
cat("model: Y_ijk = m + t_i + l_j + c_k + e_ijk\n\n")

# --- Example: bacilli for yogurt, 5 x 5 square ---
# Five bacilli (A to E) make yogurt from 25 one-liter milk samples.
# The milk is not homogeneous: it varies in fat content and in acidity.
# Rows: fat content. Columns: acidity. Each bacillus meets each fat level
# and each acidity level once. Response: yogurt volume (ml).

yogurt_y <- matrix(
  c(450, 620, 680, 620, 780,
    750, 990, 750, 660, 830,
    750, 910, 690, 990, 760,
    650, 890, 835, 850, 875,
    750, 720, 850, 770, 890),
  nrow = 5, byrow = TRUE
)
yogurt_trt <- matrix(
  c("A", "E", "C", "D", "B",
    "C", "B", "E", "A", "D",
    "D", "C", "A", "B", "E",
    "E", "D", "B", "C", "A",
    "B", "A", "D", "E", "C"),
  nrow = 5, byrow = TRUE
)
stopifnot(
  rowSums(yogurt_y) == c(3150, 3980, 4100, 4100, 3980),
  colSums(yogurt_y) == c(3350, 4130, 3805, 3890, 4135)
)

yogurt <- latin_square_anova(yogurt_y, yogurt_trt)
stopifnot(yogurt$trt_totals == c(3395, 4345, 4080, 3940, 3550))
cat("DQL 5 x 5: yogurt bacilli, volume (ml)\n")
print(yogurt$table, row.names = FALSE, digits = 5)
cat("\nFtab(4, 12) =", yogurt$f_tab, "\n")
cat("reject H0 for bacilli?", yogurt$table$F[1] > yogurt$f_tab, "\n")
yogurt_fit <- check_with_aov(yogurt_y, yogurt_trt, yogurt)
print_tukey(yogurt, yogurt_fit)

# --- Example: read a given DQL table ---
# Four tire brands on four cars. Rows: cars. Columns: wheel position.
# Each brand goes once on each car and once in each position.
# GL for each source = 3, GLRes = (4 - 1)(4 - 2) = 6.
f_given <- c(brands = 12.4, cars = 15.7, position = 2.7)
f_tab_tires <- qf(1 - alpha, 3, 6)
cat("\nDQL 4 x 4: tire brands. Ftab(3, 6) =", f_tab_tires, "\n")
print(f_given > f_tab_tires)
