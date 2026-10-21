# Randomized block design (DBC, delineamento em blocos casualizados)
#
# Use DBC when one known source of variation splits the units into
# homogeneous groups (blocks). Each block gets every treatment once.
# Treatments are randomized inside each block.
# Principles: repetition, randomization and local control (controle local).
#
# Model: Y_ij = m + t_i + b_j + e_ij
#   t_i = effect of treatment i, b_j = effect of block j.
#
# H0: m_1 = m_2 = ... = m_I.  H1: at least one mean is different.
#
# 1. G = sum of all observations. C = G^2 / (I*J).
# 2. SQTotal = sum(Y^2) - C.
# 3. SQTrat = sum(T_i^2 / J) - C. Each treatment total has J values.
# 4. SQBlocos = sum(B_j^2 / I) - C. Each block total has I values.
# 5. SQRes = SQTotal - SQTrat - SQBlocos.
# 6. GLTrat = I - 1. GLBlocos = J - 1. GLRes = (I - 1)(J - 1).
# 7. F = QMTrat / QMRes. Reject H0 when F > Ftab(GLTrat, GLRes, alpha).

alpha <- 0.05

# Example: a nutritionist compares two diets (A and B) for weight loss.
# Eight volunteers come from four age groups, two in each group. Age can
# change the weight loss, so each age group is a block. In each block,
# one person gets diet A and the other gets diet B, at random.
# Response: weight loss (kg).
weight_loss <- matrix(
  c(3, 7, 14, 8,
    7, 13, 22, 14),
  nrow = 2,
  byrow = TRUE,
  dimnames = list(
    diet = c("A", "B"),
    age = c("child", "teen", "adult", "elderly")
  )
)

n_trt <- nrow(weight_loss)
n_blk <- ncol(weight_loss)
trt_totals <- rowSums(weight_loss)
blk_totals <- colSums(weight_loss)

# Check the data against the known row and column totals.
stopifnot(trt_totals == c(32, 56), blk_totals == c(10, 20, 36, 22))

g <- sum(weight_loss)
c_corr <- g^2 / (n_trt * n_blk)
sq_total <- sum(weight_loss^2) - c_corr
sq_trat <- sum(trt_totals^2 / n_blk) - c_corr
sq_blk <- sum(blk_totals^2 / n_trt) - c_corr
sq_res <- sq_total - sq_trat - sq_blk

gl_trat <- n_trt - 1
gl_blk <- n_blk - 1
gl_res <- gl_trat * gl_blk
gl_total <- n_trt * n_blk - 1

qm_trat <- sq_trat / gl_trat
qm_blk <- sq_blk / gl_blk
qm_res <- sq_res / gl_res
f_trat <- qm_trat / qm_res
f_blk <- qm_blk / qm_res
f_tab <- qf(1 - alpha, gl_trat, gl_res)

cat("DBC: 2 diets x 4 age blocks, weight loss (kg)\n")
print(weight_loss)
cat("\nG =", g, " C =", c_corr, "\n\n")
print(data.frame(
  FV = c("Diets", "Blocks", "Residual", "Total"),
  GL = c(gl_trat, gl_blk, gl_res, gl_total),
  SQ = c(sq_trat, sq_blk, sq_res, sq_total),
  QM = c(qm_trat, qm_blk, qm_res, NA),
  F = c(f_trat, f_blk, NA, NA)
), row.names = FALSE, digits = 5)
cat("\nFtab(", gl_trat, ",", gl_res, ") =", f_tab, "\n")
cat("reject H0 for diets at alpha =", alpha, "?", f_trat > f_tab, "\n")
cat("diet means:", paste(names(trt_totals), trt_totals / n_blk), "\n")
cat("CV =", round(sqrt(qm_res) / mean(weight_loss) * 100, 2), "%\n")

# Built-in check. Block enters the model as a second factor.
long <- data.frame(
  y = as.vector(weight_loss),
  diet = factor(rep(rownames(weight_loss), times = n_blk)),
  age = factor(rep(colnames(weight_loss), each = n_trt))
)
fit <- aov(y ~ diet + age, data = long)
print(summary(fit))
stopifnot(isTRUE(all.equal(
  unname(summary(fit)[[1]][["Sum Sq"]]),
  c(sq_trat, sq_blk, sq_res)
)))

# --- why block? ---
# Analyze the same data as a DIC. The block variation goes back into the
# residual. QMRes gets larger and the diet effect is no longer significant.
sq_res_dic <- sq_total - sq_trat
gl_res_dic <- n_trt * n_blk - n_trt
f_dic <- qm_trat / (sq_res_dic / gl_res_dic)
f_tab_dic <- qf(1 - alpha, gl_trat, gl_res_dic)
cat("\nSame data as DIC: F =", f_dic, " Ftab =", f_tab_dic,
    " reject H0?", f_dic > f_tab_dic, "\n")
