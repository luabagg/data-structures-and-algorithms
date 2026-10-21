# Dunnett test (teste de Dunnett)
#
# Use Dunnett when one treatment is a control (testemunha) and you only
# compare each other treatment with the control. It makes I - 1
# comparisons, fewer than Tukey's I(I-1)/2. So its critical value is
# smaller and it finds differences from the control more easily.
#
# H0: m_i = m_control.  H1: m_i != m_control (two-sided).
#
# 1. Y_i = mean_i - mean_control.
# 2. s(Y) = sqrt(2 * QMRes / r).
# 3. Find td in the Dunnett table: columns = I - 1 (treatments without
#    the control), rows = GLRes.
# 4. d = td * s(Y).
# 5. The treatment differs from the control when |Y_i| > d.
#
# td is the two-sided 1 - alpha quantile of max|T_i|. T_1, ..., T_(I-1)
# follow a multivariate t with GLRes degrees of freedom. Each pair of
# comparisons shares the control mean, so with equal r their correlation
# is 0.5. Base R cannot compute this quantile. mvtnorm::qmvt can.
# Install once: install.packages("mvtnorm")

alpha <- 0.05

# qmvt integrates by quasi-Monte Carlo. Fix the seed so the output repeats.
set.seed(2026)

dunnett_td <- function(n_compared, gl_res) {
  corr <- matrix(0.5, n_compared, n_compared)
  diag(corr) <- 1
  mvtnorm::qmvt(
    1 - alpha,
    tail = "both.tails", df = gl_res, corr = corr,
    algorithm = mvtnorm::GenzBretz(abseps = 1e-5)
  )$quantile
}

# Dunnett table, 5%, two-sided: I - 1 = 5, GLRes = 15 gives td = 2.82.
# With one comparison Dunnett is the plain t test: td = t(GLRes).
stopifnot(
  abs(dunnett_td(5, 15) - 2.82) < 0.01,
  abs(dunnett_td(1, 15) - qt(1 - alpha / 2, 15)) < 0.01
)

# --- Example: insecticides against mealybugs (DBC) ---
# Five insecticides and a control without insecticide (F). Does each
# insecticide leave fewer live mealybugs than the control?
# Response: live mealybugs per plot. Rows: blocks I to IV.
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
fit <- aov(y ~ trt + block, data = bugs_long)
gl_res <- df.residual(fit)
qm_res <- deviance(fit) / gl_res
r <- nrow(bugs)

means <- colMeans(bugs)
control <- "F"
others <- setdiff(names(means), control)
y_hat <- means[others] - means[control]
s_y <- sqrt(2 * qm_res / r)
td <- dunnett_td(length(others), gl_res)
d <- td * s_y

cat("Insecticides: Dunnett, control = F\n")
cat("QMRes =", qm_res, " GLRes =", gl_res, " s(Y) =", s_y, "\n")
cat("td(", length(others), ",", gl_res, ") =", td, " d =", d, "\n")
print(data.frame(
  trt = others, Y = unname(y_hat), differ = unname(abs(y_hat) > d)
), row.names = FALSE, digits = 4)

# --- Example: potato cultivars, one comparison with the control ---
# Ten cultivars in a DBC with 4 blocks, SQRes = 572.9369,
# GLRes = 9 * 3 = 27. Achat is the control. Compare Matilda with it.
# td uses all 9 comparisons with the control, even if you only test one.
# The printed table has rows 24 (2.90) and 30 (2.86), not 27.
potato_qm_res <- 572.9369 / 27
potato_td <- dunnett_td(9, 27)
potato_y <- 30.73 - 41.63
potato_d <- potato_td * sqrt(2 * potato_qm_res / 4)
stopifnot(potato_td < 2.90, potato_td > 2.86)

cat("\nPotatoes: Matilda - Achat =", potato_y, "\n")
cat("td(9, 27) =", potato_td, " d =", potato_d, "\n")
cat("Matilda differs from Achat?", abs(potato_y) > potato_d, "\n")
