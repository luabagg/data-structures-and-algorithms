# McNemar test
# Study guide: duas amostras relacionadas
#
# The same people are classified in two categories at two times.
# Only the discordant counts enter the test.
#
#            After A    After B
#   Before A     a          b
#   Before B     c          d
#
# b is the count that changed from A to B.
# c is the count that changed from B to A.
# a and d did not change.
#
# H0: no systematic change (b and c are equal in the population).
# H1: there is a systematic change.
#
# Fe = (b + c) / 2.
# If Fe >= 5, use the continuity correction from the course:
#   chi^2 = (|b - c| - 1)^2 / (b + c)
#   df = 1
#   Reject H0 when chi^2 > chi^2_table(alpha, 1).
#   For alpha = 0.05 the table value is 3.84.
# If Fe < 5, do not use chi-squared. Use a two-sided binomial test
# with n = b + c and p = 0.5.

alpha <- 0.05
chi_tab <- qchisq(1 - alpha, df = 1)

# --- class example: vote intention before and after a debate ---
# 4 people moved from candidate 1 to candidate 2.
# 21 people moved from candidate 2 to candidate 1.
# 63 stayed with candidate 1. 12 stayed with candidate 2.

vote <- matrix(
  c(63, 4,
    21, 12),
  nrow = 2,
  byrow = TRUE,
  dimnames = list(
    before = c("C1", "C2"),
    after = c("C1", "C2")
  )
)
b <- vote[1, 2]
c <- vote[2, 1]
fe <- (b + c) / 2
chi_vote <- (abs(b - c) - 1)^2 / (b + c)

cat("Vote intention, before x after\n")
print(vote)
cat("b =", b, " c =", c, " Fe =", fe, "\n")
cat("chi^2 =", chi_vote, " table =", chi_tab, "\n")
cat("reject H0 at alpha =", alpha, "?", chi_vote > chi_tab, "\n\n")

# mcnemar.test uses (|b - c| - 1)^2 / (b + c) when correct = TRUE.
# correct = FALSE drops the -1.
print(mcnemar.test(vote, correct = TRUE))

# --- study guide: b = 4, c = 12 ---
# Fe = 8 >= 5, so the chi-squared form applies.
# (|4 - 12| - 1)^2 / 16 = 49/16 = 3.0625, which is below 3.84.

b_guide <- 4
c_guide <- 12
fe_guide <- (b_guide + c_guide) / 2
chi_guide <- (abs(b_guide - c_guide) - 1)^2 / (b_guide + c_guide)
cat("\nStudy guide: b = 4, c = 12\n")
cat("Fe =", fe_guide, " chi^2 =", chi_guide, " table =", chi_tab, "\n")
cat("reject H0 at alpha =", alpha, "?", chi_guide > chi_tab, "\n")

# --- Fe < 5: exact binomial, not chi-squared ---
# Illustration of the rule. b = 1, c = 6, Fe = 3.5.
b_small <- 1
c_small <- 6
n_small <- b_small + c_small
fe_small <- n_small / 2
x_small <- min(b_small, c_small)
p_small <- 2 * pbinom(x_small, size = n_small, prob = 0.5)

cat("\nSmall discordant counts: b = 1, c = 6\n")
cat("Fe =", fe_small, "(below 5, so skip chi-squared)\n")
cat("X ~ Binomial(", n_small, ", 0.5), x =", x_small, "\n")
cat("two-sided p-value =", p_small, "\n")
cat("reject H0 at alpha =", alpha, "?", p_small < alpha, "\n")
print(binom.test(x_small, n_small, p = 0.5, alternative = "two.sided"))
