# Fisher exact test
# Study guide: duas amostras independentes
#
# Two categorical variables in a 2x2 table. Use this when expected counts
# are small and the chi-squared approximation is a poor fit.
# The chi-squared independence test is in chi-squared_test.r.
#
# H0: no association. Margins of the table stay fixed.
# H1: there is an association.
#
#         B1   B2
#   A1     a    b     a+b
#   A2     c    d     c+d
#         a+c  b+d     n
#
#   P = (a+b)! (c+d)! (a+c)! (b+d)! / (a! b! c! d! n!)
#     = C(a+b, a) * C(c+d, c) / C(n, a+c)
#
# Two-sided p-value: add the probabilities of every table with the same
# margins whose probability is less than or equal to the observed one.
# Reject H0 when p < alpha.

alpha <- 0.05

# Study-guide table. Several expected counts are below 5.
table_f <- matrix(
  c(1, 7,
    6, 2),
  nrow = 2,
  byrow = TRUE,
  dimnames = list(
    group = c("G1", "G2"),
    outcome = c("success", "failure")
  )
)

a <- table_f[1, 1]
b <- table_f[1, 2]
c <- table_f[2, 1]
d <- table_f[2, 2]
n <- a + b + c + d
row1 <- a + b
row2 <- c + d
col1 <- a + c
col2 <- b + d

expected <- outer(c(row1, row2), c(col1, col2)) / n
cat("Observed table\n")
print(table_f)
cat("Expected under independence (chi-squared would use these):\n")
print(expected)
cat("Some Fe are below 5, so do not use chi-squared for the decision.\n\n")

log_p_factorial <- function(aa, bb, cc, dd) {
  nn <- aa + bb + cc + dd
  lfactorial(aa + bb) + lfactorial(cc + dd) +
    lfactorial(aa + cc) + lfactorial(bb + dd) -
    lfactorial(aa) - lfactorial(bb) - lfactorial(cc) -
    lfactorial(dd) - lfactorial(nn)
}

p_factorial <- exp(log_p_factorial(a, b, c, d))
p_choose <- choose(row1, a) * choose(row2, c) / choose(n, col1)
cat("P(observed) from factorials =", p_factorial, "\n")
cat("P(observed) from binomial coefficients =", p_choose, "\n\n")

# Same margins: row totals 8 and 8, first-column total 7.
a_min <- max(0, row1 + col1 - n)
a_max <- min(row1, col1)
a_values <- a_min:a_max
probs <- vapply(a_values, function(aa) {
  exp(log_p_factorial(aa, row1 - aa, col1 - aa, n - row1 - col1 + aa))
}, numeric(1))
tables <- data.frame(a = a_values, probability = probs)
tables$as_extreme <- tables$probability <= p_factorial * (1 + 1e-8)

cat("All tables with these margins\n")
print(tables)
p_two <- sum(tables$probability[tables$as_extreme])
cat("\nTwo-sided p-value =", p_two, "\n")
cat("reject H0 at alpha =", alpha, "?", p_two < alpha, "\n\n")

exact <- fisher.test(table_f, alternative = "two.sided")
print(exact)
