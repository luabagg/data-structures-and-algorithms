# Consider a population with N = 6 domiciles and the following characteristics:
# Table 1: values of the interest variable for each household of the population.
# | Domicile |   Wage | Number of persons | Number of workers |
# | :------- | -----: | ----------------: | ----------------: |
# | 1        |    800 |                 2 |                 2 |
# | 2        |  4,200 |                 4 |                 3 |
# | 3        |  1,600 |                 2 |                 1 |
# | 4        |    500 |                 2 |                 1 |
# | 5        |    900 |                 4 |                 2 |
# | 6        |  2,000 |                 1 |                 1 |
# | Total    | 10,000 |                15 |                10 |

# Question 1
# Interest variables: Wage, Number of persons, and Number of workers.
# For each interest variable, calculate the population parameters:
# total, mean, and variance.

cat("\nQuestion 1\n", sep = "")

population <- data.frame(
    domicile = 1:6,
    wage = c(800, 4200, 1600, 500, 900, 2000),
    number_of_persons = c(2, 4, 2, 2, 4, 1),
    number_of_workers = c(2, 3, 1, 1, 2, 1)
)

for (variable_name in names(population)[-1]) {
    observations <- population[[variable_name]]
    total <- sum(observations)
    population_mean <- mean(observations)
    variance <- stats::var(observations)

    cat("\n", variable_name, "\n", sep = "")
    cat("Total: ", total, "\n", sep = "")
    cat("Mean: ", population_mean, "\n", sep = "")
    cat("Variance: ", variance, "\n", sep = "")
}

# Question 2
# List S, the set of all possible samples of size 2.
# Consider only samples composed of distinct population units.

cat("\nQuestion 2\n", sep = "")
# combn = N! / (N - k)! k!
samples <- combn(1:6, 2)
print(samples)

# Question 3 (Plan A)
# Assuming every sample in S has the same probability of selection:
# a) Calculate the first-order inclusion probabilities for the population units.
# b) Calculate the second-order inclusion probabilities for all pairs of units.
# c) Calculate the possible values of the Horvitz-Thompson estimator of the
#    population total for Wage, Number of persons, and Number of workers.
# d) Calculate the expected value and variance of each Horvitz-Thompson
#    estimator from part (c).

cat("\nQuestion 3\n", sep = "")

N <- nrow(population)
sample_size <- 2
number_of_samples <- ncol(samples)
interest_variables <- names(population)[-1]

cat("\n(a)\n", sep = "")

# (a) In simple random sampling without replacement, every unit has pi_i = n / N.
inclusion_probabilities <- setNames(
    rep(sample_size / N, N),
    population$domicile
)
print(inclusion_probabilities)

cat("\n(b)\n", sep = "")

# (b) Every distinct pair has pi_ij = n(n - 1) / [N(N - 1)].
unit_pairs <- t(combn(population$domicile, 2))
pair_inclusion_probabilities <- data.frame(
    unit_i = unit_pairs[, 1],
    unit_j = unit_pairs[, 2],
    probability = rep(
        sample_size * (sample_size - 1) / (N * (N - 1)),
        nrow(unit_pairs)
    )
)
print(pair_inclusion_probabilities)

cat("\n(c)\n", sep = "")

# (c) Compute the Horvitz-Thompson total for every possible sample.
ht_estimates <- t(vapply(seq_len(number_of_samples), function(sample_index) {
    sampled_units <- samples[, sample_index]
    sampled_values <- population[sampled_units, interest_variables, drop = FALSE]
    colSums(sweep(
        sampled_values,
        1,
        inclusion_probabilities[as.character(sampled_units)],
        "/"
    ))
}, numeric(length(interest_variables))))
colnames(ht_estimates) <- interest_variables
rownames(ht_estimates) <- apply(samples, 2, paste, collapse = "-")
print(ht_estimates)

cat("\n(d)\n", sep = "")

# (d) Under Plan A, all samples have probability 1 / |S|.
ht_expected_values <- colMeans(ht_estimates)
ht_variances <- colMeans(sweep(ht_estimates, 2, ht_expected_values, "-")^2)
print(ht_expected_values)
print(ht_variances)

# Question 4 (Plan B)
# Let S = {(1, 2), (2, 3), (2, 4), (2, 5), (2, 6)}.
# Each sample in S has selection probability 1 / 5.
# Repeat Question 3 for Plan B:
# a) Calculate the first-order inclusion probabilities for the population units.
# b) Calculate the second-order inclusion probabilities for all pairs of units.
# c) Calculate the possible Horvitz-Thompson estimates of the population total
#    for Wage, Number of persons, and Number of workers.
# d) Calculate the expected value and variance of each estimator from part (c).

cat("\nQuestion 4\n", sep = "")
plan_b_samples <- rbind(
    c(1, 2),
    c(2, 3),
    c(2, 4),
    c(2, 5),
    c(2, 6)
)
plan_b_sample_probabilities <- rep(1 / nrow(plan_b_samples), nrow(plan_b_samples))

cat("\n(a)\n", sep = "")

# (a) pi_i is the sum of probabilities of Plan B samples that contain unit i.
plan_b_inclusion_probabilities <- sapply(population$domicile, function(unit) {
    selected <- apply(plan_b_samples, 1, function(sample) unit %in% sample)
    sum(plan_b_sample_probabilities[selected])
})
names(plan_b_inclusion_probabilities) <- population$domicile
print(plan_b_inclusion_probabilities)

cat("\n(b)\n", sep = "")

# (b) pi_ij is the sum of probabilities of Plan B samples that contain both units.
plan_b_pair_inclusion_probabilities <- apply(unit_pairs, 1, function(pair) {
    selected <- apply(plan_b_samples, 1, function(sample) all(pair %in% sample))
    sum(plan_b_sample_probabilities[selected])
})
plan_b_pair_inclusion_probabilities <- data.frame(
    unit_i = unit_pairs[, 1],
    unit_j = unit_pairs[, 2],
    probability = plan_b_pair_inclusion_probabilities
)
print(plan_b_pair_inclusion_probabilities)

cat("\n(c)\n", sep = "")

# (c) Compute the Horvitz-Thompson total for every allowed Plan B sample.
plan_b_ht_estimates <- t(vapply(seq_len(nrow(plan_b_samples)), function(sample_index) {
    sampled_units <- plan_b_samples[sample_index, ]
    sampled_values <- population[sampled_units, interest_variables, drop = FALSE]
    colSums(sweep(
        sampled_values,
        1,
        plan_b_inclusion_probabilities[as.character(sampled_units)],
        "/"
    ))
}, numeric(length(interest_variables))))
colnames(plan_b_ht_estimates) <- interest_variables
rownames(plan_b_ht_estimates) <- apply(plan_b_samples, 1, paste, collapse = "-")
print(plan_b_ht_estimates)

cat("\n(d)\n", sep = "")

# (d) E(T_hat) and Var(T_hat) under the Plan B sampling distribution.
plan_b_ht_expected_values <- colSums(
    sweep(plan_b_ht_estimates, 1, plan_b_sample_probabilities, "*")
)
plan_b_ht_variances <- colSums(sweep(
    sweep(plan_b_ht_estimates, 2, plan_b_ht_expected_values, "-")^2,
    1,
    plan_b_sample_probabilities,
    "*"
))
print(plan_b_ht_expected_values)
print(plan_b_ht_variances)

# Question 5
# Graph the possible values of the total estimators from Plan A and Plan B
# (Questions 3 and 4) to compare their sampling distributions.

cat("\nQuestion 5\n", sep = "")

# Aggregate identical HT estimates into their probability masses.
sampling_distribution <- function(estimates, sample_probabilities) {
    possible_estimates <- sort(unique(estimates))
    data.frame(
        estimate = possible_estimates,
        probability = sapply(possible_estimates, function(estimate) {
            sum(sample_probabilities[estimates == estimate])
        })
    )
}

plot_probability_mass <- function(distribution, plan_name, variable_name, x_limits, y_limits) {
    plot(
        distribution$estimate,
        distribution$probability,
        type = "h",
        lwd = 3,
        col = "steelblue",
        xlim = x_limits,
        ylim = y_limits,
        xlab = "Horvitz-Thompson total estimate",
        ylab = "Probability",
        main = paste(variable_name, "-", plan_name)
    )
    points(
        distribution$estimate,
        distribution$probability,
        pch = 16,
        col = "steelblue"
    )
}

# Use the same axis limits for both plans of each variable.
output_dir <- "out"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_file <- file.path(output_dir, "sampling-plan-distributions.pdf")
pdf(output_file, width = 10, height = 12)
par(mfrow = c(length(interest_variables), 2))
for (variable_name in interest_variables) {
    plan_a_distribution <- sampling_distribution(
        ht_estimates[, variable_name],
        rep(1 / nrow(ht_estimates), nrow(ht_estimates))
    )
    plan_b_distribution <- sampling_distribution(
        plan_b_ht_estimates[, variable_name],
        plan_b_sample_probabilities
    )
    x_limits <- range(
        plan_a_distribution$estimate,
        plan_b_distribution$estimate
    )
    y_limits <- c(
        0,
        max(plan_a_distribution$probability, plan_b_distribution$probability)
    )

    plot_probability_mass(
        plan_a_distribution,
        "Plan A",
        variable_name,
        x_limits,
        y_limits
    )
    plot_probability_mass(
        plan_b_distribution,
        "Plan B",
        variable_name,
        x_limits,
        y_limits
    )
}
par(mfrow = c(1, 1))
dev.off()
print(normalizePath(output_file))

# Question 6
# Compare Plans A and B using the results from Questions 3 and 4.
# For each interest variable, identify the preferred sampling plan for samples
# of size 2. Justify the choice by comparing estimator bias and variance.

cat("\nQuestion 6\n", sep = "")

for (variable_name in interest_variables) {
    plan_a_variance <- ht_variances[variable_name]
    plan_b_variance <- plan_b_ht_variances[variable_name]
    preferred_plan <- if (plan_a_variance < plan_b_variance) "Plan A" else "Plan B"

    cat("\n", variable_name, "\n", sep = "")
    cat("Expected total (both plans): ", ht_expected_values[variable_name], "\n", sep = "")
    cat("Plan A variance: ", plan_a_variance, "\n", sep = "")
    cat("Plan B variance: ", plan_b_variance, "\n", sep = "")
    cat("Preferred plan: ", preferred_plan, "\n", sep = "")
}