# Statistics

Personal learning repo for statistics practice code (mostly R).

## Layout

```text
nonparametric-statistics/
  binomial_test.r
  chi-squared_test.r
  kolmogorov_smirnov_test.r
  lilliefors_test.r
  runs_test.r
  fisher_exact_test.r
  mann_whitney_test.r
  mcnemar_test.r
  sign_test.r
  wilcoxon_signed_rank_test.r
```

Topic folders group related tests. One script file per procedure.

## Topics

One sample:

| Topic | Script |
|---|---|
| Binomial test | `nonparametric-statistics/binomial_test.r` |
| Chi-squared test | `nonparametric-statistics/chi-squared_test.r` |
| Kolmogorov-Smirnov | `nonparametric-statistics/kolmogorov_smirnov_test.r` |
| Lilliefors | `nonparametric-statistics/lilliefors_test.r` |
| Runs (iterações) | `nonparametric-statistics/runs_test.r` |

Two independent samples. Chi-squared independence is the independence section of `chi-squared_test.r`.

| Situation | Script |
|---|---|
| Two categorical variables, expected counts at least 5 | `nonparametric-statistics/chi-squared_test.r` |
| 2x2 table with small counts | `nonparametric-statistics/fisher_exact_test.r` |
| Ordinal or numeric outcome | `nonparametric-statistics/mann_whitney_test.r` |

Two related samples:

| Situation | Script |
|---|---|
| Binary category before and after | `nonparametric-statistics/mcnemar_test.r` |
| Paired numeric outcome, direction only | `nonparametric-statistics/sign_test.r` |
| Paired numeric outcome, direction and size | `nonparametric-statistics/wilcoxon_signed_rank_test.r` |

## Run

From this directory:

```sh
Rscript nonparametric-statistics/binomial_test.r
Rscript nonparametric-statistics/chi-squared_test.r
Rscript nonparametric-statistics/kolmogorov_smirnov_test.r
Rscript nonparametric-statistics/lilliefors_test.r
Rscript nonparametric-statistics/runs_test.r
Rscript nonparametric-statistics/fisher_exact_test.r
Rscript nonparametric-statistics/mann_whitney_test.r
Rscript nonparametric-statistics/mcnemar_test.r
Rscript nonparametric-statistics/sign_test.r
Rscript nonparametric-statistics/wilcoxon_signed_rank_test.r
```
