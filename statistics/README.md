# Statistics

Personal learning repo for statistics practice code (mostly R).

Concept notes live in the Obsidian **statistics vault**
(`obsidian-statistics-vault`). This repo holds runnable scripts only.

## Layout

```text
nonparametric-statistics/
  binomial_test.r
  chi-squared_test.r
  kolmogorov_smirnov_test.r
  lilliefors_test.r
  runs_test.r
```

Topic folders group related tests. One script file per procedure.

## Topics

| Topic | Scripts | Vault note |
|---|---|---|
| Binomial test | `nonparametric-statistics/binomial_test.r` | Statistical Inference → Binomial Test |
| Chi-squared test | `nonparametric-statistics/chi-squared_test.r` | Statistical Inference → Chi-Squared Test |
| Kolmogorov-Smirnov | `nonparametric-statistics/kolmogorov_smirnov_test.r` | Statistical Inference → Kolmogorov-Smirnov Test |
| Lilliefors | `nonparametric-statistics/lilliefors_test.r` | Statistical Inference → Lilliefors Test |
| Runs (iterações) | `nonparametric-statistics/runs_test.r` | Statistical Inference → Runs Test |

## Run

From this directory:

```sh
Rscript nonparametric-statistics/binomial_test.r
Rscript nonparametric-statistics/chi-squared_test.r
Rscript nonparametric-statistics/kolmogorov_smirnov_test.r
Rscript nonparametric-statistics/lilliefors_test.r
Rscript nonparametric-statistics/runs_test.r
```
