# Statistics

Personal learning repo for statistics practice code (mostly R).

## Layout

```text
nonparametric-statistics/
  one-sample/
    binomial_test.r
    chi-squared_test.r
    kolmogorov_smirnov_test.r
    lilliefors_test.r
    runs_test.r
  two-independent-samples/
    fisher_exact_test.r
    mann_whitney_test.r
  two-related-samples/
    mcnemar_test.r
    sign_test.r
    wilcoxon_signed_rank_test.r
  k-related-samples/
    friedman_test.r
experimental-statistics/
  completely_randomized_design.r
  randomized_block_design.r
  latin_square_design.r
  tukey_test.r
  duncan_test.r
  student_newman_keuls_test.r
  dunnett_test.r
  t_test_contrasts.r
  scheffe_test.r
  regression_in_anova.r
```

Topic folders group related tests. One script file per procedure.

## Topics

One sample:

| Topic | Script |
|---|---|
| Binomial test | `nonparametric-statistics/one-sample/binomial_test.r` |
| Chi-squared test | `nonparametric-statistics/one-sample/chi-squared_test.r` |
| Kolmogorov-Smirnov | `nonparametric-statistics/one-sample/kolmogorov_smirnov_test.r` |
| Lilliefors | `nonparametric-statistics/one-sample/lilliefors_test.r` |
| Runs (iterações) | `nonparametric-statistics/one-sample/runs_test.r` |

Two independent samples. Chi-squared independence is the independence section of `one-sample/chi-squared_test.r`.

| Situation | Script |
|---|---|
| Two categorical variables, expected counts at least 5 | `nonparametric-statistics/one-sample/chi-squared_test.r` |
| 2x2 table with small counts | `nonparametric-statistics/two-independent-samples/fisher_exact_test.r` |
| Ordinal or numeric outcome | `nonparametric-statistics/two-independent-samples/mann_whitney_test.r` |

Two related samples:

| Situation | Script |
|---|---|
| Binary category before and after | `nonparametric-statistics/two-related-samples/mcnemar_test.r` |
| Paired numeric outcome, direction only | `nonparametric-statistics/two-related-samples/sign_test.r` |
| Paired numeric outcome, direction and size | `nonparametric-statistics/two-related-samples/wilcoxon_signed_rank_test.r` |

k related samples:

| Situation | Script |
|---|---|
| k treatments on the same blocks, ordinal or numeric outcome | `nonparametric-statistics/k-related-samples/friedman_test.r` |

## Experimental statistics

Each script works through examples. A comment above each example gives
its context. Each script checks its results against table values, hand
calculations or a built-in R function.

ANOVA designs:

| Design | Script | Examples |
|---|---|---|
| DIC, no blocks | `experimental-statistics/completely_randomized_design.r` | Car brands, time from 0 to 100 km/h |
| DBC, one blocking criterion | `experimental-statistics/randomized_block_design.r` | Two diets, age groups as blocks |
| DQL, rows and columns | `experimental-statistics/latin_square_design.r` | Sugarcane varieties, yogurt bacilli, tire brands |

Comparisons after the ANOVA:

| Goal | Script | Examples |
|---|---|---|
| All pairs, one critical value | `experimental-statistics/tukey_test.r` | Tomato, car brands, insecticides, potato, Tukey vs t |
| All pairs, critical value grows with the range | `experimental-statistics/duncan_test.r` | Car brands |
| All pairs, between Tukey and Duncan | `experimental-statistics/student_newman_keuls_test.r` | Car brands, all three tests side by side |
| Each treatment against a control | `experimental-statistics/dunnett_test.r` | Insecticides vs control, potato vs control |
| Contrasts, orthogonality (equal and unequal r), planned t test, DMS | `experimental-statistics/t_test_contrasts.r` | Pineapple intercropping, car brands by price |
| Any contrast, also after you see the data | `experimental-statistics/scheffe_test.r` | Car brands by price, Scheffe vs t, Scheffe vs Dunnett |

Quantitative treatments (doses):

| Goal | Script | Examples |
|---|---|---|
| Linear, quadratic and lack-of-fit F tests | `experimental-statistics/regression_in_anova.r` | Nitrogen doses (illustrative values) |

`dunnett_test.r` needs the `mvtnorm` package. Install it once:

```sh
Rscript -e 'install.packages("mvtnorm")'
```

## Run

From this directory:

```sh
Rscript nonparametric-statistics/one-sample/binomial_test.r
Rscript nonparametric-statistics/one-sample/chi-squared_test.r
Rscript nonparametric-statistics/one-sample/kolmogorov_smirnov_test.r
Rscript nonparametric-statistics/one-sample/lilliefors_test.r
Rscript nonparametric-statistics/one-sample/runs_test.r
Rscript nonparametric-statistics/two-independent-samples/fisher_exact_test.r
Rscript nonparametric-statistics/two-independent-samples/mann_whitney_test.r
Rscript nonparametric-statistics/two-related-samples/mcnemar_test.r
Rscript nonparametric-statistics/two-related-samples/sign_test.r
Rscript nonparametric-statistics/two-related-samples/wilcoxon_signed_rank_test.r
Rscript nonparametric-statistics/k-related-samples/friedman_test.r
Rscript experimental-statistics/completely_randomized_design.r
Rscript experimental-statistics/randomized_block_design.r
Rscript experimental-statistics/latin_square_design.r
Rscript experimental-statistics/tukey_test.r
Rscript experimental-statistics/duncan_test.r
Rscript experimental-statistics/student_newman_keuls_test.r
Rscript experimental-statistics/dunnett_test.r
Rscript experimental-statistics/t_test_contrasts.r
Rscript experimental-statistics/scheffe_test.r
Rscript experimental-statistics/regression_in_anova.r
```
