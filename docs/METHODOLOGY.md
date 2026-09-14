# Methodology

## Objective

Estimate the probability that a Kepler Q1–Q17 DR25 Object of Interest has the pipeline disposition `CANDIDATE` rather than `FALSE POSITIVE`, then quantify predictive performance and sensitivity to reported measurement errors.

This is catalogue-label modelling. It is not physical confirmation or an independent replacement for Kepler vetting.

## Population and target

The fixed DR25 KOI table contains 8,054 signals: 4,034 candidates and 4,020 false positives. The positive class is `CANDIDATE`.

DR25 is preferred to the cumulative table because it is a uniform release. The NASA Exoplanet Archive warns that the cumulative table combines information from different deliveries and is not intended for statistical studies requiring a uniform population.

## Inputs and leakage policy

The 11 inputs are orbital period, impact parameter, transit duration, transit depth, candidate radius, equilibrium temperature, transit-model signal-to-noise, stellar effective temperature, stellar surface gravity, stellar radius, and Kepler magnitude.

The query excludes `koi_score`, the four `koi_fpflag_*` flags, `koi_disposition`, and `koi_comment` because they directly encode or explain the disposition. IDs are not model inputs.

## Feature preparation

Non-negative skewed quantities use `log(1 + x)`. Each candidate pipeline performs median imputation and adds missingness indicators using training data only. Logistic-regression inputs are then standardised. Gradient boosting uses imputed values without scaling.

## Grouped sample design

`StratifiedGroupKFold` creates five approximately balanced folds while keeping all KOIs with the same `kepid` together:

- folds 0–1: initial estimation;
- fold 2: model selection;
- fold 3: probability calibration;
- fold 4: final testing.

After selection, the chosen estimator is refitted on folds 0–2 and Platt-calibrated on fold 3. Fold 4 remains untouched until final evaluation.

## Candidate models

### Logistic regression

A regularised, standardised linear-log-odds baseline. It is transparent and stable but may miss non-linear structure.

### Histogram gradient boosting

A controlled non-linear challenger with learning rate 0.05, 200 boosting iterations, at most 12 leaves per tree, minimum 30 observations per leaf, and L2 regularisation. These constraints reduce—but do not eliminate—overfitting and complexity.

## Selection

Choose gradient boosting only when its selection-fold Gini exceeds logistic regression by at least 0.02 and its Brier score is no higher. Otherwise retain logistic regression. The rule is fixed in code and unit-tested.

## Validation

- grouped five-fold AUC on initial estimation data;
- ROC-AUC and Gini for ranking;
- Brier score and log loss for probability error;
- calibration intercept, slope, expected calibration error, and plot;
- candidate capture in the highest-scored 20%;
- 500-resample bootstrap intervals for AUC and Brier;
- five probability bins fixed from calibration data;
- permutation importance on the final test fold;
- performance slices by magnitude and signal-to-noise.

## Measurement uncertainty

For eight inputs with quoted upper/lower errors, one standard deviation is approximated as their mean absolute magnitude. One hundred independent normal perturbations are drawn in the original measurement space, non-negative physical quantities are clipped at zero, features are rebuilt, and probabilities are recalculated.

The output summarises each signal's 5th–95th percentile probability width. This is conditional sensitivity to reported random measurement error, not total predictive uncertainty.

## Reproducibility

- official TAP query and dataset DOI recorded;
- source row count, schema, labels, uniqueness, and missingness checked;
- fixed random seed;
- host-group isolation asserted;
- preprocessing inside model pipelines;
- generated reports and figures driven by the same metrics object;
- unit tests and GitHub Actions CI;
- raw data and fitted binary model excluded from version control.

## Limitations

The label is not ground truth; derived catalogue features are related to the vetting process; raw light curves and pixel diagnostics are absent; random grouped validation is not future-mission validation; imputation and independent normal perturbations are simplified; and no external catalogue is tested.
