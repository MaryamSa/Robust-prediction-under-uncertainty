# Model card

## Identity

- Name: Kepler Signals Under Uncertainty
- Version: 0.2.0
- Selected estimator: gradient boosting
- Owner: Maryam Saberi
- Status: educational portfolio model

## Intended use

Demonstrate reproducible binary classification, probability calibration, grouped validation, explainability, and measurement-uncertainty propagation with public astronomy data.

## Prohibited interpretation

Do not describe the output as a confirmed-planet probability. Do not use it to replace the Kepler Robovetter, scientific validation, follow-up observations, or expert review.

## Data

NASA Exoplanet Archive Kepler Q1–Q17 DR25 KOI table. The target is the pipeline disposition CANDIDATE versus FALSE POSITIVE. Features that directly encode the disposition are excluded.

## Performance snapshot

- Test ROC-AUC: 0.928
- Test Gini: 0.856
- Test Brier score: 0.104
- Expected calibration error: 0.020
- Calibration slope: 0.909

See [the validation report](VALIDATION_REPORT.md) for uncertainty intervals, measurement-error propagation, monitoring, and limitations.

## Controls demonstrated

- explicit objective, target, and prohibited claims;
- leakage-prone fields excluded before download;
- host-star-grouped folds;
- separate estimation, selection, calibration, and test data;
- pre-declared simple-versus-complex model hurdle;
- missing-data checks and transparent median imputation;
- ranking, probability, calibration, and uncertainty metrics;
- Monte Carlo propagation of quoted measurement errors;
- permutation importance and performance slices;
- deterministic seeds, tests, and generated artefacts.

## Review triggers in another domain

A real owner should set evidence-based tolerances for data quality, population mix, ranking, calibration, uncertainty, subgroup performance, operational impact, and model changes. Those limits depend on the domain, costs, governance, and law; they are not invented here.
