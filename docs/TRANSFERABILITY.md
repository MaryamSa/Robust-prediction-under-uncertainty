# Transferability across domains

The reusable asset is the reasoning pattern, not an astronomy model transplanted into another sector.

## What transfers

| Capability | Transferable question |
|---|---|
| Target definition | What exactly is the event, over what horizon, and how reliable is the label? |
| Leakage control | Does any feature contain the answer or information unavailable at prediction time? |
| Entity grouping | Could related records appear on both sides of a data split? |
| Baseline/challenger comparison | Does added model complexity produce material, reproducible value? |
| Calibration | Do predicted probabilities agree with observed event frequencies? |
| Multiple validation metrics | Can the model rank cases, estimate probabilities, and remain stable? |
| Bootstrap intervals | How uncertain are reported performance estimates? |
| Input-error propagation | How do uncertain measurements, estimates, or forecasts affect outputs? |
| Permutation importance | Which inputs support predictive ranking, without claiming causation? |
| Slice monitoring | Where does performance differ across data quality or operating regimes? |
| Decision logging | Can a reviewer see what was chosen, why, alternatives, and remaining risk? |

## Banking and insurance examples

The binary label could become default, fraud, churn, or claim. Host-star grouping becomes customer, household, or company grouping. Probability calibration, Gini, Brier score, leakage checks, challenger governance, metric uncertainty, and segment monitoring remain useful.

What does **not** transfer automatically: regulatory definitions, outcome horizons, capital or pricing logic, protected-attribute requirements, economic-cycle treatment, costs, approval processes, and customer impact. This project is evidence of modelling method, not banking experience.

## Energy and industrial examples

The label could become equipment failure, outage, curtailment, threshold exceedance, or anomalous operation. Reported astronomy errors become sensor uncertainty, weather-forecast uncertainty, or uncertain engineering parameters. Host-star grouping becomes asset, site, grid region, or forecast initialization.

What does **not** transfer automatically: temporal dependence, spatial correlation, physical constraints, seasonality, dispatch economics, safety thresholds, or operational response. Those require time-aware validation and domain-specific models.

## Other applications

The same structure applies to medical screening, quality control, remote sensing, and environmental monitoring when the positive label, entity grouping, uncertainty model, costs, and ethical constraints are explicitly redefined.

## Interview-safe description

> I built a reproducible classification pipeline using NASA Kepler DR25 data. I compared logistic regression with gradient boosting, prevented host-star leakage through grouped validation, calibrated the selected probabilities, quantified metric uncertainty with bootstrapping, and propagated reported measurement errors with Monte Carlo sampling. The astronomy target is specific, but the workflow—clear target definition, leakage control, calibration, robustness, monitoring, and documented model choice—is transferable to risk, energy, and other probabilistic modelling problems.

