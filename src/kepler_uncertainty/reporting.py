"""Create reviewable figures and plain-language reports."""

# Long lines inside generated Markdown are intentional.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#255FA7"
LIGHT_BLUE = "#8FB7E5"
ORANGE = "#D97706"
GREY = "#5E6875"

FEATURE_LABELS = {
    "log_period_days": "Orbital period",
    "impact_parameter": "Impact parameter",
    "log_duration_hours": "Transit duration",
    "log_transit_depth_ppm": "Transit depth",
    "log_candidate_radius_earth": "Candidate radius",
    "log_equilibrium_temperature_k": "Equilibrium temperature",
    "log_model_signal_to_noise": "Transit-model signal-to-noise",
    "stellar_temperature_k": "Stellar temperature",
    "stellar_surface_gravity": "Stellar surface gravity",
    "log_stellar_radius_solar": "Stellar radius",
    "kepler_magnitude": "Kepler magnitude",
}


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.20)


def plot_roc(y_true: pd.Series, probabilities: np.ndarray, auc: float, path: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color=BLUE,
        linewidth=2.4,
        label=f"Selected model (AUC = {auc:.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color=GREY, label="Random ordering")
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="How well does the model rank planet-candidate signals?",
    )
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_calibration(table: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot([0, 1], [0, 1], linestyle="--", color=GREY, label="Perfect agreement")
    axis.plot(
        table["predicted_probability"],
        table["observed_candidate_fraction"],
        marker="o",
        color=BLUE,
        linewidth=2.2,
        label="Test bins",
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted candidate probability",
        ylabel="Observed candidate fraction",
        title="Do predicted probabilities match observed frequencies?",
    )
    axis.legend(frameon=False)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_confidence_bins(table: pd.DataFrame, path: Path) -> None:
    positions = np.arange(len(table))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.bar(
        positions - width / 2,
        table["mean_probability"],
        width,
        color=LIGHT_BLUE,
        label="Mean predicted probability",
    )
    axis.bar(
        positions + width / 2,
        table["observed_candidate_fraction"],
        width,
        color=BLUE,
        label="Observed candidate fraction",
    )
    axis.set_xticks(positions, table["confidence_bin"], rotation=12)
    axis.set(
        xlabel="Candidate-probability bin",
        ylabel="Fraction",
        ylim=(0, 1),
        title="Probability separation and calibration",
    )
    axis.legend(frameon=False)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_model_comparison(table: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for (_, row), color in zip(table.iterrows(), [BLUE, ORANGE], strict=False):
        axis.scatter(
            row["selection_brier_score"],
            row["selection_gini"],
            s=100,
            color=color,
        )
        axis.annotate(
            row["model"].replace("_", " "),
            (row["selection_brier_score"], row["selection_gini"]),
            xytext=(7, 7),
            textcoords="offset points",
        )
    axis.set(
        xlabel="Brier score (lower is better)",
        ylabel="Gini (higher is better)",
        title="Candidate comparison on selection data",
    )
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_feature_importance(table: pd.DataFrame, path: Path) -> None:
    display = table.head(10).sort_values("importance_mean")
    labels = [FEATURE_LABELS.get(name, name) for name in display["feature"]]
    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    axis.barh(labels, display["importance_mean"], color=BLUE)
    axis.errorbar(
        display["importance_mean"],
        labels,
        xerr=display["importance_std"],
        fmt="none",
        ecolor=GREY,
        capsize=2,
    )
    axis.set(
        xlabel="Drop in test AUC after shuffling the feature",
        title="Which measurements matter most for ranking?",
    )
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_measurement_uncertainty(detail: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.hist(detail["interval_width"], bins=35, color=BLUE, alpha=0.90)
    median = float(detail["interval_width"].median())
    axis.axvline(
        median,
        color=ORANGE,
        linestyle="--",
        linewidth=2,
        label=f"Median = {median:.3f}",
    )
    axis.set(
        xlabel="Width of 90% prediction interval from measurement errors",
        ylabel="Number of test signals",
        title="How much do reported measurement errors move predictions?",
    )
    axis.legend(frameon=False)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_validation_report(report: dict[str, Any], path: Path) -> None:
    metrics = report["test_metrics"]
    auc_ci = report["bootstrap_intervals"]["roc_auc"]
    brier_ci = report["bootstrap_intervals"]["brier_score"]
    uncertainty = report["measurement_uncertainty"]
    comparison = pd.DataFrame(report["candidate_comparison"])
    comparison_rows = "\n".join(
        "| {model} | {cv:.3f} ± {cv_std:.3f} | {gini:.3f} | {brier:.3f} |".format(
            model=row["model"].replace("_", " ").title(),
            cv=row["cv_roc_auc_mean"],
            cv_std=row["cv_roc_auc_std"],
            gini=row["selection_gini"],
            brier=row["selection_brier_score"],
        )
        for _, row in comparison.iterrows()
    )
    text = f"""# Validation report

Generated by `python -m kepler_uncertainty.run`. All final metrics use a held-out test fold whose host stars do not appear in estimation, selection, or calibration data.

## Executive conclusion

The pre-declared rule selected **{report['selected_model'].replace('_', ' ')}**. On the test fold it achieved ROC-AUC **{metrics['roc_auc']:.3f}**, Gini **{metrics['gini']:.3f}**, and Brier score **{metrics['brier_score']:.3f}**. This supports a portfolio demonstration of calibrated classification and robustness analysis. It does not establish an autonomous exoplanet-validation system or performance on future missions.

## Data and design

- Source: NASA Exoplanet Archive, Kepler Q1–Q17 DR25 KOI table, {report['data']['records']:,} signals.
- Target: Kepler pipeline disposition, CANDIDATE (1) versus FALSE POSITIVE (0); “candidate” does not mean confirmed planet.
- Inputs: 11 transit, candidate, stellar, and observation measurements.
- Leakage controls: the disposition score, disposition flags, comments, and final/archive disposition were not downloaded or used.
- Group isolation: all KOIs around the same host star remain in one fold.
- Split: {report['sample_sizes']['train']:,} estimation / {report['sample_sizes']['selection']:,} selection / {report['sample_sizes']['calibration']:,} calibration / {report['sample_sizes']['test']:,} test signals.

## Candidate decision

| Candidate | 5-fold grouped train AUC | Selection Gini | Selection Brier |
|---|---:|---:|---:|
{comparison_rows}

Gradient boosting is selected only if its selection Gini is at least 0.02 higher than logistic regression and its Brier score is no worse. The selected estimator is then refitted on estimation plus selection data and calibrated separately.

## Final test results

| Diagnostic | Result | Plain-language interpretation |
|---|---:|---|
| ROC-AUC | {metrics['roc_auc']:.3f} | Chance that a random candidate is ranked above a random false positive. |
| Gini | {metrics['gini']:.3f} | A rescaled AUC useful for cross-sector comparison. |
| Brier score | {metrics['brier_score']:.3f} | Mean squared probability error; lower is better. |
| Log loss | {metrics['log_loss']:.3f} | Penalises confident wrong probabilities; lower is better. |
| Calibration error | {metrics['expected_calibration_error']:.3f} | Average predicted-versus-observed gap across ten bins. |
| Calibration intercept | {metrics['calibration_intercept']:.3f} | Near 0 is desirable. |
| Calibration slope | {metrics['calibration_slope']:.3f} | Near 1 is desirable. |
| Candidates captured in top 20% | {metrics['candidate_capture_top_20pct']:.1%} | Share of all candidates in the highest-scored fifth. |

Bootstrap uncertainty (500 resamples): AUC **{auc_ci['estimate']:.3f}** (95% interval {auc_ci['lower_95']:.3f}–{auc_ci['upper_95']:.3f}); Brier **{brier_ci['estimate']:.3f}** (95% interval {brier_ci['lower_95']:.3f}–{brier_ci['upper_95']:.3f}).

## Measurement-error propagation

The pipeline makes {uncertainty['draws']} Monte Carlo draws from the quoted upper/lower errors for eight inputs. The median 90% prediction-interval width is **{uncertainty['median_90pct_interval_width']:.3f}**; the 90th percentile is **{uncertainty['p90_90pct_interval_width']:.3f}**. **{uncertainty['fraction_interval_width_above_0_20']:.1%}** of test signals have a width above 0.20.

This quantifies sensitivity to reported measurement errors under independent normal approximations. It does not include label uncertainty, systematic measurement error, model uncertainty, or correlations among fitted parameters.

## Monitoring slices

Results by Kepler magnitude and transit-model signal-to-noise are in `artifacts/tables/slice_metrics.csv`. Differences are diagnostic signals—not causal explanations—and should prompt review of sample size, missingness, measurement quality, and population shift.

## Main limitations

1. DR25 dispositions are pipeline labels for candidates and false positives, not ground-truth confirmation labels.
2. Several inputs are derived from the same transit fits used in scientific vetting, so this is not an independent discovery pipeline.
3. The model uses tabular summary measurements, not pixel data, light curves, centroid tests, or follow-up observations.
4. The fixed DR25 table allows group-isolated random validation but not a genuine future-mission or out-of-time test.
5. Median imputation, independent normal measurement errors, and the chosen model-selection hurdle are documented simplifications.
6. External validation on a separately constructed catalogue or mission is required before scientific use.

## Validation opinion

**Suitable as a general modelling portfolio project.** It demonstrates leakage control, entity-grouped validation, baseline/challenger governance, calibrated probabilities, uncertainty intervals, measurement-error propagation, explainability, monitoring slices, reproducibility, and honest limitations.
"""
    path.write_text(text, encoding="utf-8")

def write_model_card(report: dict[str, Any], path: Path) -> None:
    metrics = report["test_metrics"]
    text = f"""# Model card

## Identity

- Name: Kepler Signals Under Uncertainty
- Version: 0.2.0
- Selected estimator: {report['selected_model'].replace('_', ' ')}
- Owner: Maryam Saberi
- Status: educational portfolio model

## Intended use

Demonstrate reproducible binary classification, probability calibration, grouped validation, explainability, and measurement-uncertainty propagation with public astronomy data.

## Prohibited interpretation

Do not describe the output as a confirmed-planet probability. Do not use it to replace the Kepler Robovetter, scientific validation, follow-up observations, or expert review.

## Data

NASA Exoplanet Archive Kepler Q1–Q17 DR25 KOI table. The target is the pipeline disposition CANDIDATE versus FALSE POSITIVE. Features that directly encode the disposition are excluded.

## Performance snapshot

- Test ROC-AUC: {metrics['roc_auc']:.3f}
- Test Gini: {metrics['gini']:.3f}
- Test Brier score: {metrics['brier_score']:.3f}
- Expected calibration error: {metrics['expected_calibration_error']:.3f}
- Calibration slope: {metrics['calibration_slope']:.3f}

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
"""
    path.write_text(text, encoding="utf-8")
