"""Validation metrics with plain, explicit definitions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Weighted predicted-versus-observed gap in equal-sized probability bins."""

    frame = pd.DataFrame(
        {"outcome": np.asarray(y_true), "probability": probabilities}
    )
    frame["bin"] = pd.qcut(frame["probability"], q=n_bins, duplicates="drop")
    summary = frame.groupby("bin", observed=True).agg(
        count=("outcome", "size"),
        observed_rate=("outcome", "mean"),
        predicted_rate=("probability", "mean"),
    )
    gaps = (summary["observed_rate"] - summary["predicted_rate"]).abs()
    return float(np.average(gaps, weights=summary["count"]))


def calibration_intercept_and_slope(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Fit observed outcomes against predicted log-odds."""

    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logit(clipped).reshape(-1, 1), np.asarray(y_true))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def positive_capture_at_fraction(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    fraction: float = 0.20,
) -> float:
    """Share of all positives contained in the highest-scored fraction."""

    outcomes = np.asarray(y_true)
    order = np.argsort(-probabilities)
    selected = order[: int(np.ceil(len(order) * fraction))]
    return float(outcomes[selected].sum() / outcomes.sum())


def binary_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute ranking, probability-error, and calibration diagnostics."""

    auc = roc_auc_score(y_true, probabilities)
    intercept, slope = calibration_intercept_and_slope(y_true, probabilities)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "candidate_capture_top_20pct": positive_capture_at_fraction(
            y_true, probabilities, fraction=0.20
        ),
    }


def bootstrap_interval(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 500,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Percentile bootstrap interval from repeated samples with replacement."""

    outcomes = np.asarray(y_true)
    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sample = rng.integers(0, len(outcomes), size=len(outcomes))
        if np.unique(outcomes[sample]).size < 2:
            continue
        estimates.append(float(metric(outcomes[sample], probabilities[sample])))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": float(metric(outcomes, probabilities)),
        "lower_95": float(low),
        "upper_95": float(high),
        "resamples": len(estimates),
    }


def calibration_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Observed versus predicted candidate fractions in probability bins."""

    frame = pd.DataFrame(
        {"outcome": np.asarray(y_true), "predicted_probability": probabilities}
    )
    frame["probability_bin"] = pd.qcut(
        frame["predicted_probability"], q=n_bins, labels=False, duplicates="drop"
    )
    result = (
        frame.groupby("probability_bin", observed=True)
        .agg(
            signals=("outcome", "size"),
            predicted_probability=("predicted_probability", "mean"),
            observed_candidate_fraction=("outcome", "mean"),
        )
        .reset_index()
    )
    result["probability_bin"] = result["probability_bin"] + 1
    return result


def confidence_boundaries(
    calibration_probabilities: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """Fix display-bin boundaries from calibration data before final testing."""

    internal = np.quantile(
        calibration_probabilities,
        np.linspace(0, 1, n_bins + 1)[1:-1],
    )
    return np.concatenate(([-np.inf], np.unique(internal), [np.inf]))


def confidence_bin_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    boundaries: np.ndarray,
) -> pd.DataFrame:
    """Summarise five fixed ranges from lowest to highest candidate probability."""

    labels = ["Very low", "Low", "Medium", "High", "Very high"]
    labels = labels[: len(boundaries) - 1]
    frame = pd.DataFrame(
        {"outcome": np.asarray(y_true), "predicted_probability": probabilities}
    )
    frame["confidence_bin"] = pd.cut(
        frame["predicted_probability"],
        bins=boundaries,
        labels=labels,
        include_lowest=True,
    )
    return (
        frame.groupby("confidence_bin", observed=True)
        .agg(
            signals=("outcome", "size"),
            min_probability=("predicted_probability", "min"),
            mean_probability=("predicted_probability", "mean"),
            max_probability=("predicted_probability", "max"),
            observed_candidate_fraction=("outcome", "mean"),
        )
        .reset_index()
    )


def measurement_uncertainty_summary(draw_probabilities: np.ndarray) -> dict[str, float | int]:
    """Summarise per-signal probability spread across Monte Carlo draws."""

    lower = np.quantile(draw_probabilities, 0.05, axis=0)
    upper = np.quantile(draw_probabilities, 0.95, axis=0)
    widths = upper - lower
    standard_deviations = draw_probabilities.std(axis=0, ddof=1)
    return {
        "draws": int(draw_probabilities.shape[0]),
        "signals": int(draw_probabilities.shape[1]),
        "median_probability_std": float(np.median(standard_deviations)),
        "p90_probability_std": float(np.quantile(standard_deviations, 0.90)),
        "median_90pct_interval_width": float(np.median(widths)),
        "p90_90pct_interval_width": float(np.quantile(widths, 0.90)),
        "fraction_interval_width_above_0_20": float(np.mean(widths > 0.20)),
    }
