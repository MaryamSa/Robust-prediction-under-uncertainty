import numpy as np

from kepler_uncertainty.metrics import (
    binary_metrics,
    confidence_bin_table,
    confidence_boundaries,
    expected_calibration_error,
    measurement_uncertainty_summary,
)


def test_binary_metrics_rank_perfectly() -> None:
    outcomes = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.70, 0.90])
    metrics = binary_metrics(outcomes, probabilities)

    assert metrics["roc_auc"] == 1.0
    assert metrics["gini"] == 1.0
    assert 0 <= metrics["brier_score"] <= 1


def test_expected_calibration_error_matches_two_groups() -> None:
    outcomes = np.array([0] * 90 + [1] * 10 + [0] * 20 + [1] * 80)
    probabilities = np.array([0.10] * 100 + [0.80] * 100)

    assert expected_calibration_error(outcomes, probabilities, n_bins=2) < 1e-12


def test_confidence_boundaries_are_fixed_before_test_summary() -> None:
    calibration_probabilities = np.linspace(0.01, 0.99, 500)
    boundaries = confidence_boundaries(calibration_probabilities, n_bins=5)
    test_probabilities = np.array([0.02, 0.20, 0.50, 0.80, 0.98])
    outcomes = np.array([0, 0, 1, 1, 1])
    table = confidence_bin_table(outcomes, test_probabilities, boundaries)

    assert boundaries[0] == -np.inf
    assert boundaries[-1] == np.inf
    assert table["signals"].sum() == len(outcomes)


def test_measurement_summary_uses_probability_spread() -> None:
    draws = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    summary = measurement_uncertainty_summary(draws)

    assert summary["draws"] == 3
    assert summary["signals"] == 2
    assert summary["median_90pct_interval_width"] > 0

