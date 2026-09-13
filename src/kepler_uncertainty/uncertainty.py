"""Propagate reported measurement errors through fitted predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kepler_uncertainty.data import UNCERTAINTY_COLUMNS, engineer_features

NON_NEGATIVE_FEATURES = {
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_steff",
    "koi_srad",
}


def symmetric_uncertainty(
    frame: pd.DataFrame,
    positive_error: str,
    negative_error: str,
) -> pd.Series:
    """Approximate one sigma by averaging the absolute upper and lower errors."""

    errors = pd.concat(
        [frame[positive_error].abs(), frame[negative_error].abs()], axis=1
    )
    return errors.mean(axis=1, skipna=True)


def perturb_measurements(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw one plausible measurement set from independent normal errors."""

    perturbed = frame.copy()
    for feature, (positive_error, negative_error) in UNCERTAINTY_COLUMNS.items():
        centre = pd.to_numeric(frame[feature], errors="coerce")
        sigma = symmetric_uncertainty(frame, positive_error, negative_error)
        noise = rng.normal(size=len(frame)) * sigma.fillna(0).to_numpy()
        values = centre.to_numpy() + noise
        if feature in NON_NEGATIVE_FEATURES:
            values = np.maximum(values, 0)
        perturbed[feature] = values
    return perturbed


def monte_carlo_predictions(
    model,
    raw_frame: pd.DataFrame,
    n_draws: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Repeat predictions after perturbing inputs within reported uncertainties."""

    rng = np.random.default_rng(random_state)
    draws = np.empty((n_draws, len(raw_frame)), dtype=float)
    for draw in range(n_draws):
        perturbed = perturb_measurements(raw_frame, rng)
        draws[draw] = model.predict_proba(engineer_features(perturbed))[:, 1]

    lower = np.quantile(draws, 0.05, axis=0)
    upper = np.quantile(draws, 0.95, axis=0)
    detail = pd.DataFrame(
        {
            "kepoi_name": raw_frame["kepoi_name"].to_numpy(),
            "median_candidate_probability": np.median(draws, axis=0),
            "lower_90pct": lower,
            "upper_90pct": upper,
            "interval_width": upper - lower,
            "probability_std": draws.std(axis=0, ddof=1),
        }
    ).sort_values("interval_width", ascending=False)
    return draws, detail

