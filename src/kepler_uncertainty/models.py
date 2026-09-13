"""Candidate models and probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PlattCalibratedModel(ClassifierMixin, BaseEstimator):
    """Rescale a fitted classifier's probabilities on separate data."""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.calibrator = LogisticRegression(C=1e6, solver="lbfgs")

    @staticmethod
    def _raw_log_odds(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return logit(clipped).reshape(-1, 1)

    def fit_calibrator(self, x: pd.DataFrame, y: pd.Series) -> "PlattCalibratedModel":
        raw_probabilities = self.estimator.predict_proba(x)[:, 1]
        self.calibrator.fit(self._raw_log_odds(raw_probabilities), y)
        self.classes_ = np.asarray(self.estimator.classes_)
        return self

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "PlattCalibratedModel":
        """Compatibility method; the main analysis calibrates on separate data."""

        self.estimator.fit(x, y)
        return self.fit_calibrator(x, y)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw_probabilities = self.estimator.predict_proba(x)[:, 1]
        calibrated = self.calibrator.predict_proba(
            self._raw_log_odds(raw_probabilities)
        )[:, 1]
        return np.column_stack([1 - calibrated, calibrated])


@dataclass(frozen=True)
class Candidate:
    """A named candidate and the reason it is included."""

    name: str
    estimator: BaseEstimator
    rationale: str


def candidate_models(random_state: int = 42) -> list[Candidate]:
    """Return a transparent baseline and a controlled non-linear challenger."""

    logistic = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=2_000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
    challenger = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=12,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return [
        Candidate(
            name="logistic_regression",
            estimator=logistic,
            rationale=(
                "Transparent baseline with reviewable coefficient directions and a "
                "simple mathematical form."
            ),
        ),
        Candidate(
            name="gradient_boosting",
            estimator=challenger,
            rationale=(
                "Non-linear challenger that can learn interactions and thresholds but "
                "must earn its added complexity."
            ),
        ),
    ]


def challenger_clears_selection_hurdle(
    baseline_gini: float,
    baseline_brier: float,
    challenger_gini: float,
    challenger_brier: float,
    minimum_gini_gain: float = 0.02,
) -> bool:
    """Apply the documented complexity hurdle without hidden judgement."""

    return bool(
        challenger_gini - baseline_gini >= minimum_gini_gain
        and challenger_brier <= baseline_brier
    )


def calibration_parameters(model: PlattCalibratedModel) -> dict[str, float]:
    """Return the fitted intercept and slope used to rescale probabilities."""

    return {
        "intercept": float(model.calibrator.intercept_[0]),
        "slope": float(model.calibrator.coef_[0, 0]),
    }
