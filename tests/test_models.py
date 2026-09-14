import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kepler_uncertainty.models import (
    PlattCalibratedModel,
    challenger_clears_selection_hurdle,
)


def test_platt_calibrated_probabilities_are_valid() -> None:
    x_train = pd.DataFrame({"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    estimator = LogisticRegression().fit(x_train, y_train)
    calibrated = PlattCalibratedModel(estimator).fit_calibrator(x_train, y_train)
    probabilities = calibrated.predict_proba(x_train)

    assert probabilities.shape == (6, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_challenger_must_improve_gini_without_worsening_brier() -> None:
    assert challenger_clears_selection_hurdle(0.40, 0.15, 0.43, 0.14)
    assert not challenger_clears_selection_hurdle(0.40, 0.15, 0.41, 0.14)
    assert not challenger_clears_selection_hurdle(0.40, 0.15, 0.43, 0.16)
