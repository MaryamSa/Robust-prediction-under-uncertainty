import numpy as np
import pandas as pd

from kepler_uncertainty.uncertainty import perturb_measurements, symmetric_uncertainty


def test_symmetric_uncertainty_averages_absolute_errors() -> None:
    frame = pd.DataFrame({"up": [2.0], "down": [-4.0]})

    assert symmetric_uncertainty(frame, "up", "down").iloc[0] == 3.0


def test_perturbation_is_reproducible_and_non_negative() -> None:
    columns = {
        "koi_period": [1.0],
        "koi_period_err1": [10.0],
        "koi_period_err2": [-10.0],
    }
    for feature in [
        "koi_impact",
        "koi_duration",
        "koi_depth",
        "koi_prad",
        "koi_steff",
        "koi_slogg",
        "koi_srad",
    ]:
        columns[feature] = [1.0]
        columns[f"{feature}_err1"] = [0.0]
        columns[f"{feature}_err2"] = [0.0]
    frame = pd.DataFrame(columns)

    first = perturb_measurements(frame, np.random.default_rng(42))
    second = perturb_measurements(frame, np.random.default_rng(42))

    assert first.equals(second)
    assert first["koi_period"].iloc[0] >= 0

