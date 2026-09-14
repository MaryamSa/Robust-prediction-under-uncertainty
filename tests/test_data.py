import numpy as np
import pandas as pd

from kepler_uncertainty.data import (
    MODEL_FEATURES,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RAW_FEATURES,
    TARGET_COLUMN,
    engineer_features,
    make_group_splits,
    target_values,
)


def synthetic_catalogue(rows: int = 200) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "kepid": np.repeat(np.arange(rows // 2), 2),
            "kepoi_name": [f"K{i:05d}.01" for i in range(rows)],
            TARGET_COLUMN: np.tile([POSITIVE_LABEL, NEGATIVE_LABEL], rows // 2),
            **{
                feature: np.linspace(1, 10, rows)
                for feature in RAW_FEATURES
            },
        }
    )
    return frame


def test_feature_engineering_has_declared_columns() -> None:
    features = engineer_features(synthetic_catalogue())

    assert features.columns.tolist() == MODEL_FEATURES
    assert np.isfinite(features.to_numpy()).all()


def test_target_mapping_is_explicit() -> None:
    outcomes = target_values(synthetic_catalogue(10))

    assert outcomes.tolist() == [1, 0] * 5


def test_group_splits_never_share_host_stars() -> None:
    frame = synthetic_catalogue()
    splits = make_group_splits(frame, random_state=42)
    index_sets = [
        set(splits.x_train.index),
        set(splits.x_selection.index),
        set(splits.x_calibration.index),
        set(splits.x_test.index),
    ]

    assert len(set.union(*index_sets)) == len(frame)
    for left in range(len(index_sets)):
        for right in range(left + 1, len(index_sets)):
            left_hosts = set(frame.loc[list(index_sets[left]), "kepid"])
            right_hosts = set(frame.loc[list(index_sets[right]), "kepid"])
            assert left_hosts.isdisjoint(right_hosts)
