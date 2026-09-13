"""Download, validate, transform, and split the public Kepler DR25 data."""

from __future__ import annotations

import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DATA_DOI = "https://doi.org/10.26133/NEA5"
TARGET_COLUMN = "koi_pdisposition"
POSITIVE_LABEL = "CANDIDATE"
NEGATIVE_LABEL = "FALSE POSITIVE"

RAW_FEATURES = [
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

UNCERTAINTY_COLUMNS = {
    "koi_period": ("koi_period_err1", "koi_period_err2"),
    "koi_impact": ("koi_impact_err1", "koi_impact_err2"),
    "koi_duration": ("koi_duration_err1", "koi_duration_err2"),
    "koi_depth": ("koi_depth_err1", "koi_depth_err2"),
    "koi_prad": ("koi_prad_err1", "koi_prad_err2"),
    "koi_steff": ("koi_steff_err1", "koi_steff_err2"),
    "koi_slogg": ("koi_slogg_err1", "koi_slogg_err2"),
    "koi_srad": ("koi_srad_err1", "koi_srad_err2"),
}

MODEL_FEATURES = [
    "log_period_days",
    "impact_parameter",
    "log_duration_hours",
    "log_transit_depth_ppm",
    "log_candidate_radius_earth",
    "log_equilibrium_temperature_k",
    "log_model_signal_to_noise",
    "stellar_temperature_k",
    "stellar_surface_gravity",
    "log_stellar_radius_solar",
    "kepler_magnitude",
]


def _download_columns() -> list[str]:
    columns = ["kepid", "kepoi_name", TARGET_COLUMN, *RAW_FEATURES]
    for error_columns in UNCERTAINTY_COLUMNS.values():
        columns.extend(error_columns)
    return list(dict.fromkeys(columns))


@dataclass(frozen=True)
class DataSplits:
    """Group-isolated samples for estimation, selection, calibration, and testing."""

    x_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series
    x_selection: pd.DataFrame
    y_selection: pd.Series
    x_calibration: pd.DataFrame
    y_calibration: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    raw_test: pd.DataFrame


def dataset_url() -> str:
    """Build the auditable NASA TAP query used by the analysis."""

    query = (
        f"select {','.join(_download_columns())} "
        "from q1_q17_dr25_koi order by kepoi_name"
    )
    return f"{TAP_URL}?{urllib.parse.urlencode({'query': query, 'format': 'csv'})}"


def load_dataset(cache_dir: Path) -> pd.DataFrame:
    """Load the fixed DR25 table, downloading and caching it on first run."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "kepler_dr25_koi.csv"
    if not csv_path.exists():
        request = urllib.request.Request(
            dataset_url(),
            headers={"User-Agent": "robust-kepler-classification/0.1"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
            csv_path.write_bytes(response.read())

    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip().lower() for column in frame.columns]
    validate_dataset(frame)
    return frame


def validate_dataset(frame: pd.DataFrame) -> None:
    """Fail loudly if NASA's fixed source or the local cache is not as expected."""

    required = {"kepid", "kepoi_name", TARGET_COLUMN, *_download_columns()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(frame) != 8_054:
        raise ValueError(f"Expected 8,054 DR25 rows, found {len(frame):,}")
    if frame["kepoi_name"].duplicated().any():
        raise ValueError("Each KOI name must be unique")
    labels = set(frame[TARGET_COLUMN].dropna().unique())
    if labels != {POSITIVE_LABEL, NEGATIVE_LABEL}:
        raise ValueError(f"Unexpected disposition labels: {sorted(labels)}")
    if frame[RAW_FEATURES].isna().mean().max() > 0.30:
        raise ValueError("At least one model feature exceeds 30% missing values")


def target_values(frame: pd.DataFrame) -> pd.Series:
    """Map NASA's disposition label to 1 for candidate and 0 for false positive."""

    return frame[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).astype(int)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent transforms; non-positive values become missing."""

    def safe_log1p(column: str) -> pd.Series:
        values = pd.to_numeric(frame[column], errors="coerce")
        return np.log1p(values.where(values >= 0))

    return pd.DataFrame(
        {
            "log_period_days": safe_log1p("koi_period"),
            "impact_parameter": pd.to_numeric(frame["koi_impact"], errors="coerce"),
            "log_duration_hours": safe_log1p("koi_duration"),
            "log_transit_depth_ppm": safe_log1p("koi_depth"),
            "log_candidate_radius_earth": safe_log1p("koi_prad"),
            "log_equilibrium_temperature_k": safe_log1p("koi_teq"),
            "log_model_signal_to_noise": safe_log1p("koi_model_snr"),
            "stellar_temperature_k": pd.to_numeric(frame["koi_steff"], errors="coerce"),
            "stellar_surface_gravity": pd.to_numeric(frame["koi_slogg"], errors="coerce"),
            "log_stellar_radius_solar": safe_log1p("koi_srad"),
            "kepler_magnitude": pd.to_numeric(frame["koi_kepmag"], errors="coerce"),
        },
        index=frame.index,
    )


def make_group_splits(frame: pd.DataFrame, random_state: int = 42) -> DataSplits:
    """Create fixed 40%/20%/20%/20% folds without sharing host stars."""

    outcomes = target_values(frame)
    groups = frame["kepid"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    fold = pd.Series(-1, index=frame.index, dtype=int)
    for fold_number, (_, held_out) in enumerate(
        splitter.split(frame, outcomes, groups=groups)
    ):
        fold.iloc[held_out] = fold_number
    if (fold < 0).any():
        raise RuntimeError("Every row must be assigned to a fold")

    train_index = fold.index[fold.isin([0, 1])]
    selection_index = fold.index[fold == 2]
    calibration_index = fold.index[fold == 3]
    test_index = fold.index[fold == 4]
    features = engineer_features(frame)

    group_sets = [
        set(groups.loc[index])
        for index in [train_index, selection_index, calibration_index, test_index]
    ]
    for left in range(len(group_sets)):
        for right in range(left + 1, len(group_sets)):
            if group_sets[left] & group_sets[right]:
                raise RuntimeError("A host star appears in more than one data split")

    return DataSplits(
        x_train=features.loc[train_index].copy(),
        y_train=outcomes.loc[train_index].copy(),
        groups_train=groups.loc[train_index].copy(),
        x_selection=features.loc[selection_index].copy(),
        y_selection=outcomes.loc[selection_index].copy(),
        x_calibration=features.loc[calibration_index].copy(),
        y_calibration=outcomes.loc[calibration_index].copy(),
        x_test=features.loc[test_index].copy(),
        y_test=outcomes.loc[test_index].copy(),
        raw_test=frame.loc[test_index].copy(),
    )
