"""Run the full Kepler classification and uncertainty analysis."""

# Environment variables must be set before importing Matplotlib or joblib.
# ruff: noqa: E402

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_CACHE = Path(__file__).resolve().parents[2] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from kepler_uncertainty.data import (
    DATA_DOI,
    MODEL_FEATURES,
    RAW_FEATURES,
    load_dataset,
    make_group_splits,
    target_values,
)
from kepler_uncertainty.metrics import (
    binary_metrics,
    bootstrap_interval,
    calibration_table,
    confidence_bin_table,
    confidence_boundaries,
    measurement_uncertainty_summary,
)
from kepler_uncertainty.models import (
    PlattCalibratedModel,
    calibration_parameters,
    candidate_models,
    challenger_clears_selection_hurdle,
)
from kepler_uncertainty.reporting import (
    plot_calibration,
    plot_confidence_bins,
    plot_feature_importance,
    plot_measurement_uncertainty,
    plot_model_comparison,
    plot_roc,
    write_json,
    write_model_card,
    write_validation_report,
)
from kepler_uncertainty.uncertainty import monte_carlo_predictions

RANDOM_STATE = 42


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _slice_metrics(
    outcomes: pd.Series,
    probabilities: np.ndarray,
    raw_test: pd.DataFrame,
) -> pd.DataFrame:
    frame = raw_test.copy()
    frame["outcome"] = outcomes.to_numpy()
    frame["probability"] = probabilities
    frame["magnitude_band"] = pd.cut(
        frame["koi_kepmag"],
        bins=[-np.inf, 14, 15, np.inf],
        labels=["bright_<=14", "mid_14_to_15", "faint_>15"],
    )
    frame["snr_band"] = pd.cut(
        frame["koi_model_snr"],
        bins=[-np.inf, 20, 100, np.inf],
        labels=["low_<20", "medium_20_to_100", "high_>100"],
    )

    rows: list[dict[str, float | int | str]] = []
    for dimension in ["magnitude_band", "snr_band"]:
        for group, subset in frame.groupby(dimension, observed=True):
            row: dict[str, float | int | str] = {
                "dimension": dimension,
                "group": str(group),
                "signals": int(len(subset)),
                "observed_candidate_fraction": float(subset["outcome"].mean()),
                "mean_predicted_probability": float(subset["probability"].mean()),
                "brier_score": float(
                    brier_score_loss(subset["outcome"], subset["probability"])
                ),
            }
            row["roc_auc"] = (
                float(roc_auc_score(subset["outcome"], subset["probability"]))
                if subset["outcome"].nunique() == 2
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    root = _project_root()
    raw_dir = root / "data" / "raw"
    artifact_dir = root / "artifacts"
    figure_dir = artifact_dir / "figures"
    table_dir = artifact_dir / "tables"
    model_dir = artifact_dir / "models"
    docs_dir = root / "docs"
    for directory in [figure_dir, table_dir, model_dir, docs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(raw_dir)
    splits = make_group_splits(frame, random_state=RANDOM_STATE)

    fitted_estimators = {}
    candidate_rows: list[dict[str, float | str]] = []
    cross_validation = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    for candidate in candidate_models(random_state=RANDOM_STATE):
        cv_scores = cross_val_score(
            candidate.estimator,
            splits.x_train,
            splits.y_train,
            groups=splits.groups_train,
            scoring="roc_auc",
            cv=cross_validation,
            n_jobs=1,
        )
        candidate.estimator.fit(splits.x_train, splits.y_train)
        selection_probabilities = candidate.estimator.predict_proba(splits.x_selection)[
            :, 1
        ]
        selection_metrics = binary_metrics(splits.y_selection, selection_probabilities)
        fitted_estimators[candidate.name] = candidate.estimator
        candidate_rows.append(
            {
                "model": candidate.name,
                "rationale": candidate.rationale,
                "cv_roc_auc_mean": float(cv_scores.mean()),
                "cv_roc_auc_std": float(cv_scores.std(ddof=1)),
                "selection_gini": selection_metrics["gini"],
                "selection_brier_score": selection_metrics["brier_score"],
            }
        )

    comparison = pd.DataFrame(candidate_rows)
    indexed = comparison.set_index("model")
    baseline = indexed.loc["logistic_regression"]
    challenger = indexed.loc["gradient_boosting"]
    challenger_clears_hurdle = challenger_clears_selection_hurdle(
        baseline_gini=baseline["selection_gini"],
        baseline_brier=baseline["selection_brier_score"],
        challenger_gini=challenger["selection_gini"],
        challenger_brier=challenger["selection_brier_score"],
    )
    selected_name = "gradient_boosting" if challenger_clears_hurdle else "logistic_regression"
    selected_estimator = fitted_estimators[selected_name]

    combined_x = pd.concat([splits.x_train, splits.x_selection])
    combined_y = pd.concat([splits.y_train, splits.y_selection])
    selected_estimator.fit(combined_x, combined_y)
    selected_model = PlattCalibratedModel(selected_estimator).fit_calibrator(
        splits.x_calibration, splits.y_calibration
    )

    test_probabilities = selected_model.predict_proba(splits.x_test)[:, 1]
    calibration_probabilities = selected_model.predict_proba(splits.x_calibration)[:, 1]
    test_metrics = binary_metrics(splits.y_test, test_probabilities)
    auc_interval = bootstrap_interval(
        splits.y_test,
        test_probabilities,
        roc_auc_score,
        n_resamples=500,
        random_state=RANDOM_STATE,
    )
    brier_interval = bootstrap_interval(
        splits.y_test,
        test_probabilities,
        brier_score_loss,
        n_resamples=500,
        random_state=RANDOM_STATE,
    )

    calibration = calibration_table(splits.y_test, test_probabilities)
    boundaries = confidence_boundaries(calibration_probabilities, n_bins=5)
    confidence = confidence_bin_table(
        splits.y_test, test_probabilities, boundaries
    )
    slices = _slice_metrics(splits.y_test, test_probabilities, splits.raw_test)

    draw_probabilities, uncertainty_detail = monte_carlo_predictions(
        selected_model,
        splits.raw_test,
        n_draws=100,
        random_state=RANDOM_STATE,
    )
    uncertainty_summary = measurement_uncertainty_summary(draw_probabilities)

    importance_result = permutation_importance(
        selected_model,
        splits.x_test,
        splits.y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean": importance_result.importances_mean,
            "importance_std": importance_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    missingness = (
        frame[RAW_FEATURES]
        .isna()
        .mean()
        .rename("missing_fraction")
        .reset_index(name="missing_fraction")
        .rename(columns={"index": "raw_feature"})
        .sort_values("missing_fraction", ascending=False)
    )

    comparison.to_csv(table_dir / "candidate_comparison.csv", index=False)
    calibration.to_csv(table_dir / "calibration_by_bin.csv", index=False)
    confidence.to_csv(table_dir / "confidence_bins.csv", index=False)
    slices.to_csv(table_dir / "slice_metrics.csv", index=False)
    importance.to_csv(table_dir / "permutation_importance.csv", index=False)
    missingness.to_csv(table_dir / "missingness.csv", index=False)
    uncertainty_detail.to_csv(table_dir / "measurement_uncertainty.csv", index=False)

    plot_model_comparison(comparison, figure_dir / "model_comparison.png")
    plot_roc(
        splits.y_test,
        test_probabilities,
        test_metrics["roc_auc"],
        figure_dir / "roc_curve.png",
    )
    plot_calibration(calibration, figure_dir / "calibration.png")
    plot_confidence_bins(confidence, figure_dir / "confidence_bins.png")
    plot_feature_importance(importance, figure_dir / "feature_importance.png")
    plot_measurement_uncertainty(
        uncertainty_detail, figure_dir / "measurement_uncertainty.png"
    )

    outcomes = target_values(frame)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "data": {
            "name": "NASA Kepler Q1-Q17 DR25 KOI table",
            "doi": DATA_DOI,
            "records": int(len(frame)),
            "unique_host_stars": int(frame["kepid"].nunique()),
            "candidate_fraction": float(outcomes.mean()),
            "model_features": MODEL_FEATURES,
            "leakage_fields_excluded": [
                "koi_score",
                "koi_fpflag_nt",
                "koi_fpflag_ss",
                "koi_fpflag_co",
                "koi_fpflag_ec",
                "koi_disposition",
                "koi_comment",
            ],
        },
        "sample_sizes": {
            "train": int(len(splits.x_train)),
            "selection": int(len(splits.x_selection)),
            "calibration": int(len(splits.x_calibration)),
            "test": int(len(splits.x_test)),
        },
        "selection_rule": {
            "description": (
                "Select gradient boosting only if selection Gini improves by at "
                "least 0.02 and selection Brier score does not worsen."
            ),
            "challenger_clears_hurdle": challenger_clears_hurdle,
        },
        "selected_model": selected_name,
        "selected_model_calibration": calibration_parameters(selected_model),
        "candidate_comparison": comparison.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "bootstrap_intervals": {
            "roc_auc": auc_interval,
            "brier_score": brier_interval,
        },
        "confidence_bin_internal_boundaries": [
            float(value) for value in boundaries[1:-1]
        ],
        "measurement_uncertainty": uncertainty_summary,
    }
    write_json(report, artifact_dir / "metrics.json")
    write_validation_report(report, docs_dir / "VALIDATION_REPORT.md")
    write_model_card(report, docs_dir / "MODEL_CARD.md")
    joblib.dump(selected_model, model_dir / "selected_model.joblib")

    print(
        f"Selected {selected_name}; test AUC={test_metrics['roc_auc']:.3f}; "
        f"Brier={test_metrics['brier_score']:.3f}."
    )
    print(f"Artefacts written to {artifact_dir}")


if __name__ == "__main__":
    main()
