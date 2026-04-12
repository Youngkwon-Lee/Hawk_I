"""
Train and evaluate cheap finger-tapping score baselines from existing kinematic features.

Purpose:
- establish a feature-only reference point for the finger-tapping task
- compare whether kinematic features are already stronger than current VLM observation pipelines
- identify the most useful feature subset before building multimodal fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "data" / "processed" / "features"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "test_outputs" / "finger_feature_baseline_report_v0_1.json"
DEFAULT_MODEL_OUTPUT = ROOT / "experiments" / "results" / "test_outputs" / "finger_feature_baseline_best_v0_1.joblib"

FEATURE_SHORTLIST = [
    "tapping_speed",
    "amplitude_mean",
    "peak_velocity_mean",
    "amplitude_decrement",
    "rhythm_variability",
    "fatigue_rate",
    "amplitude_slope",
    "velocity_decrement",
    "hesitation_count",
    "halt_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=FEATURE_ROOT / "finger_tapping_train_features_stratified.csv")
    parser.add_argument("--valid", type=Path, default=FEATURE_ROOT / "finger_tapping_valid_features_stratified.csv")
    parser.add_argument("--test", type=Path, default=FEATURE_ROOT / "finger_tapping_test_features_stratified.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    return parser.parse_args()


def clamp_round(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 4).astype(int)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "exact": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> int:
    args = parse_args()
    train_df = pd.read_csv(args.train)
    valid_df = pd.read_csv(args.valid)
    test_df = pd.read_csv(args.test)

    y_train = train_df["score"].astype(int)
    y_valid = valid_df["score"].astype(int)
    y_test = test_df["score"].astype(int)

    feature_sets = {
        "shortlist": FEATURE_SHORTLIST,
        "all": [c for c in train_df.columns if c not in {"video_id", "subject", "score"}],
    }

    model_factories = {
        "logreg_balanced": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        ),
        "rf_classifier": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=42,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "rf_regressor_round": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "reg",
                    RandomForestRegressor(
                        n_estimators=500,
                        random_state=42,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "extra_classifier": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=800,
                        random_state=42,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "extra_regressor_round": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "reg",
                    ExtraTreesRegressor(
                        n_estimators=800,
                        random_state=42,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "xgb_classifier": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softmax",
                        num_class=5,
                        eval_metric="mlogloss",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgb_regressor_round": lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "reg",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    report: dict[str, object] = {
        "feature_sets": {name: cols for name, cols in feature_sets.items()},
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "score_distribution": {
            "train": train_df["score"].value_counts().sort_index().to_dict(),
            "valid": valid_df["score"].value_counts().sort_index().to_dict(),
            "test": test_df["score"].value_counts().sort_index().to_dict(),
        },
        "models": {},
    }

    best_name = None
    best_feature_set = None
    best_valid_mae = None
    fitted_models: dict[str, Pipeline] = {}

    for feature_set_name, feature_cols in feature_sets.items():
        X_train = train_df[feature_cols]
        X_valid = valid_df[feature_cols]
        X_test = test_df[feature_cols]
        for name, make_model in model_factories.items():
            model = make_model()
            full_name = f"{name}__{feature_set_name}"
            model.fit(X_train, y_train)
            fitted_models[full_name] = model

            if "regressor" in name:
                valid_pred = clamp_round(model.predict(X_valid))
                test_pred = clamp_round(model.predict(X_test))
            else:
                valid_pred = model.predict(X_valid)
                test_pred = model.predict(X_test)

            valid_metrics = metrics(y_valid, valid_pred)
            test_metrics = metrics(y_test, test_pred)
            report["models"][full_name] = {
                "n_features": len(feature_cols),
                "valid": valid_metrics,
                "test": test_metrics,
            }

            if best_valid_mae is None or valid_metrics["mae"] < best_valid_mae:
                best_valid_mae = valid_metrics["mae"]
                best_name = full_name
                best_feature_set = feature_set_name

    assert best_name is not None
    report["best_model"] = best_name
    report["best_feature_set"] = best_feature_set

    best_model = fitted_models[best_name]
    if best_name.startswith("rf_") or best_name.startswith("extra_"):
        estimator_name = "clf" if "classifier" in best_name else "reg"
        forest = best_model.named_steps[estimator_name]
        best_feature_columns = feature_sets[best_feature_set]
        importances = dict(
            sorted(
                zip(best_feature_columns, forest.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        report["best_model_feature_importance"] = importances

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_name,
            "feature_set": best_feature_set,
            "model": best_model,
            "feature_columns": feature_sets[best_feature_set],
        },
        args.model_output,
    )
    report["best_model_output"] = str(args.model_output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
