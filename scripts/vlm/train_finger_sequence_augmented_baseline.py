"""
Train finger-tapping score baselines from sequence-level 3D hand landmarks and
derived dynamics.

Uses the existing `finger_*_v2.pkl` assets, which already contain:
- raw 3D hand landmarks
- distance / velocity / acceleration channels
- wrist-normalized distances

This script derives additional summary features aligned with our severe-case
hypotheses:
- index-tip y trajectory / velocity / acceleration
- palm-center normalized distances
- beginning vs end peak ratio
- longest pause
- below-threshold cycle ratio
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "scripts" / "hpc" / "data"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "test_outputs" / "finger_sequence_augmented_baseline_report_v0_1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_ROOT / "finger_train_v2.pkl")
    parser.add_argument("--valid", type=Path, default=DATA_ROOT / "finger_valid_v2.pkl")
    parser.add_argument("--test", type=Path, default=DATA_ROOT / "finger_test_v2.pkl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clamp_round(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 4).astype(int)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "exact": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def idx_map(feature_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(den) > 1e-8 else 0.0


def summarize_sequence(seq: np.ndarray, fmap: dict[str, int]) -> dict[str, float]:
    def col(name: str) -> np.ndarray:
        return seq[:, fmap[name]]

    wrist = np.stack([col("wrist_x"), col("wrist_y"), col("wrist_z")], axis=1)
    thumb_tip = np.stack([col("thumb_tip_x"), col("thumb_tip_y"), col("thumb_tip_z")], axis=1)
    index_tip = np.stack([col("index_tip_x"), col("index_tip_y"), col("index_tip_z")], axis=1)
    index_mcp = np.stack([col("index_mcp_x"), col("index_mcp_y"), col("index_mcp_z")], axis=1)
    middle_mcp = np.stack([col("middle_mcp_x"), col("middle_mcp_y"), col("middle_mcp_z")], axis=1)
    ring_mcp = np.stack([col("ring_mcp_x"), col("ring_mcp_y"), col("ring_mcp_z")], axis=1)
    pinky_mcp = np.stack([col("pinky_mcp_x"), col("pinky_mcp_y"), col("pinky_mcp_z")], axis=1)

    palm_center = np.mean(np.stack([wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0), axis=0)

    finger_distance = col("finger_distance")
    norm_distance = col("normalized_distance")
    dist_velocity = col("dist_velocity")
    dist_accel = col("dist_accel")
    hand_size = col("hand_size")

    index_tip_y = index_tip[:, 1]
    index_tip_y_rel = index_tip_y - palm_center[:, 1]
    index_tip_y_vel = np.diff(index_tip_y_rel, prepend=index_tip_y_rel[0])
    index_tip_y_acc = np.diff(index_tip_y_vel, prepend=index_tip_y_vel[0])

    thumb_palm_dist = np.linalg.norm(thumb_tip - palm_center, axis=1)
    index_palm_dist = np.linalg.norm(index_tip - palm_center, axis=1)
    thumb_palm_norm = thumb_palm_dist / np.maximum(hand_size, 1e-8)
    index_palm_norm = index_palm_dist / np.maximum(hand_size, 1e-8)

    peaks, _ = find_peaks(norm_distance, distance=max(2, len(norm_distance) // 30))
    peak_vals = norm_distance[peaks] if len(peaks) else np.array([])
    half = max(1, len(peak_vals) // 2)
    begin_end_peak_ratio = safe_ratio(float(np.mean(peak_vals[-half:])) if len(peak_vals) else 0.0,
                                      float(np.mean(peak_vals[:half])) if len(peak_vals) else 0.0)

    if len(peaks) > 1:
        intervals = np.diff(peaks)
        longest_pause = float(np.max(intervals))
        pause_ratio = safe_ratio(longest_pause, float(np.mean(intervals)))
    else:
        longest_pause = float(len(norm_distance))
        pause_ratio = 0.0

    threshold = 0.4 * float(np.percentile(norm_distance, 90)) if len(norm_distance) else 0.0
    below_threshold_cycle_ratio = float(np.mean(norm_distance < threshold)) if threshold > 0 else 0.0

    out = {
        # existing dynamic summaries
        "finger_distance_mean": float(np.mean(finger_distance)),
        "finger_distance_std": float(np.std(finger_distance)),
        "norm_distance_mean": float(np.mean(norm_distance)),
        "norm_distance_std": float(np.std(norm_distance)),
        "dist_velocity_mean": float(np.mean(np.abs(dist_velocity))),
        "dist_velocity_std": float(np.std(dist_velocity)),
        "dist_accel_mean": float(np.mean(np.abs(dist_accel))),
        "dist_accel_std": float(np.std(dist_accel)),
        # y-axis motion
        "index_tip_y_rel_mean": float(np.mean(index_tip_y_rel)),
        "index_tip_y_rel_std": float(np.std(index_tip_y_rel)),
        "index_tip_y_vel_mean": float(np.mean(np.abs(index_tip_y_vel))),
        "index_tip_y_vel_std": float(np.std(index_tip_y_vel)),
        "index_tip_y_acc_mean": float(np.mean(np.abs(index_tip_y_acc))),
        "index_tip_y_acc_std": float(np.std(index_tip_y_acc)),
        # palm-normalized geometry
        "thumb_palm_norm_mean": float(np.mean(thumb_palm_norm)),
        "thumb_palm_norm_std": float(np.std(thumb_palm_norm)),
        "index_palm_norm_mean": float(np.mean(index_palm_norm)),
        "index_palm_norm_std": float(np.std(index_palm_norm)),
        # severe / interruption cues
        "peak_count": float(len(peaks)),
        "begin_end_peak_ratio": begin_end_peak_ratio,
        "longest_pause": longest_pause,
        "pause_ratio": pause_ratio,
        "below_threshold_cycle_ratio": below_threshold_cycle_ratio,
    }
    return out


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    X = np.asarray(obj["X"])
    y = np.asarray(obj["y"]).astype(int)
    feature_names = obj["features"]
    return X, y, feature_names


def build_frame_dataset(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    fmap = idx_map(feature_names)
    rows = [summarize_sequence(seq, fmap) for seq in X]
    keys = list(rows[0].keys())
    mat = np.array([[row[key] for key in keys] for row in rows], dtype=float)
    return mat, keys


def main() -> int:
    args = parse_args()
    X_train_seq, y_train, feature_names = load_dataset(args.train)
    X_valid_seq, y_valid, _ = load_dataset(args.valid)
    X_test_seq, y_test, _ = load_dataset(args.test)

    X_train, feature_keys = build_frame_dataset(X_train_seq, feature_names)
    X_valid, _ = build_frame_dataset(X_valid_seq, feature_names)
    X_test, _ = build_frame_dataset(X_test_seq, feature_names)

    models = {
        "extra_regressor": ExtraTreesRegressor(n_estimators=1200, random_state=42, min_samples_leaf=2),
        "rf_regressor": RandomForestRegressor(n_estimators=800, random_state=42, min_samples_leaf=2),
        "extra_classifier": ExtraTreesClassifier(n_estimators=1200, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2),
        "rf_classifier": RandomForestClassifier(n_estimators=800, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2),
    }

    report: dict[str, object] = {
        "n_features": len(feature_keys),
        "feature_keys": feature_keys,
        "models": {},
    }

    best_name = None
    best_valid_mae = None
    best_model = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        if "regressor" in name:
            valid_pred = clamp_round(model.predict(X_valid))
            test_pred = clamp_round(model.predict(X_test))
        else:
            valid_pred = model.predict(X_valid)
            test_pred = model.predict(X_test)

        valid_m = metrics(y_valid, valid_pred)
        test_m = metrics(y_test, test_pred)
        report["models"][name] = {"valid": valid_m, "test": test_m}
        if best_valid_mae is None or valid_m["mae"] < best_valid_mae:
            best_valid_mae = valid_m["mae"]
            best_name = name
            best_model = model

    report["best_model"] = best_name
    if best_name and hasattr(best_model, "feature_importances_"):
        report["best_model_feature_importance"] = dict(
            sorted(
                zip(feature_keys, best_model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
