"""
Evaluate the finger performability gate on the existing finger_v2 splits.

Reports:
- status distribution by split
- status distribution by score
- binary proxy metrics for detecting severe cases (score >= 3)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.finger_performability import get_finger_performability_gate
from backend.services.metrics_calculator import MetricsCalculator


PKL_ROOT = ROOT / "scripts" / "hpc" / "data"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "test_outputs" / "finger_performability_gate_report_v0_1.json"

ID_MAP = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "ring_mcp": 13,
    "pinky_mcp": 17,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=PKL_ROOT / "finger_train_v2.pkl")
    parser.add_argument("--valid", type=Path, default=PKL_ROOT / "finger_valid_v2.pkl")
    parser.add_argument("--test", type=Path, default=PKL_ROOT / "finger_test_v2.pkl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def seq_to_landmark_frames(seq: np.ndarray, feature_names: list[str], fps: float = 30.0) -> list[dict]:
    fmap = {name: i for i, name in enumerate(feature_names)}
    frames: list[dict] = []
    for t in range(seq.shape[0]):
        keypoints = []
        for prefix, idx in ID_MAP.items():
            keypoints.append(
                {
                    "id": idx,
                    "x": float(seq[t, fmap[f"{prefix}_x"]]),
                    "y": float(seq[t, fmap[f"{prefix}_y"]]),
                    "z": float(seq[t, fmap[f"{prefix}_z"]]),
                    "visibility": 1.0,
                }
            )
        frames.append({"frame_number": t, "timestamp": t / fps, "keypoints": keypoints})
    return frames


def summarize_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(y_true)) < 2:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": None,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "positive_n": int(np.sum(y_true)),
            "pred_positive_n": int(np.sum(y_pred)),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_n": int(np.sum(y_true)),
        "pred_positive_n": int(np.sum(y_pred)),
    }


def evaluate_split(path: Path) -> dict:
    with path.open("rb") as handle:
        obj = pickle.load(handle)

    calc = MetricsCalculator(fps=30.0)
    gate = get_finger_performability_gate()

    rows = []
    for seq, score, vid in zip(obj["X"], obj["y"], obj["ids"]):
        metrics = calc.calculate_finger_tapping_metrics(seq_to_landmark_frames(seq, obj["features"]))
        assessment = gate.assess(metrics)
        rows.append(
            {
                "video_id": vid,
                "score": int(score),
                "status": assessment.status,
                "confidence": assessment.confidence,
                "triggers": assessment.triggers,
            }
        )

    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1

    by_score: dict[int, dict[str, int]] = {}
    for row in rows:
        score_bucket = by_score.setdefault(row["score"], {})
        score_bucket[row["status"]] = score_bucket.get(row["status"], 0) + 1

    y_true = np.array([1 if row["score"] >= 3 else 0 for row in rows], dtype=int)
    y_pred = np.array(
        [1 if row["status"] == "non_performable_or_near_impossible" else 0 for row in rows],
        dtype=int,
    )

    return {
        "rows": len(rows),
        "status_counts": statuses,
        "status_by_score": by_score,
        "severe_ge3_binary": summarize_binary(y_true, y_pred),
        "example_non_performable": [row for row in rows if row["status"] == "non_performable_or_near_impossible"][:10],
        "example_uncertain": [row for row in rows if row["status"] == "uncertain"][:10],
    }


def main() -> int:
    args = parse_args()
    report = {
        "train": evaluate_split(args.train),
        "valid": evaluate_split(args.valid),
        "test": evaluate_split(args.test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
