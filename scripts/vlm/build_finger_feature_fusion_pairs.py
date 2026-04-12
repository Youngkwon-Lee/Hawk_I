"""
Merge finger observation-score pairs with extracted kinematic features.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_PAIRS = Path("experiments/results/test_outputs/finger_observation_score_pairs_v0_3.jsonl")
DEFAULT_FEATURES = Path("experiments/results/test_outputs/finger_bootstrap_features_v0_1.csv")
DEFAULT_OUTPUT = Path("experiments/results/test_outputs/finger_observation_feature_pairs_v0_1.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    feature_rows = {}
    with args.features.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate_id = row["candidate_id"]
            feature_rows[candidate_id] = {
                key: float(value) if value not in {"", None} else 0.0
                for key, value in row.items()
                if key
                not in {
                    "candidate_id",
                    "task",
                    "video_id",
                    "subject_id",
                    "score",
                    "video_path",
                }
            }

    merged = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row["candidate_id"]
            if candidate_id not in feature_rows:
                continue
            row["kinematic_features"] = feature_rows[candidate_id]
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            merged += 1

    print({"pairs": str(args.pairs), "features": str(args.features), "output": str(args.output), "merged_rows": merged})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
