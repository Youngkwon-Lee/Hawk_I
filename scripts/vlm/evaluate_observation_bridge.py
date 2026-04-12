"""
Evaluate the observation-to-score bridge on paired MotorExam-VQA examples.

This script looks for trials that have both:
- a score question
- an observation question

It then applies the heuristic bridge to the observation answer and compares the
derived score against the gold score label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

try:
    from scripts.vlm.bridge_observation_to_score import bridge_score, parse_jsonish
except ModuleNotFoundError:
    from vlm.bridge_observation_to_score import bridge_score, parse_jsonish


DEFAULT_INPUT = Path("experiments/data/motorexam_qwen_sft_small_train_v0_1.jsonl")
DEFAULT_OUTPUT = Path("experiments/results/test_outputs/qwen_observation_bridge_eval_v0_1.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def group_key(row: dict) -> tuple[str, tuple[str, ...]]:
    task = row.get("task", "unknown")
    videos = tuple(item["video"] for item in row["messages"][0]["content"] if item.get("type") == "video")
    return task, videos


def main() -> int:
    args = parse_args()
    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    grouped: dict[tuple[str, tuple[str, ...]], dict[str, dict]] = {}
    for row in rows:
        qtype = row.get("metadata", {}).get("question_type", "unknown")
        key = group_key(row)
        grouped.setdefault(key, {})[qtype] = row

    paired = []
    errors = []
    for key, bundle in grouped.items():
        if "score" not in bundle or "observation" not in bundle:
            continue
        score_row = bundle["score"]
        observation_row = bundle["observation"]
        score_json = parse_jsonish(score_row["messages"][1]["content"])
        obs_json = parse_jsonish(observation_row["messages"][1]["content"])
        target_score = int(score_json["answer"])
        bridged_score, bridge_meta = bridge_score(obs_json, task=key[0])
        paired.append(
            {
                "task": key[0],
                "videos": list(key[1]),
                "score_sample_id": score_row["sample_id"],
                "observation_sample_id": observation_row["sample_id"],
                "target_score": target_score,
                "observation_answer": obs_json,
                "bridged_score": bridged_score,
                "bridge_meta": bridge_meta,
                "absolute_error": abs(target_score - bridged_score),
                "exact_match": target_score == bridged_score,
            }
        )
        errors.append(abs(target_score - bridged_score))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in paired:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    exact = sum(1 for row in paired if row["exact_match"])
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "paired_trials": len(paired),
        "exact_match": exact,
        "exact_rate": (exact / len(paired)) if paired else 0.0,
        "mae": mean(errors) if errors else None,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
