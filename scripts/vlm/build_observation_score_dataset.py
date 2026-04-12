"""
Build paired observation->score calibration examples from MotorExam-VQA SFT JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

try:
    from scripts.vlm.bridge_observation_to_score import parse_jsonish
except ModuleNotFoundError:
    from vlm.bridge_observation_to_score import parse_jsonish


DEFAULT_INPUT = Path("experiments/data/motorexam_qwen_sft_small_train_v0_1.jsonl")
DEFAULT_OUTPUT = Path("experiments/results/test_outputs/observation_score_pairs_v0_1.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", type=str, default="", help="Optional task filter, e.g. gait or finger_tapping")
    return parser.parse_args()


def trial_key(row: dict) -> tuple[str, tuple[str, ...]]:
    task = row.get("task", "unknown")
    videos = tuple(item["video"] for item in row["messages"][0]["content"] if item.get("type") == "video")
    return task, videos


def main() -> int:
    args = parse_args()
    grouped: dict[tuple[str, tuple[str, ...]], dict[str, dict]] = {}
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qtype = row.get("metadata", {}).get("question_type", "unknown")
            grouped.setdefault(trial_key(row), {})[qtype] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    paired = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for (task, videos), bundle in grouped.items():
            if "score" not in bundle or "observation" not in bundle:
                continue
            if args.task and task != args.task:
                continue
            score_row = bundle["score"]
            observation_row = bundle["observation"]
            score_json = parse_jsonish(score_row["messages"][1]["content"])
            observation_json = parse_jsonish(observation_row["messages"][1]["content"])
            out = {
                "task": task,
                "videos": list(videos),
                "score_sample_id": score_row["sample_id"],
                "observation_sample_id": observation_row["sample_id"],
                "target_score": int(score_json["answer"]),
                "observation": observation_json,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            paired += 1
    print(
        json.dumps(
            {"input": str(args.input), "output": str(args.output), "task": args.task or "all", "paired_trials": paired},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
