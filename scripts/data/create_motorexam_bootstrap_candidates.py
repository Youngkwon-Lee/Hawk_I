"""
Create a small task-specific bootstrap candidate CSV from the silver candidate pool.

Intended use:
- build finger_tapping observation bootstrap sets
- keep score buckets roughly balanced
- optionally override question_pack for focused collection
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "motorexam_vqa" / "silver_train_candidates_v0_1.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "motorexam_vqa" / "bootstrap_candidates_v0_1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", required=True, help="Task filter, e.g. finger_tapping")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of candidates")
    parser.add_argument(
        "--question-pack",
        default="",
        help="Optional override for question_pack, e.g. FT-O01;FT-O02",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return ROOT / path


def main() -> int:
    args = parse_args()
    args.input = resolve_path(args.input)
    args.output = resolve_path(args.output)

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader if row.get("task") == args.task]

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row.get("score", "unknown")].append(row)

    selected: list[dict[str, str]] = []
    max_bucket_len = max((len(bucket) for bucket in buckets.values()), default=0)
    ordered_scores = sorted(buckets.keys(), key=lambda value: (value == "unknown", int(value) if value.isdigit() else 999))

    for index in range(max_bucket_len):
        for score in ordered_scores:
            bucket = buckets[score]
            if index < len(bucket):
                row = dict(bucket[index])
                if args.question_pack:
                    row["question_pack"] = args.question_pack
                selected.append(row)
                if len(selected) >= args.limit:
                    break
        if len(selected) >= args.limit:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(
        {
            "input": str(args.input),
            "output": str(args.output),
            "task": args.task,
            "rows": len(selected),
            "question_pack": args.question_pack or "original",
            "scores": [row.get("score") for row in selected],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
