"""Export MotorExam-VQA records to Qwen-style SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "processed" / "motorexam_vqa" / "approved_exports" / "motorexam_vqa_gold_approved_v0_1.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "motorexam_vqa" / "sft" / "motorexam_qwen_sft_v0_1.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input full-record JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output SFT JSONL")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=[],
        help="Optional task filter, e.g. gait finger_tapping",
    )
    parser.add_argument(
        "--question-types",
        nargs="*",
        default=[],
        help="Optional question_type filter",
    )
    parser.add_argument(
        "--include-rationale",
        action="store_true",
        help="Include notes/rationale fields in assistant target when available",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return ROOT / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_instruction(record: dict[str, Any]) -> str:
    lines = [
        "You are reviewing a Parkinson's motor assessment example.",
        f"Task: {record['task']}",
        f"Question Type: {record['question_type']}",
        f"Question: {record['question']}",
        f"Answer Type: {record['answer_type']}",
        "Return JSON only.",
    ]
    return "\n".join(lines)


def build_assistant_target(record: dict[str, Any], include_rationale: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "answer": record["answer"],
        "visibility": record.get("visibility", "good"),
        "uncertainty_flag": record.get("uncertainty_flag", "none"),
    }

    if "motion_cue" in record:
        payload["motion_cue"] = record["motion_cue"]
    if "body_region" in record:
        payload["body_region"] = record["body_region"]
    if "longitudinal_change" in record:
        payload["longitudinal_change"] = record["longitudinal_change"]
    if "evidence_span" in record:
        payload["evidence_span"] = record["evidence_span"]
    if include_rationale:
        if "notes" in record:
            payload["notes"] = record["notes"]
        if "rationale_draft" in record:
            payload["rationale_draft"] = record["rationale_draft"]
    return payload


def build_sft_record(record: dict[str, Any], include_rationale: bool) -> dict[str, Any]:
    video_items: list[dict[str, Any]] = []

    if record.get("question_type") == "comparison":
        video_items.extend(
            [
                {"type": "video", "video": record["video_path_a"]},
                {"type": "video", "video": record["video_path_b"]},
            ]
        )
    else:
        video_items.append({"type": "video", "video": record["video_path"]})

    user_content = [{"type": "text", "text": build_instruction(record)}, *video_items]
    assistant_payload = build_assistant_target(record, include_rationale)

    return {
        "sample_id": record["sample_id"],
        "task": record["task"],
        "question_id": record["question_id"],
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
        ],
        "metadata": {
            "split": record.get("split"),
            "quality_tier": record.get("quality_tier"),
            "question_type": record.get("question_type"),
            "answer_type": record.get("answer_type"),
            "source_dataset": record.get("source_dataset"),
        },
    }


def main() -> int:
    args = parse_args()
    args.input = resolve_path(args.input)
    args.output = resolve_path(args.output)

    rows = load_jsonl(args.input)
    if args.tasks:
        rows = [row for row in rows if row.get("task") in set(args.tasks)]
    if args.question_types:
        rows = [row for row in rows if row.get("question_type") in set(args.question_types)]

    exported = [build_sft_record(row, include_rationale=args.include_rationale) for row in rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in exported:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Exported {len(exported)} SFT records.")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
