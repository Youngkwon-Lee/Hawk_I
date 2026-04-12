"""
Run batch VLM inference on expanded MotorExam-VQA task JSONL.

Intended for cheap teacher-style observation extraction on bootstrap subsets,
especially when we want task-specific weak labels such as finger-tapping
observation outputs to pair with known scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

try:
    from scripts.vlm.compare_qwen_lora_runs import (
        load_model_bundle,
        sample_video_to_frame_paths,
    )
except ModuleNotFoundError:
    from vlm.compare_qwen_lora_runs import load_model_bundle, sample_video_to_frame_paths


DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "vlm" / "qwen_vl_lora_motorexam_small_v0_1.yaml"
DEFAULT_INPUT = ROOT / "data" / "processed" / "motorexam_vqa" / "finger_observation_bootstrap_expanded_tasks_v0_1.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "test_outputs" / "finger_observation_bootstrap_predictions_v0_1.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter", type=Path, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_user_prompt(row: dict[str, Any]) -> str:
    answer_options = row.get("answer_options")
    answer_options_text = ", ".join(str(option) for option in answer_options) if answer_options else "Use the schema exactly."
    task_specific_notes: list[str] = []
    if row["task"] == "finger_tapping":
        task_specific_notes.extend(
            [
                "For finger tapping, avoid generic labels like 'finger_tapping' as the final cue when a more specific cue is visible.",
                "Prefer canonical motion cues such as amplitude_decrement, slowed_open_close_rate, rhythm_irregularity, hesitation, interruption, fatigue_over_time, left_right_asymmetry.",
                "If the question asks severity_4, answer must be exactly one of: none, mild, moderate, severe.",
                "If you cannot clearly justify a severe answer from the video, prefer the lower severity.",
            ]
        )
    return "\n".join(
        [
            "You are reviewing a Parkinson's motor assessment example.",
            f"Task: {row['task']}",
            f"Question ID: {row['question_id']}",
            f"Question Type: {row['question_type']}",
            f"Question: {row['question']}",
            f"Answer Type: {row['answer_type']}",
            f"Allowed Answer Options: {answer_options_text}",
            "Return JSON only.",
            "Use only visible video evidence.",
            *task_specific_notes,
            "Return schema:",
            '{',
            '  "answer": "...",',
            '  "visibility": "good|acceptable|poor",',
            '  "uncertainty_flag": "none|low_visibility|partial_occlusion|short_duration|ambiguous_cue|conflicting_cues|insufficient_temporal_context",',
            '  "motion_cue": ["..."],',
            '  "body_region": ["..."],',
            '  "longitudinal_change": "present|absent|uncertain",',
            '  "evidence_span": {"start_sec": 0.0, "end_sec": 0.0}',
            '}',
        ]
    )


def render_messages(row: dict[str, Any], frame_budget: int, frame_width: int) -> list[dict[str, Any]]:
    task = row["task"]
    content: list[dict[str, Any]] = [{"type": "text", "text": build_user_prompt(row)}]
    video_path = ROOT / row["video_path"]
    frame_paths = sample_video_to_frame_paths(
        video_path,
        task=task,
        frame_budget=frame_budget,
        frame_width=frame_width,
    )
    for frame_path in frame_paths:
        content.append({"type": "image", "image": frame_path})
    return [{"role": "user", "content": content}]


def run_inference(processor: Any, model: Any, row: dict[str, Any], frame_budget: int, frame_width: int) -> str:
    import torch

    messages = render_messages(row, frame_budget=frame_budget, frame_width=frame_width)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    prompt_len = int(inputs["input_ids"].shape[1])
    decoded = processor.batch_decode(
        [outputs[0][prompt_len:]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return decoded


def main() -> int:
    args = parse_args()
    args.config = args.config if args.config.is_absolute() else ROOT / args.config
    args.input = args.input if args.input.is_absolute() else ROOT / args.input
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.adapter is not None and not args.adapter.is_absolute():
        args.adapter = ROOT / args.adapter

    config = load_yaml(args.config)
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    frame_budget_by_task = {k: int(v) for k, v in config["data"].get("sampling_policy", {}).items()}
    processor, model = load_model_bundle(
        config["model"]["base_model"],
        str(config["model"].get("quantization", "none")),
        adapter_path=args.adapter,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            frame_budget = int(frame_budget_by_task.get(row["task"], 8))
            pred = run_inference(processor, model, row, frame_budget=frame_budget, frame_width=512)
            out = {
                "sample_id": row["sample_id"],
                "candidate_id": row.get("candidate_id"),
                "task": row["task"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "answer_type": row["answer_type"],
                "video_path": row["video_path"],
                "prediction": pred,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
