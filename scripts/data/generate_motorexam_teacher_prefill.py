"""Generate MotorExam-VQA teacher prefill from candidate CSV and actual videos."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT
from data.motorexam_vqa_config import answer_options_for_type, get_question_spec

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "rationale")))
from _env import (
    configure_google_genai,
    get_open_model_api_key,
    get_open_model_base_url,
    get_open_model_generate_model,
)

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


ROOT = PROJECT_ROOT
DEFAULT_CANDIDATES = ROOT / "data" / "processed" / "motorexam_vqa" / "gold_benchmark_candidates_v0_1.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "motorexam_vqa" / "prefill" / "gold_benchmark_prefill_v0_1.jsonl"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_OPEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_FRAME_COUNT = 8

TASK_FRAME_BUDGETS = {
    "gait": 16,
    "finger_tapping": 24,
    "hand_movement": 24,
    "leg_agility": 24,
}

TASK_PHASE_POSITIONS = {
    "gait": [
        0.00, 0.06, 0.12, 0.20, 0.28, 0.36, 0.44, 0.52,
        0.60, 0.68, 0.76, 0.84, 0.90, 0.94, 0.98,
    ],
    "finger_tapping": [
        0.00, 0.03, 0.06, 0.09, 0.14, 0.20, 0.28, 0.36,
        0.44, 0.52, 0.60, 0.68, 0.76, 0.84, 0.90, 0.95, 0.99,
    ],
    "hand_movement": [
        0.00, 0.03, 0.06, 0.10, 0.16, 0.24, 0.32, 0.40,
        0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.94, 0.98,
    ],
    "leg_agility": [
        0.00, 0.03, 0.06, 0.10, 0.16, 0.24, 0.32, 0.40,
        0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.94, 0.98,
    ],
}

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}

BODY_REGION_MAP = {
    "upper body": "bilateral_upper",
    "lower body": "bilateral_lower",
    "whole body": "whole_body",
    "right hand": "right_hand",
    "left hand": "left_hand",
    "right arm": "right_arm",
    "left arm": "left_arm",
    "right leg": "right_leg",
    "left leg": "left_leg",
    "pelvis": "pelvis",
    "trunk": "trunk",
    "head": "head",
    "fingers": "fingers",
    "wrist": "wrist",
    "shoulder": "shoulder",
    "elbow": "elbow",
    "knee": "knee",
    "ankle": "ankle",
    "foot": "foot",
}

MOTION_CUE_MAP = {
    "arm swing": "reduced_arm_swing",
    "reduced arm swing": "reduced_arm_swing",
    "pelvic rotation": "reduced_range_of_motion",
    "slow gait": "slowed_gait",
    "slowed gait": "slowed_gait",
    "shuffling gait": "shortened_step_length",
    "short step length": "shortened_step_length",
    "shortened step length": "shortened_step_length",
    "hesitation": "hesitation",
    "rhythm irregularity": "rhythm_irregularity",
    "irregular rhythm": "rhythm_irregularity",
    "amplitude decrement": "amplitude_decrement",
    "fatigue over time": "fatigue_over_time",
    "reduced range of motion": "reduced_range_of_motion",
    "asymmetry": "left_right_asymmetry",
    "left-right asymmetry": "left_right_asymmetry",
    "slowed movement": "slowed_movement",
    "turning difficulty": "turning_difficulty",
    "postural instability": "postural_instability",
    "freezing": "freezing_like_pause",
    "slow open close rate": "slowed_open_close_rate",
    "slowed hand open close": "slowed_hand_open_close",
    "slowed leg lifts": "slowed_leg_lifts",
    "reduced lift amplitude": "reduced_lift_amplitude",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES, help="Candidate CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Prefill JSONL output path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--limit", type=int, default=4, help="Maximum number of candidate rows to process")
    parser.add_argument(
        "--mode",
        choices=["mock", "gemini", "open_model", "runpod"],
        default="mock",
        help="Use mock generation, Gemini native video, or an OpenAI-compatible open model / Runpod endpoint",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=2,
        help="Maximum number of questions per candidate row",
    )
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL for open_model/runpod mode")
    parser.add_argument("--api-key", default="", help="API key for open_model/runpod mode")
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAME_COUNT,
        help="Number of frames to sample for open_model/runpod mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Maximum completion tokens for open_model/runpod mode",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=512,
        help="Target width for sampled frames in open_model/runpod mode",
    )
    parser.add_argument(
        "--disable-task-aware-sampling",
        action="store_true",
        help="Use exactly --frames uniformly instead of task-aware frame budgets",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return ROOT / path


def parse_question_pack(row: dict[str, str], question_limit: int) -> list[str]:
    question_ids = [item.strip() for item in row["question_pack"].split(";") if item.strip()]
    return question_ids[:question_limit]


def build_sample_id(candidate_id: str, question_id: str) -> str:
    return f"{candidate_id}__{question_id.lower().replace('-', '_')}"


def build_prompt(candidate: dict[str, str], task: str, question_id: str, spec: dict[str, Any]) -> str:
    answer_options = answer_options_for_type(spec["answer_type"])
    answer_options_text = (
        ", ".join(str(option) for option in answer_options)
        if answer_options is not None
        else "Use the answer type schema exactly."
    )
    return f"""
You are reviewing a Parkinson's motor assessment video.

Task: {task}
Question ID: {question_id}
Question Type: {spec["question_type"]}
Question: {spec["question"]}
Answer Type: {spec["answer_type"]}
Allowed Answer Options: {answer_options_text}

Important rules:
- Use only visible video evidence.
- Do not infer diagnosis, medication state, or hidden clinical facts.
- If evidence is weak, use uncertainty_flag accordingly.
- Return JSON only.

Return this schema:
{{
  "answer": "...",
  "visibility": "good|acceptable|poor",
  "uncertainty_flag": "none|low_visibility|partial_occlusion|short_duration|ambiguous_cue|conflicting_cues|insufficient_temporal_context",
  "motion_cue": ["..."],
  "body_region": ["..."],
  "longitudinal_change": "present|absent|uncertain",
  "evidence_span": {{"start_sec": 0.0, "end_sec": 0.0}},
  "notes": "short reviewer aid note",
  "rationale_draft": "1-2 sentence structured rationale"
}}

If the answer type is not temporal_span, you may omit evidence_span if unsupported.
If motion_cue or body_region are unknown, return an empty list.
""".strip()


def try_parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def normalize_answer(answer: Any, answer_type: str) -> Any:
    if answer_type == "ordinal_0_4":
        if isinstance(answer, str):
            answer = answer.strip()
            if answer.isdigit():
                return int(answer)
        return answer

    if answer_type == "binary_yes_no" and isinstance(answer, str):
        lowered = answer.strip().lower()
        if lowered in {"yes", "no"}:
            return lowered

    if answer_type == "severity_4" and isinstance(answer, str):
        lowered = answer.strip().lower()
        if lowered in {"none", "mild", "moderate", "severe"}:
            return lowered

    if answer_type == "longitudinal_change_3way" and isinstance(answer, str):
        lowered = answer.strip().lower()
        if lowered in {"present", "absent", "uncertain"}:
            return lowered

    return answer


def normalize_list(values: Any, mapping: dict[str, str]) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        key = value.strip().lower()
        canonical = mapping.get(key)
        if canonical and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def infer_evidence_type(payload: dict[str, Any]) -> list[str]:
    evidence_type: list[str] = []
    if "evidence_span" in payload and payload["evidence_span"]:
        evidence_type.append("temporal_span")
    if payload.get("body_region"):
        evidence_type.append("body_region")
    if payload.get("motion_cue"):
        evidence_type.append("motion_cue")
    if payload.get("longitudinal_change") in {"present", "absent", "uncertain"}:
        evidence_type.append("longitudinal_change")
    return evidence_type


def effective_frame_budget(task: str, requested_frames: int, task_aware_enabled: bool) -> int:
    if not task_aware_enabled:
        return requested_frames
    if requested_frames != DEFAULT_FRAME_COUNT:
        return requested_frames
    return TASK_FRAME_BUDGETS.get(task, requested_frames)


def uniform_indices(total_frames: int, n_frames: int) -> list[int]:
    if n_frames <= 0:
        return []
    if total_frames <= 1:
        return [0]
    return sorted({int(i * (total_frames - 1) / max(n_frames - 1, 1)) for i in range(n_frames)})


def task_aware_indices(total_frames: int, n_frames: int, task: str) -> list[int]:
    if n_frames <= 0:
        return []
    if total_frames <= 1:
        return [0]

    positions = TASK_PHASE_POSITIONS.get(task)
    if not positions:
        return uniform_indices(total_frames, n_frames)

    anchor_indices = sorted({int(pos * (total_frames - 1)) for pos in positions})
    if len(anchor_indices) > n_frames:
        selected = []
        for idx in uniform_indices(len(anchor_indices), n_frames):
            selected.append(anchor_indices[idx])
        return sorted(set(selected))

    selected = set(anchor_indices)
    if len(selected) < n_frames:
        for idx in uniform_indices(total_frames, n_frames):
            selected.add(idx)
            if len(selected) >= n_frames:
                break

    return sorted(selected)


def normalize_prefill_payload(
    payload: dict[str, Any],
    *,
    row: dict[str, str],
    spec: dict[str, Any],
    mode: str,
    model_name: str,
    base_url: str,
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["answer"] = normalize_answer(normalized.get("answer"), spec["answer_type"])
    normalized["motion_cue"] = normalize_list(normalized.get("motion_cue", []), MOTION_CUE_MAP)
    normalized["body_region"] = normalize_list(normalized.get("body_region", []), BODY_REGION_MAP)

    split = SPLIT_MAP.get(row["split_target"], row["split_target"])
    normalized["split"] = split
    normalized["quality_tier"] = row["quality_tier_target"]
    normalized["quality_tier_target"] = row["quality_tier_target"]
    normalized["question"] = spec["question"]
    normalized["annotation_source"] = "teacher_only"
    normalized["source_dataset"] = "PD4T"
    normalized["video_id"] = row["video_id"]
    normalized["subject_id"] = row["subject_id"]
    normalized["updrs_score"] = int(row["score"])
    normalized["evidence_type"] = infer_evidence_type(normalized)
    normalized["prefill_source"] = mode
    normalized["prefill_model"] = model_name
    if base_url:
        normalized["prefill_base_url"] = base_url
    return normalized


def mock_prefill(sample_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    answer_type = spec["answer_type"]
    if answer_type == "binary_yes_no":
        answer: Any = "yes"
    elif answer_type == "ordinal_0_4":
        answer = 2
    elif answer_type == "comparison_3way":
        answer = "video_b"
    elif answer_type == "temporal_span":
        answer = {"start_sec": 3.5, "end_sec": 7.5}
    elif answer_type == "motion_cue_multi":
        answer = ["amplitude_decrement"]
    elif answer_type == "longitudinal_change_3way":
        answer = "present"
    else:
        answer = "moderate"

    payload = {
        "sample_id": sample_id,
        "answer": answer,
        "visibility": "good",
        "uncertainty_flag": "none",
        "motion_cue": ["amplitude_decrement"],
        "body_region": ["right_hand"],
        "longitudinal_change": "present",
        "notes": "Mock prefill for pipeline validation.",
        "rationale_draft": "Visible decrement appears over time and supports the proposed answer.",
    }
    if answer_type == "temporal_span":
        payload["evidence_span"] = answer
    return payload


def gemini_prefill(video_path: Path, model_name: str, prompt: str, sample_id: str) -> dict[str, Any]:
    if genai is None:
        raise ImportError("google.generativeai is not installed")
    configure_google_genai()
    model = genai.GenerativeModel(model_name)
    video_file = genai.upload_file(path=str(video_path))
    response = model.generate_content([video_file, prompt], stream=True)
    response.resolve()
    payload = try_parse_json(response.text)
    payload["sample_id"] = sample_id
    return payload


def extract_frames_to_base64(video_path: Path, n_frames: int, frame_width: int) -> list[str]:
    if cv2 is None:
        raise ImportError("opencv-python is required for open_model/runpod frame extraction")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")

    indices = sorted({int(i * (total_frames - 1) / max(n_frames - 1, 1)) for i in range(n_frames)})
    encoded_frames: list[str] = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue

        if frame_width > 0 and frame.shape[1] > frame_width:
            scale = frame_width / frame.shape[1]
            resized_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (frame_width, resized_height), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        encoded_frames.append(base64.b64encode(buffer).decode("utf-8"))

    cap.release()

    if not encoded_frames:
        raise RuntimeError(f"Failed to extract frames from video: {video_path}")

    return encoded_frames


def extract_task_aware_frames_to_base64(
    video_path: Path,
    *,
    task: str,
    n_frames: int,
    frame_width: int,
) -> list[str]:
    if cv2 is None:
        raise ImportError("opencv-python is required for open_model/runpod frame extraction")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")

    indices = task_aware_indices(total_frames=total_frames, n_frames=n_frames, task=task)
    encoded_frames: list[str] = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue

        if frame_width > 0 and frame.shape[1] > frame_width:
            scale = frame_width / frame.shape[1]
            resized_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (frame_width, resized_height), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        encoded_frames.append(base64.b64encode(buffer).decode("utf-8"))

    cap.release()

    if not encoded_frames:
        raise RuntimeError(f"Failed to extract task-aware frames from video: {video_path}")

    return encoded_frames


def open_model_prefill(
    video_path: Path,
    model_name: str,
    prompt: str,
    sample_id: str,
    *,
    base_url: str,
    api_key: str,
    frames: int,
    frame_width: int,
    max_tokens: int,
    task: str,
    task_aware_sampling: bool,
) -> dict[str, Any]:
    if OpenAI is None:
        raise ImportError("openai package is required for open_model/runpod mode")

    client = OpenAI(api_key=api_key, base_url=base_url)
    effective_frames = effective_frame_budget(task, frames, task_aware_sampling)
    encoded_frames = extract_task_aware_frames_to_base64(
        video_path,
        task=task,
        n_frames=effective_frames,
        frame_width=frame_width,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for encoded in encoded_frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                },
            }
        )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    payload = try_parse_json(text)
    payload["sample_id"] = sample_id
    payload["sampled_frame_count"] = len(encoded_frames)
    payload["sampling_policy"] = "task_aware_v0_1" if task_aware_sampling else "uniform"
    return payload


def main() -> int:
    args = parse_args()
    args.candidates = resolve_path(args.candidates)
    args.output = resolve_path(args.output)

    with args.candidates.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    emitted: list[dict[str, Any]] = []
    processed_rows = rows[: args.limit]
    open_model_base_url = ""
    open_model_api_key = ""
    open_model_name = args.model
    if args.mode in {"open_model", "runpod"}:
        open_model_base_url = args.base_url or get_open_model_base_url()
        open_model_api_key = args.api_key or get_open_model_api_key()
        open_model_name = args.model if args.model != DEFAULT_MODEL else get_open_model_generate_model()

    for row in processed_rows:
        task = row["task"]
        video_path = ROOT / row["video_path"]
        if not video_path.exists():
            print(f"Skipping missing video: {video_path}")
            continue

        for question_id in parse_question_pack(row, args.question_limit):
            spec = get_question_spec(task, question_id)
            sample_id = build_sample_id(row["candidate_id"], question_id)
            prompt = build_prompt(row, task, question_id, spec)

            if args.mode == "mock":
                payload = mock_prefill(sample_id, spec)
            elif args.mode == "gemini":
                payload = gemini_prefill(video_path, args.model, prompt, sample_id)
                time.sleep(1.0)
            else:
                payload = open_model_prefill(
                    video_path,
                    open_model_name,
                    prompt,
                    sample_id,
                    base_url=open_model_base_url,
                    api_key=open_model_api_key,
                    frames=args.frames,
                    frame_width=args.frame_width,
                    max_tokens=args.max_tokens,
                    task=task,
                    task_aware_sampling=not args.disable_task_aware_sampling,
                )
                time.sleep(0.5)

            payload["task"] = task
            payload["candidate_id"] = row["candidate_id"]
            payload["split_target"] = row["split_target"]
            payload["quality_tier_target"] = row["quality_tier_target"]
            payload["question_id"] = question_id
            payload["question_type"] = spec["question_type"]
            payload["answer_type"] = spec["answer_type"]
            payload["video_path"] = row["video_path"]
            payload = normalize_prefill_payload(
                payload,
                row=row,
                spec=spec,
                mode=args.mode,
                model_name=open_model_name if args.mode in {"open_model", "runpod"} else args.model,
                base_url=open_model_base_url if args.mode in {"open_model", "runpod"} else "",
            )
            emitted.append(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in emitted:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Generated {len(emitted)} prefill items.")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
