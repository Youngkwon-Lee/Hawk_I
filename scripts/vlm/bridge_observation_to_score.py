"""
Cheap stage-2 bridge from observation JSON to a provisional UPDRS-like score.

This is intentionally heuristic and is meant for fast experimentation after an
observation-only VLM stage. It should be treated as a baseline bridge, not a
clinical scoring model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("experiments/results/test_outputs/qwen_comparison_actual_video_observation_only_v0_2.jsonl")
DEFAULT_OUTPUT = Path("experiments/results/test_outputs/qwen_observation_bridge_score_v0_1.jsonl")

SEVERE_CUES = {
    "interruption",
    "hesitation",
    "fatigue_over_time",
    "freezing_like_pause",
    "turning_difficulty",
    "postural_instability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_jsonish(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :]
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```") :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def bridge_score(observation: dict[str, Any], *, task: str | None = None) -> tuple[int, dict[str, Any]]:
    answer = str(observation.get("answer", "")).strip().lower()
    cues = observation.get("motion_cue") or []
    if isinstance(cues, str):
        cues = [cues]
    cues = [str(c).strip() for c in cues if str(c).strip()]
    longitudinal = str(observation.get("longitudinal_change", "")).strip().lower()

    if answer in {"no", "absent"}:
        return 0, {
            "reason": "negative observation answer",
            "cue_count": 0,
            "longitudinal_change": longitudinal or "missing",
            "task": task or "unknown",
        }

    task = (task or "").strip().lower()

    if task == "gait":
        severe_present = any(cue in SEVERE_CUES for cue in cues)
        if severe_present and longitudinal == "present":
            score = 3
        elif severe_present:
            score = 2
        elif longitudinal == "present":
            score = 2
        else:
            score = 1
    elif task == "finger_tapping":
        score = 1
        if cues:
            score += 1
        if longitudinal == "present":
            score += 1
        if len(cues) >= 2 or any(cue in SEVERE_CUES for cue in cues):
            score += 1
    else:
        score = 1
        if cues:
            score += 1
        if longitudinal == "present":
            score += 1
        if len(cues) >= 2 or any(cue in SEVERE_CUES for cue in cues):
            score += 1

    score = max(0, min(4, score))
    return score, {
        "reason": "positive observation bridged to score",
        "cue_count": len(cues),
        "cues": cues,
        "longitudinal_change": longitudinal or "missing",
        "task": task or "unknown",
    }


def main() -> int:
    args = parse_args()
    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            base = parse_jsonish(row["base"])
            adapted = parse_jsonish(row["small"])
            base_score, base_meta = bridge_score(base)
            adapted_score, adapted_meta = bridge_score(adapted)
            out = {
                "sample_id": row["sample_id"],
                "task": row["task"],
                "question_id": row["question_id"],
                "target": row["target"],
                "base_observation": base,
                "base_bridge_score": base_score,
                "base_bridge_meta": base_meta,
                "adapted_observation": adapted,
                "adapted_bridge_score": adapted_score,
                "adapted_bridge_meta": adapted_meta,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
