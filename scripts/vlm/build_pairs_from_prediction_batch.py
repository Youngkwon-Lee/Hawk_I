"""
Build score/observation pair JSONL from bootstrap candidate CSV + observation prediction batch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_CANDIDATES = Path("data/processed/motorexam_vqa/silver_train_candidates_finger_observation_bootstrap_v0_1.csv")
DEFAULT_PREDICTIONS = Path("experiments/results/test_outputs/finger_observation_bootstrap_predictions_v0_1.jsonl")
DEFAULT_OUTPUT = Path("experiments/results/test_outputs/finger_observation_score_pairs_v0_1.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_jsonish(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :]
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```") :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]
    return json.loads(cleaned.strip())


def aggregate_observations(rows: list[dict]) -> dict:
    answers = []
    motion_cues: list[str] = []
    body_regions: list[str] = []
    longitudinal_values = []
    evidence_spans = []
    question_ids = []

    for row in rows:
        question_ids.append(row["question_id"])
        obs = parse_jsonish(row["prediction"])
        answers.append(str(obs.get("answer", "")).strip().lower())
        cues = obs.get("motion_cue") or []
        if isinstance(cues, str):
            cues = [cues]
        for cue in cues:
            cue = str(cue).strip()
            if cue and cue not in motion_cues:
                motion_cues.append(cue)
        regions = obs.get("body_region") or []
        if isinstance(regions, str):
            regions = [regions]
        for region in regions:
            region = str(region).strip()
            if region and region not in body_regions:
                body_regions.append(region)
        longitudinal = str(obs.get("longitudinal_change", "")).strip().lower()
        if longitudinal:
            longitudinal_values.append(longitudinal)
        if "evidence_span" in obs:
            evidence_spans.append(obs["evidence_span"])

    if any(answer == "yes" for answer in answers):
        final_answer = "yes"
    elif answers:
        final_answer = "no"
    else:
        final_answer = ""

    if any(value == "present" for value in longitudinal_values):
        longitudinal_change = "present"
    elif longitudinal_values and all(value == "absent" for value in longitudinal_values):
        longitudinal_change = "absent"
    else:
        longitudinal_change = "uncertain"

    return {
        "answer": final_answer,
        "motion_cue": motion_cues,
        "body_region": body_regions,
        "longitudinal_change": longitudinal_change,
        "observation_question_ids": question_ids,
        "evidence_spans": evidence_spans,
    }


def main() -> int:
    args = parse_args()
    candidate_rows = {}
    with args.candidates.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate_rows[row["candidate_id"]] = row

    grouped: dict[str, list[dict]] = {}
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("question_type") != "observation":
                continue
            grouped.setdefault(row["candidate_id"], []).append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    paired = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate_id, observation_rows in grouped.items():
            candidate = candidate_rows.get(candidate_id)
            if not candidate:
                continue
            out = {
                "task": candidate["task"],
                "videos": [candidate["video_path"]],
                "candidate_id": candidate_id,
                "target_score": int(candidate["score"]),
                "observation_count": len(observation_rows),
                "observation": aggregate_observations(observation_rows),
                "raw_observations": [
                    {
                        "question_id": row["question_id"],
                        "prediction": parse_jsonish(row["prediction"]),
                    }
                    for row in observation_rows
                ],
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            paired += 1

    print(json.dumps({"candidates": str(args.candidates), "predictions": str(args.predictions), "output": str(args.output), "paired_trials": paired}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
