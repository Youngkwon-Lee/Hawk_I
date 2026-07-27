#!/usr/bin/env python3
"""Run the real Hawk I pipeline on a synthetic, hand-only smoke video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/hawkeye_local_medication_e2e"),
    )
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video not found: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["UPLOAD_FOLDER"] = str(args.output_dir)
    # This smoke must never write to a configured clinical database or call a
    # remote LLM, even if a developer has local credentials.
    for key in (
        "HAWKEYE_SUPABASE_URL",
        "HAWKEYE_SUPABASE_SERVICE_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        os.environ[key] = ""

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "backend"))

    from routes.analyze import process_video_background
    from services.medication_context import parse_medication_context

    medication_context = parse_medication_context(json.dumps({
        "available": True,
        "source": "patient_reported_local",
        "medication": "비식별 테스트 약물",
        "dose_mg": 100,
        "taken_at": "2026-07-27T00:00:00Z",
        "assessment_at": "2026-07-27T01:30:00Z",
        "hours_before_assessment": 1.5,
    }, ensure_ascii=False))
    video_id = "synthetic-local-finger-e2e"
    process_video_background(
        str(args.video),
        video_id,
        "synthetic-local-patient",
        "finger_tapping",
        {"UPLOAD_FOLDER": str(args.output_dir)},
        "rule",
        "rf",
        None,
        "synthetic-local-assessment-001",
        medication_context,
    )

    result_path = args.output_dir / f"{video_id}_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    score = result.get("updrs_score") or {}
    summary = {
        "success": result.get("success"),
        "video_type": result.get("video_type"),
        "landmark_frames": (result.get("skeleton_data") or {}).get("total_frames"),
        "score": score.get("total_score", score.get("score")),
        "assessment_session_id": result.get("assessment_session_id"),
        "medication_hours": (result.get("medication_timing") or {}).get("hours_after_reported_dose"),
        "can_infer_effect": (result.get("medication_timing") or {}).get("can_infer_medication_effect"),
        "supabase_saved": ((result.get("integrations") or {}).get("supabase_observation") or {}).get("saved"),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))

    if not (
        summary["success"] is True
        and summary["video_type"] == "finger_tapping"
        and (summary["landmark_frames"] or 0) > 0
        and summary["assessment_session_id"] == "synthetic-local-assessment-001"
        and summary["medication_hours"] == 1.5
        and summary["can_infer_effect"] is False
        and summary["supabase_saved"] is False
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
