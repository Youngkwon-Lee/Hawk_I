#!/usr/bin/env python3
"""Load offline VLM predictions (JSONL) into the shared patient timeline.

The KHF demo runs the fine-tuned model offline on a training GPU, so its
predictions are ingested here instead of served live. Each prediction becomes an
``observations`` row that the History page renders alongside ParkiCheck
assessments, tagged as an offline research prediction.

Dry run (default) prints what would be written and touches nothing:

    python backend/scripts/ingest_model_predictions.py preds.jsonl \\
        --subject-person-id <demo-person-uuid>

Add --apply to write. Re-running the same file updates rows in place because
each prediction has a stable fhir_id.

Input format — one JSON object per line:

    {"clip_id": "PD4T_S12_gait_03", "task": "gait", "predicted_score": 2,
     "rationale": "...", "dataset": "PD4T", "split": "test",
     "model": "qwen3-vl-4b-c3", "condition": "C3", "confidence": 0.81,
     "true_score": 2, "subject_ref": "S12",
     "observed_at": "2026-08-09T10:00:00Z"}

Required: clip_id, task, predicted_score, dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.prediction_ingest import (  # noqa: E402
    analysis_id_for,
    attach_research_provenance,
    load_predictions,
    prediction_to_result,
)
from services.supabase_observations import (  # noqa: E402
    _extract_returned_id,
    _post_row,
    build_activity_session_row,
    build_observation_row,
    get_supabase_observation_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, help="JSONL file of model predictions")
    parser.add_argument(
        "--subject-person-id",
        required=True,
        help="physio_app person UUID the demo timeline belongs to",
    )
    parser.add_argument("--organization-id", help="override HAWKEYE_SUPABASE_ORGANIZATION_ID")
    parser.add_argument("--limit", type=int, help="ingest at most N predictions")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write to Supabase (default is a dry run)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.predictions.exists():
        print(f"error: {args.predictions} not found", file=sys.stderr)
        return 2

    predictions, summary = load_predictions(args.predictions.read_text(encoding="utf-8").splitlines())
    if args.limit is not None:
        predictions = predictions[: args.limit]

    print(f"parsed {summary.converted}/{summary.total} predictions from {args.predictions}")
    for line, reason in summary.skipped:
        print(f"  skipped line {line}: {reason}", file=sys.stderr)
    if not predictions:
        print("nothing to ingest")
        return 1 if summary.skipped else 0

    datasets = sorted({str(p.get("dataset")) for p in predictions})
    models = sorted({str(p.get("model") or "unspecified") for p in predictions})
    conditions = sorted({str(p.get("condition") or "NA") for p in predictions})
    print(f"dataset(s): {', '.join(datasets)} | model(s): {', '.join(models)} | condition(s): {', '.join(conditions)}")
    print(f"target subject_person_id: {args.subject_person_id}")

    if not args.apply:
        print("\n-- DRY RUN (no writes). Sample of what would be created: --")
        for prediction in predictions[:3]:
            result = prediction_to_result(prediction)
            print(
                json.dumps(
                    {
                        "fhir_id": f"hawkeye-{analysis_id_for(prediction)}",
                        "code_task": result["video_type"],
                        "score": result["updrs_score"]["total_score"],
                        "rationale_present": "ai_interpretation" in result,
                        "research_provenance": {
                            "dataset": prediction.get("dataset"),
                            "split": prediction.get("split"),
                            "model": prediction.get("model"),
                            "condition": prediction.get("condition"),
                        },
                    },
                    ensure_ascii=False,
                )
            )
        print(f"\n{len(predictions)} row(s) would be written. Re-run with --apply to write.")
        return 0

    config = get_supabase_observation_config()
    if config is None:
        print(
            "error: Supabase is not configured. Set HAWKEYE_SUPABASE_URL, "
            "HAWKEYE_SUPABASE_SERVICE_KEY, HAWKEYE_SUPABASE_ORGANIZATION_ID, "
            "and the operator person ids.",
            file=sys.stderr,
        )
        return 2

    config = replace(
        config,
        subject_person_id=args.subject_person_id,
        organization_id=args.organization_id or config.organization_id,
        activity_session_id=None,
    )

    written = 0
    failed = 0
    for prediction in predictions:
        clip_id = prediction.get("clip_id")
        try:
            result = prediction_to_result(prediction)
            session_response = _post_row(
                config, config.activity_sessions_table, build_activity_session_row(result, config)
            )
            if session_response.status_code >= 400:
                print(f"  FAIL {clip_id}: activity session {session_response.status_code} {session_response.text[:160]}", file=sys.stderr)
                failed += 1
                continue

            activity_session_id = _extract_returned_id(session_response)
            if not activity_session_id:
                print(f"  FAIL {clip_id}: activity session returned no id", file=sys.stderr)
                failed += 1
                continue

            row = attach_research_provenance(
                build_observation_row(result, config, activity_session_id), prediction
            )
            response = _post_row(config, config.table, row)
            if response.status_code >= 400:
                print(f"  FAIL {clip_id}: observation {response.status_code} {response.text[:160]}", file=sys.stderr)
                failed += 1
                continue
            written += 1
        except Exception as exc:  # keep going so one bad row cannot abort the batch
            print(f"  FAIL {clip_id}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nwrote {written} observation(s); {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
