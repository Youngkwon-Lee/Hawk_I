"""Unified patient timeline read from the shared physio_app Supabase project.

ParkiCheck writes device-sourced observations and Hawk I writes ai-sourced
observations into the same ``observations`` table. This module reads them back
as one normalized timeline so the History screen can show both alongside the
local file-based analyses.
"""

from __future__ import annotations

from typing import Any

import requests

from services.supabase_auth import caller_headers
from services.supabase_observations import (
    SupabaseObservationConfig,
    get_supabase_observation_config,
)

OBSERVATION_SELECT = ",".join(
    [
        "fhir_id",
        "code",
        "status",
        "source_type",
        "value_integer",
        "value_quantity",
        "effective_datetime",
        "activity_session_id",
        "subject_person_id",
        "measurement_context",
    ]
)


def _resolve_app_source(row: dict[str, Any], context: dict[str, Any]) -> str:
    app_source = context.get("app_source")
    if isinstance(app_source, str) and app_source.strip():
        return app_source.strip()
    fhir_id = str(row.get("fhir_id") or "")
    if fhir_id.startswith("parkicheck-"):
        return "parkicheck"
    if row.get("source_type") == "ai":
        return "hawk_i"
    return "unknown"


def normalize_observation(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("measurement_context")
    context = context if isinstance(context, dict) else {}

    score = row.get("value_integer")
    if score is None:
        score = row.get("value_quantity")

    medication_context = context.get("medication_context")
    hawk_i = context.get("hawk_i")

    return {
        "observed_at": row.get("effective_datetime"),
        "code": row.get("code"),
        "status": row.get("status"),
        "score": score,
        "source_type": row.get("source_type"),
        "app_source": _resolve_app_source(row, context),
        "confidence": context.get("confidence"),
        "analysis_id": context.get("analysis_id"),
        "activity_session_id": row.get("activity_session_id"),
        "subject_person_id": row.get("subject_person_id"),
        "fhir_id": row.get("fhir_id"),
        "has_medication_context": isinstance(medication_context, dict)
        and bool(medication_context.get("available")),
        "has_hawk_i_review": isinstance(hawk_i, dict) and bool(hawk_i),
    }


def fetch_timeline(
    subject_person_id: str,
    access_token: str,
    limit: int = 100,
    config: SupabaseObservationConfig | None = None,
) -> list[dict[str, Any]] | None:
    """Return normalized timeline items for a subject, newest first.

    Returns ``None`` when the Supabase integration is not configured/enabled.
    """
    config = config or get_supabase_observation_config()
    if config is None:
        return None

    response = requests.get(
        f"{config.url}/rest/v1/{config.table}",
        params={
            "select": OBSERVATION_SELECT,
            "subject_person_id": f"eq.{subject_person_id}",
            "organization_id": f"eq.{config.organization_id}",
            "order": "effective_datetime.desc.nullslast",
            "limit": str(limit),
        },
        headers=caller_headers(config, access_token),
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        return []
    return [normalize_observation(row) for row in rows if isinstance(row, dict)]
