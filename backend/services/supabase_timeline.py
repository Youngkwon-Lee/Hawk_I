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
        "id",
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

MEDICATION_SELECT = ",".join(
    [
        "fhir_id",
        "status",
        "medication_code",
        "medication_display",
        "effective_start",
        "date_asserted",
        "dosage",
        "information_source_type",
        "subject_person_id",
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
    medication_context = medication_context if isinstance(medication_context, dict) else {}
    hawk_i = context.get("hawk_i")
    hawk_i = hawk_i if isinstance(hawk_i, dict) else {}

    return {
        "observed_at": row.get("effective_datetime"),
        "code": row.get("code"),
        "status": row.get("status"),
        "score": score,
        "source_type": row.get("source_type"),
        "app_source": _resolve_app_source(row, context),
        "confidence": context.get("confidence"),
        "analysis_id": context.get("analysis_id") or hawk_i.get("analysis_id"),
        "observation_id": row.get("id"),
        "activity_session_id": row.get("activity_session_id"),
        "subject_person_id": row.get("subject_person_id"),
        "fhir_id": row.get("fhir_id"),
        "has_medication_context": bool(medication_context.get("available")),
        "medication_name": medication_context.get("medication"),
        "medication_dose_mg": medication_context.get("dose_mg"),
        "medication_taken_at": medication_context.get("taken_at"),
        "hours_after_reported_dose": medication_context.get("hours_before_assessment"),
        "has_hawk_i_review": bool(hawk_i),
    }


def normalize_medication_statement(row: dict[str, Any]) -> dict[str, Any]:
    dosage = row.get("dosage")
    dosage = dosage if isinstance(dosage, dict) else {}
    app_source = dosage.get("app_source")
    if not isinstance(app_source, str) or not app_source.strip():
        app_source = (
            "parkicheck"
            if str(row.get("fhir_id") or "").startswith("parkicheck-medication-")
            else "physio_app"
        )

    return {
        "event_id": row.get("fhir_id"),
        "observed_at": row.get("effective_start") or row.get("date_asserted"),
        "status": row.get("status"),
        "medication_code": row.get("medication_code"),
        "medication_display": row.get("medication_display"),
        "dose_mg": dosage.get("dose_mg"),
        "dose_unit": dosage.get("unit") or "mg",
        "information_source_type": row.get("information_source_type"),
        "subject_person_id": row.get("subject_person_id"),
        "app_source": app_source.strip(),
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


def fetch_medication_statements(
    subject_person_id: str,
    access_token: str,
    limit: int = 100,
    config: SupabaseObservationConfig | None = None,
) -> list[dict[str, Any]] | None:
    """Return patient medication statements newest first without effect inference."""
    config = config or get_supabase_observation_config()
    if config is None:
        return None

    response = requests.get(
        f"{config.url}/rest/v1/medication_statements",
        params={
            "select": MEDICATION_SELECT,
            "subject_person_id": f"eq.{subject_person_id}",
            "organization_id": f"eq.{config.organization_id}",
            "order": "effective_start.desc.nullslast,date_asserted.desc",
            "limit": str(limit),
        },
        headers=caller_headers(config, access_token),
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        return []
    return [
        normalize_medication_statement(row)
        for row in rows
        if isinstance(row, dict)
    ]
