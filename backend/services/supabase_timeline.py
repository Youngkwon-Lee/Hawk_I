"""Unified patient timeline read from the shared physio_app Supabase project.

ParkiCheck writes device-sourced observations and Hawk I writes ai-sourced
observations into the same ``observations`` table. This module reads them back
as one normalized timeline so the History screen can show both alongside the
local file-based analyses.
"""

from __future__ import annotations

from datetime import datetime, timezone
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

    metrics = context.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    interpretation = context.get("ai_interpretation")
    interpretation = interpretation if isinstance(interpretation, dict) else {}
    advisory = context.get("score_advisory")
    advisory = advisory if isinstance(advisory, dict) else {}
    performability = context.get("performability_assessment")
    performability = performability if isinstance(performability, dict) else {}

    return {
        "observed_at": row.get("effective_datetime"),
        "code": row.get("code"),
        "status": row.get("status"),
        "score": score,
        "severity": context.get("severity"),
        "source_type": row.get("source_type"),
        "app_source": _resolve_app_source(row, context),
        "confidence": context.get("confidence"),
        "score_confidence": context.get("score_confidence"),
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
        # Quantitative evidence: kinematic measurements behind the score.
        "metrics": metrics,
        # Qualitative evidence: the narrative finding a clinician reads first.
        "rationale": interpretation.get("summary") or interpretation.get("explanation"),
        # Whether the score may be relied on at all, kept separate from the score.
        "score_advisory_level": advisory.get("level"),
        "score_advisory_summary": advisory.get("summary"),
        "performability_status": performability.get("status"),
        # Provenance so a clinician can tell which pipeline produced this.
        "scoring_method": context.get("scoring_method"),
        "model_type": context.get("ml_model_type"),
    }


# A prescribed dose and a dose the patient reports taking answer different
# clinical questions, and the gap between them (a missed dose) is only visible
# if they stay distinct. FHIR MedicationStatement.status carries the split:
# "intended" is planned, "completed" is taken.
SCHEDULED_STATUSES = frozenset({"intended", "not-taken"})
TAKEN_STATUSES = frozenset({"completed", "active"})


def _medication_event_kind(row: dict[str, Any], dosage: dict[str, Any]) -> str:
    event_type = dosage.get("event_type")
    if isinstance(event_type, str):
        normalized = event_type.strip().lower()
        if "schedul" in normalized or "planned" in normalized:
            return "scheduled"
        if "dose" in normalized or "taken" in normalized or "intake" in normalized:
            return "taken"

    status = str(row.get("status") or "").strip().lower()
    if status in SCHEDULED_STATUSES:
        return "scheduled"
    if status in TAKEN_STATUSES:
        return "taken"
    return "unknown"


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
        "event_kind": _medication_event_kind(row, dosage),
        "medication_code": row.get("medication_code"),
        "medication_display": row.get("medication_display"),
        "dose_mg": dosage.get("dose_mg"),
        "dose_unit": dosage.get("unit") or "mg",
        "information_source_type": row.get("information_source_type"),
        "subject_person_id": row.get("subject_person_id"),
        "app_source": app_source.strip(),
    }


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def attach_dose_context(
    observations: list[dict[str, Any]],
    medications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Link each assessment to the most recent dose reported before it.

    A single score is not interpretable without knowing where in the dose cycle
    it was captured, so the elapsed time is reported as a plain fact. No ON/OFF
    state, drug effect, next-dose time, or causal claim is inferred - that
    boundary is a project rule, and the temporal accuracy of such labels is not
    supported at this granularity.
    """
    taken = [
        (parsed, dose)
        for dose in medications
        if (parsed := _parse_iso(dose.get("observed_at"))) is not None
    ]
    taken.sort(key=lambda pair: pair[0])

    for observation in observations:
        observed_at = _parse_iso(observation.get("observed_at"))
        previous = None
        if observed_at is not None:
            for dose_time, dose in taken:
                if dose_time <= observed_at:
                    previous = (dose_time, dose)
                else:
                    break

        if previous is None:
            observation["last_dose_at"] = None
            observation["hours_since_last_dose"] = None
            observation["last_dose_medication"] = None
            observation["last_dose_mg"] = None
            continue

        dose_time, dose = previous
        elapsed_hours = (observed_at - dose_time).total_seconds() / 3600
        observation["last_dose_at"] = dose.get("observed_at")
        observation["hours_since_last_dose"] = round(elapsed_hours, 2)
        observation["last_dose_medication"] = (
            dose.get("medication_display") or dose.get("medication_code")
        )
        observation["last_dose_mg"] = dose.get("dose_mg")

    return observations


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

    # Do not pre-filter by the backend's default organization here. The
    # caller JWT plus Supabase RLS is the authorization boundary: providers
    # see rows in organizations where they are active members, while a
    # subject can still see their own legacy/personal ParkiCheck rows. This
    # keeps the shared timeline compatible with both migrated and historical
    # ParkiCheck sessions without elevating the backend to service-role reads.
    response = requests.get(
        f"{config.url}/rest/v1/{config.table}",
        params={
            "select": OBSERVATION_SELECT,
            "subject_person_id": f"eq.{subject_person_id}",
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
