"""Validation and descriptive timing for patient-reported medication context."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Any


MAX_CONTEXT_BYTES = 4096
MAX_MEDICATION_NAME_LENGTH = 100
MAX_DOSE_MG = 100_000


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def parse_medication_context(raw_value: str | None) -> dict[str, Any] | None:
    """Parse a small, whitelisted patient-reported context from multipart input."""
    if raw_value is None or raw_value == "":
        return None
    if not isinstance(raw_value, str) or len(raw_value.encode("utf-8")) > MAX_CONTEXT_BYTES:
        raise ValueError("medication_context is too large")

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("medication_context must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("medication_context must be a JSON object")

    context: dict[str, Any] = {
        "available": payload.get("available") is True,
        "source": "patient_reported_local",
    }

    assessment_at = None
    if payload.get("assessment_at"):
        assessment_at = _parse_datetime(payload["assessment_at"], "assessment_at")
        context["assessment_at"] = _isoformat(assessment_at)

    if not context["available"]:
        return context

    taken_at = _parse_datetime(payload.get("taken_at"), "taken_at")
    context["taken_at"] = _isoformat(taken_at)

    medication = payload.get("medication")
    context["medication"] = (
        medication.strip()[:MAX_MEDICATION_NAME_LENGTH]
        if isinstance(medication, str) and medication.strip()
        else None
    )

    dose_mg = payload.get("dose_mg")
    if isinstance(dose_mg, bool):
        dose_mg = None
    try:
        dose_value = float(dose_mg) if dose_mg is not None else None
    except (TypeError, ValueError):
        dose_value = None
    context["dose_mg"] = (
        dose_value
        if dose_value is not None and math.isfinite(dose_value) and 0 <= dose_value <= MAX_DOSE_MG
        else None
    )

    if assessment_at is not None:
        elapsed = (assessment_at - taken_at).total_seconds() / 3600
    else:
        supplied_elapsed = payload.get("hours_before_assessment")
        try:
            elapsed = float(supplied_elapsed)
        except (TypeError, ValueError):
            elapsed = math.nan
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("medication taken_at must not be after the assessment")
    context["hours_before_assessment"] = round(elapsed, 2)
    return context


def describe_medication_timing(context: dict[str, Any] | None) -> dict[str, Any]:
    """Describe temporal proximity without inferring efficacy or an ON/OFF state."""
    if not context or context.get("available") is not True:
        return {
            "available": False,
            "evidence_level": "none",
            "can_infer_medication_effect": False,
        }

    hours = context.get("hours_before_assessment")
    if hours is None:
        window = "unknown"
    elif hours <= 2:
        window = "within_2_hours"
    elif hours <= 6:
        window = "between_2_and_6_hours"
    else:
        window = "over_6_hours"
    return {
        "available": True,
        "relationship": "after_patient_reported_dose",
        "hours_after_reported_dose": hours,
        "timing_window": window,
        "evidence_level": "single_observation",
        "can_infer_medication_effect": False,
    }
