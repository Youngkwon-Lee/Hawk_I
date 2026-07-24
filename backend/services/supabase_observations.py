"""
Optional Supabase persistence for Hawkeye analysis results.

The existing file-based result store remains the source for the current UI.
When server-side Supabase credentials and a physio_app person/org context are
provided, completed analyses are also written to public.activity_sessions and
public.observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from numbers import Real
import os
from typing import Any

import requests


TASK_CODES = {
    "finger_tapping": {
        "code": "UPDRS_3_4",
        "display": "UPDRS Part III Item 3.4 - Finger Tapping",
    },
    "hand_movement": {
        "code": "UPDRS_3_5",
        "display": "UPDRS Part III Item 3.5 - Hand Movements",
    },
    "pronation_supination": {
        "code": "UPDRS_3_6",
        "display": "UPDRS Part III Item 3.6 - Pronation-Supination",
    },
    "leg_agility": {
        "code": "UPDRS_3_8",
        "display": "UPDRS Part III Item 3.8 - Leg Agility",
    },
    "gait": {
        "code": "UPDRS_3_9",
        "display": "UPDRS Part III Item 3.9 - Gait",
    },
}


@dataclass(frozen=True)
class SupabaseObservationResult:
    enabled: bool
    saved: bool
    table: str = "observations"
    reason: str | None = None
    observation_id: str | None = None
    activity_session_id: str | None = None
    status_code: int | None = None

    def as_public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "saved": self.saved,
            "table": self.table,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.observation_id:
            payload["observation_id"] = self.observation_id
        if self.activity_session_id:
            payload["activity_session_id"] = self.activity_session_id
        if self.status_code:
            payload["status_code"] = self.status_code
        return payload


@dataclass(frozen=True)
class SupabaseObservationConfig:
    url: str
    key: str
    subject_person_id: str | None
    organization_id: str
    created_by: str
    performer_person_id: str
    activity_session_id: str | None = None
    table: str = "observations"
    activity_sessions_table: str = "activity_sessions"
    timeout_seconds: float = 8.0


def _get_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _get_result_context_value(result: dict[str, Any], *names: str) -> str | None:
    context = result.get("physio_context")
    context_dict = context if isinstance(context, dict) else {}
    for name in names:
        value = result.get(name)
        if value is None:
            value = context_dict.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def get_supabase_observation_config() -> SupabaseObservationConfig | None:
    enabled = os.getenv("HAWKEYE_SUPABASE_ENABLED", "").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None

    url = _get_env("HAWKEYE_SUPABASE_URL", "SUPABASE_URL")
    key = _get_env(
        "HAWKEYE_SUPABASE_SERVICE_KEY",
        "SUPABASE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    )
    subject_person_id = _get_env(
        "HAWKEYE_SUPABASE_SUBJECT_PERSON_ID",
        "HAWKEYE_SUPABASE_PERSON_ID",
    )
    organization_id = _get_env(
        "HAWKEYE_SUPABASE_ORGANIZATION_ID",
        "HAWKEYE_SUPABASE_ORG_ID",
    )
    created_by = _get_env(
        "HAWKEYE_SUPABASE_CREATED_BY_PERSON_ID",
        "HAWKEYE_SUPABASE_CREATED_BY",
    ) or subject_person_id
    performer_person_id = _get_env(
        "HAWKEYE_SUPABASE_PERFORMER_PERSON_ID",
        "HAWKEYE_SUPABASE_PERFORMER_ID",
    ) or created_by
    activity_session_id = _get_env("HAWKEYE_SUPABASE_ACTIVITY_SESSION_ID")

    if not (url and key and organization_id and created_by and performer_person_id):
        return None

    timeout_raw = os.getenv("HAWKEYE_SUPABASE_TIMEOUT_SECONDS", "8")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 8.0

    return SupabaseObservationConfig(
        url=url.rstrip("/"),
        key=key,
        subject_person_id=subject_person_id,
        organization_id=organization_id,
        created_by=created_by,
        performer_person_id=performer_person_id,
        activity_session_id=activity_session_id,
        table=os.getenv("HAWKEYE_SUPABASE_OBSERVATIONS_TABLE", "observations").strip()
        or "observations",
        activity_sessions_table=os.getenv(
            "HAWKEYE_SUPABASE_ACTIVITY_SESSIONS_TABLE", "activity_sessions"
        ).strip()
        or "activity_sessions",
        timeout_seconds=timeout,
    )


def config_for_analysis_result(
    config: SupabaseObservationConfig,
    result: dict[str, Any],
) -> SupabaseObservationConfig:
    subject_person_id = _get_result_context_value(
        result,
        "supabase_subject_person_id",
        "physio_subject_person_id",
        "subject_person_id",
    ) or config.subject_person_id
    organization_id = _get_result_context_value(
        result,
        "supabase_organization_id",
        "physio_organization_id",
        "organization_id",
    ) or config.organization_id
    created_by = _get_result_context_value(
        result,
        "supabase_created_by_person_id",
        "physio_created_by_person_id",
        "created_by_person_id",
    ) or config.created_by
    performer_person_id = _get_result_context_value(
        result,
        "supabase_performer_person_id",
        "physio_performer_person_id",
        "performer_person_id",
    ) or config.performer_person_id or created_by

    return replace(
        config,
        subject_person_id=subject_person_id,
        organization_id=organization_id,
        created_by=created_by,
        performer_person_id=performer_person_id,
    )


def has_explicit_subject_context(result: dict[str, Any]) -> bool:
    subject_person_id = _get_result_context_value(
        result,
        "supabase_subject_person_id",
        "physio_subject_person_id",
        "subject_person_id",
    )
    organization_id = _get_result_context_value(
        result,
        "supabase_organization_id",
        "physio_organization_id",
        "organization_id",
    )
    return bool(subject_person_id and organization_id)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return int(value) if isinstance(value, int) else numeric
    if isinstance(value, str):
        return value
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _score_value(result: dict[str, Any]) -> float | None:
    score = result.get("updrs_score") or {}
    if not isinstance(score, dict):
        return None
    value = score.get("total_score", score.get("score"))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _score_fields(score: float) -> dict[str, Any]:
    if abs(score - round(score)) < 1e-6:
        return {"value_type": "integer", "value_integer": int(round(score))}
    return {"value_type": "quantity", "value_quantity": score, "value_unit": "score"}


def _interpretation(result: dict[str, Any]) -> str | None:
    score = _score_value(result)
    if score is None:
        return None
    return "normal" if score < 0.5 else "abnormal"


def _task_meta(video_type: str) -> dict[str, str]:
    return TASK_CODES.get(
        video_type,
        {
            "code": f"HAWKEYE_UPDRS_{video_type or 'unknown'}",
            "display": f"Hawkeye UPDRS Motor Analysis - {video_type or 'unknown'}",
        },
    )


def _analysis_id(result: dict[str, Any]) -> str:
    return str(result.get("id") or result.get("video_id") or "")


def build_activity_session_row(
    result: dict[str, Any],
    config: SupabaseObservationConfig,
) -> dict[str, Any]:
    video_type = str(result.get("video_type") or "unknown")
    analysis_id = _analysis_id(result)
    score = _score_value(result)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    row = {
        "subject_person_id": config.subject_person_id,
        "organization_id": config.organization_id,
        "created_by": config.created_by,
        "activity_type": "assessment",
        "source": "camera",
        "status": "completed",
        "performed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metrics": _json_safe(
            {
                "app_source": "hawk_i",
                "analysis_id": analysis_id or None,
                "patient_id": result.get("patient_id"),
                "video_type": video_type,
                "score_raw": score,
                "confidence": result.get("confidence"),
                "scoring_method": result.get("scoring_method"),
                "ml_model_type": result.get("ml_model_type"),
                "metrics": metrics,
            }
        ),
        "exercise_log": _json_safe(
            {
                "task_code": _task_meta(video_type)["code"],
                "task_display": _task_meta(video_type)["display"],
                "events": result.get("events"),
            }
        ),
        "notes": f"Hawkeye VLM motor assessment: {video_type}",
    }
    duration_seconds = result.get("duration_seconds")
    try:
        duration = int(float(duration_seconds))
    except (TypeError, ValueError):
        duration = None
    if duration is not None and duration >= 0:
        row["duration_seconds"] = duration
    return {key: value for key, value in row.items() if value is not None}


def build_observation_row(
    result: dict[str, Any],
    config: SupabaseObservationConfig,
    activity_session_id: str | None = None,
) -> dict[str, Any]:
    video_type = str(result.get("video_type") or "unknown")
    task = _task_meta(video_type)
    score = _score_value(result)
    if score is None:
        raise ValueError("analysis result does not contain an UPDRS score")

    session_id = activity_session_id or config.activity_session_id
    if not session_id:
        raise ValueError("activity session context is required")

    analysis_id = _analysis_id(result)
    score_payload = result.get("updrs_score") if isinstance(result.get("updrs_score"), dict) else {}
    skeleton_data = result.get("skeleton_data") if isinstance(result.get("skeleton_data"), dict) else {}

    measurement_context = {
        "app_source": "hawk_i",
        "analysis_id": analysis_id or None,
        "patient_id": result.get("patient_id"),
        "physio_context": result.get("physio_context"),
        "video_type": video_type,
        "auto_detected": result.get("auto_detected"),
        "confidence": result.get("confidence"),
        "scoring_method": result.get("scoring_method"),
        "ml_model_type": result.get("ml_model_type"),
        "score_raw": score,
        "score_confidence": score_payload.get("confidence"),
        "severity": score_payload.get("severity"),
        "score_details": score_payload.get("details"),
        "metrics": result.get("metrics"),
        "performability_assessment": result.get("performability_assessment"),
        "score_advisory": result.get("score_advisory"),
        "events": result.get("events"),
        "motion_analysis": result.get("motion_analysis"),
        "skeleton_summary": {
            "total_frames": skeleton_data.get("total_frames"),
            "detection_rate": skeleton_data.get("detection_rate"),
            "mode": skeleton_data.get("mode"),
            "skeleton_video_url": skeleton_data.get("skeleton_video_url"),
            "original_video_url": skeleton_data.get("original_video_url"),
            "fps": skeleton_data.get("fps"),
        },
        "visualization_urls": result.get("visualization_urls"),
    }

    row = {
        "fhir_id": f"hawkeye-{analysis_id}" if analysis_id else None,
        "subject_person_id": config.subject_person_id,
        "organization_id": config.organization_id,
        "created_by": config.created_by,
        "performer_person_id": config.performer_person_id,
        "activity_session_id": session_id,
        "status": "final",
        "source_type": "ai",
        "code": task["code"],
        "code_system": "http://www.nih.gov/updrs",
        "code_display": task["display"],
        "category": ["motor-assessment", "neurological", "hawkeye"],
        "interpretation": _interpretation(result),
        "measurement_context": _json_safe(measurement_context),
        "effective_datetime": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    row.update(_score_fields(score))
    return {key: value for key, value in row.items() if value is not None}


def _post_row(
    config: SupabaseObservationConfig,
    table: str,
    row: dict[str, Any],
) -> requests.Response:
    return requests.post(
        f"{config.url}/rest/v1/{table}",
        headers={
            "apikey": config.key,
            "Authorization": f"Bearer {config.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        json=row,
        timeout=config.timeout_seconds,
    )


def _extract_returned_id(response: requests.Response) -> str | None:
    try:
        data = response.json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            returned_id = data[0].get("id")
            return str(returned_id) if returned_id else None
    except ValueError:
        return None
    return None


def save_analysis_observation(result: dict[str, Any]) -> SupabaseObservationResult:
    config = get_supabase_observation_config()
    if not config:
        return SupabaseObservationResult(
            enabled=False,
            saved=False,
            reason="missing Supabase URL/key or organization/operator context",
        )
    if not has_explicit_subject_context(result):
        return SupabaseObservationResult(
            enabled=True,
            saved=False,
            table=config.table,
            reason="missing explicit physio_app subject context",
        )
    config = config_for_analysis_result(config, result)

    activity_session_id = config.activity_session_id

    try:
        if not activity_session_id:
            session_row = build_activity_session_row(result, config)
            session_response = _post_row(config, config.activity_sessions_table, session_row)
            if session_response.status_code >= 400:
                return SupabaseObservationResult(
                    enabled=True,
                    saved=False,
                    table=config.table,
                    status_code=session_response.status_code,
                    reason=f"activity session insert failed: {session_response.text[:260]}",
                )
            activity_session_id = _extract_returned_id(session_response)
            if not activity_session_id:
                return SupabaseObservationResult(
                    enabled=True,
                    saved=False,
                    table=config.table,
                    status_code=session_response.status_code,
                    reason="activity session insert did not return an id",
                )

        row = build_observation_row(result, config, activity_session_id)
        response = _post_row(config, config.table, row)
    except requests.RequestException as exc:
        return SupabaseObservationResult(
            enabled=True,
            saved=False,
            table=config.table,
            reason=f"request failed: {exc}",
        )
    except Exception as exc:
        return SupabaseObservationResult(
            enabled=True,
            saved=False,
            table=config.table,
            activity_session_id=activity_session_id,
            reason=f"invalid analysis result: {exc}",
        )

    if response.status_code >= 400:
        return SupabaseObservationResult(
            enabled=True,
            saved=False,
            table=config.table,
            activity_session_id=activity_session_id,
            status_code=response.status_code,
            reason=response.text[:300],
        )

    observation_id = _extract_returned_id(response)

    return SupabaseObservationResult(
        enabled=True,
        saved=True,
        table=config.table,
        observation_id=observation_id,
        activity_session_id=activity_session_id,
        status_code=response.status_code,
    )
