"""Observed repeated-assessment comparison without medication-effect inference."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _date_key(value: Any) -> datetime:
    if not isinstance(value, str):
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def _observation(result: dict[str, Any]) -> dict[str, Any] | None:
    context = result.get("medication_context")
    timing = result.get("medication_timing")
    score = _finite(result.get("score"))
    if not isinstance(context, dict) or context.get("available") is not True or score is None:
        return None
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    hours = None
    if isinstance(timing, dict):
        hours = _finite(timing.get("hours_after_reported_dose"))
    if hours is None:
        hours = _finite(context.get("hours_before_assessment"))
    return {
        "video_id": result.get("video_id"),
        "date": context.get("assessment_at") or result.get("date"),
        "patient_id": result.get("patient_id") or "anonymous",
        "task_type": result.get("task_type") or "unknown",
        "medication": context.get("medication") or "약물명 미입력",
        "dose_mg": _finite(context.get("dose_mg")),
        "hours_after_reported_dose": hours,
        "score": score,
        "tapping_speed": _finite(metrics.get("tapping_speed")),
        "amplitude_mean": _finite(metrics.get("amplitude_mean")),
        "fatigue_rate": _finite(metrics.get("fatigue_rate")),
    }


def build_medication_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [item for result in results if (item := _observation(result))]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in observations:
        key = (
            item["patient_id"],
            item["task_type"],
            item["medication"],
            item["dose_mg"],
        )
        groups.setdefault(key, []).append(item)

    candidates = []
    for group in groups.values():
        group.sort(key=lambda item: _date_key(item.get("date")))
        candidates.append(group)
    candidates.sort(
        key=lambda group: (len(group), _date_key(group[-1].get("date"))),
        reverse=True,
    )
    selected = candidates[0] if candidates else []
    if len(selected) < 2:
        return {
            "available": False,
            "observation_count": len(selected),
            "reason": "needs_repeated_comparable_assessments",
            "can_infer_medication_effect": False,
        }

    first, latest = selected[0], selected[-1]

    def delta(field: str) -> float | None:
        before, after = first.get(field), latest.get(field)
        if before is None or after is None:
            return None
        return round(after - before, 2)

    return {
        "available": True,
        "observation_count": len(selected),
        "patient_id": latest["patient_id"],
        "task_type": latest["task_type"],
        "medication": latest["medication"],
        "dose_mg": latest["dose_mg"],
        "first": first,
        "latest": latest,
        "observed_change": {
            "score": delta("score"),
            "tapping_speed": delta("tapping_speed"),
            "amplitude_mean": delta("amplitude_mean"),
            "fatigue_rate": delta("fatigue_rate"),
        },
        "evidence_level": "observational_repeated_assessments",
        "can_infer_medication_effect": False,
        "requires_clinician_review": True,
    }
