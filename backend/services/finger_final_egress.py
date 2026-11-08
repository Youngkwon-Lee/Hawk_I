"""Research candidate: score-free Finger Tapping result egress.

Only for use *after* every model/agent branch and *before* any score-based
persistence or file/API output. Retain minimal canonical auth metadata; an
unlinked/public analysis must never be made patient-linked by guessing IDs.
This is not clinical approval or a replacement for proper research trace QC.
"""
from __future__ import annotations
from typing import Any

_AUTH_FIELDS = frozenset({
    "subject_person_id", "organization_id", "activity_session_id",
    "persistence_owner", "contract_version", "created_by_person_id",
    "performer_person_id",
})
_VIDEO_FIELDS = frozenset({"width", "height", "fps", "duration", "total_frames"})
_MEDIA_FIELDS = frozenset({"original_video_url", "skeleton_video_url"})
_MAP_FIELDS = frozenset({"heatmap", "temporal_map", "attention_map"})


def is_finger_task(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    name = value.strip().casefold().replace("-", "_").replace(" ", "_")
    return name == "finger_tapping" or name.startswith("finger_tapping_")


def _fields(value: Any, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {name: value[name] for name in allowed if name in value}


def build_finger_research_hold_result(response: dict[str, Any]) -> dict[str, Any]:
    """Allowlist safe *research-review* output, not a clinical result.

    Critically preserve the subject/session/persistence owner auth selectors
    from the canonical source; stripping all context could make results
    appear unlinked and allow an unauthenticated legacy read.
    """
    if not isinstance(response, dict):
        raise ValueError("result must be an object")
    if not is_finger_task(response.get("video_type")):
        return response

    raw_context = response.get("physio_context")
    if raw_context is not None and not isinstance(raw_context, dict):
        raise ValueError("Malformed patient authorization metadata")
    if isinstance(raw_context, dict) and raw_context:
        if raw_context.get("persistence_owner") == "parkicheck":
            if not raw_context.get("activity_session_id"):
                raise ValueError("Missing ParkiCheck authorization session")
        elif not raw_context.get("subject_person_id"):
            raise ValueError("Missing patient authorization subject")

    context = _fields(raw_context, _AUTH_FIELDS)
    metadata = _fields(response.get("video_metadata"), _VIDEO_FIELDS)
    skeleton = _fields(response.get("skeleton_data"), _MEDIA_FIELDS | frozenset({
        "total_frames", "detection_rate", "mode", "fps"
    }))
    maps = _fields(response.get("visualization_urls"), _MAP_FIELDS)

    # Fail closed: all unreviewed Finger Tapping scores, interpretations and
    # unknown future inference fields are excluded rather than redacted by
    # free-text regex. Clinical zero and unmeasured/null remain distinguishable
    # only in restricted research evidence, never as an approved score here.
    return {
        "success": response.get("success") is True,
        "id": response.get("id"),
        "video_type": "finger_tapping",
        "physio_context": context or None,
        "assessment_session_id": response.get("assessment_session_id"),
        "timeline_contract_version": response.get("timeline_contract_version"),
        "video_metadata": metadata,
        "skeleton_data": skeleton,
        "visualization_urls": maps,
        "updrs_score": None,
        "clinical_score": None,
        "ai_interpretation": None,
        "score_advisory": None,
        "metrics": None,
        "reasoning": None,
        "reasoning_log": [],
        "research_review": {
            "state": "hold",
            "reasons": ["clinical_score_release_not_authorized"],
            "source": "research_only_post_inference_egress",
            "score_anchor": {"updrs_3_4": None, "source": "not_scored"},
        },
    }
