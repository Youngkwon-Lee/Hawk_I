"""Research-only fail-closed display projection for finger-tapping candidates.

This is not wired into Flask/clinical scoring/persistence. It intentionally
builds an allowlisted object instead of copying a result, because a rejected
score could leak via reports or nested VLM fields.
"""
from __future__ import annotations

from typing import Any

VERSION = "finger_tapping_review_projection_v0"


def _gate_status(record: Any, status_key: str, *, wrapper: str | None = None) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get(wrapper) if wrapper else record
    if not isinstance(value, dict):
        return None
    status = value.get(status_key)
    return status if isinstance(status, str) else None


def project_finger_review(
    analysis: dict[str, Any],
    *,
    sampling_precheck: dict[str, Any] | None,
    trace_qc: dict[str, Any] | None,
    validation_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create a sanitized research view; never release an automated score.

    Inputs must be from authorized in-process analysis, not browser-supplied
    quality flags. A passing quality check never grants clinical approval.
    """
    if not isinstance(analysis, dict) or analysis.get("video_type") != "finger_tapping":
        raise ValueError("Only finger_tapping analysis results are supported")

    reasons: list[str] = []

    sampling_status = _gate_status(sampling_precheck, "status", wrapper="precheck")
    if not isinstance(sampling_precheck, dict) or sampling_precheck.get("task") != "finger_tapping":
        reasons.append("sampling_provenance_missing_or_task_mismatch")
    if sampling_status not in {"review", "structural_precheck_pass"}:
        reasons.append("sampling_precheck_not_satisfied")

    trace_status = _gate_status(trace_qc, "status", wrapper="quality_gate")
    if not isinstance(trace_qc, dict) or trace_qc.get("task") != "finger_tapping":
        reasons.append("trace_qc_missing_or_task_mismatch")
    if trace_status != "pass":
        reasons.append("trace_quality_not_passed")

    validation_ok = (
        isinstance(validation_result, dict)
        and validation_result.get("is_valid") is True
    )
    if not validation_ok:
        reasons.append("analysis_validation_not_passed")

    state = "hold" if reasons else "human_review_required"
    # Never copy subject identifiers, score, VLM text, report, media or metrics
    # from the untrusted analysis object.
    return {
        "schema": VERSION,
        "task": "finger_tapping",
        "state": state,
        "hold_reasons": reasons,
        "quality_evidence": {
            "sampling": sampling_status,
            "trace": trace_status,
            "analysis_valid": validation_ok,
        },
        "clinical_score": None,
        "score_anchor": {"updrs_3_4": None, "source": "not_scored"},
        "clinical_report": None,
        "release_status": "not_authorized",
        "research_prediction_exposed": False,
    }
