"""Research-only frame-sampling and timestamp provenance precheck for finger tapping.

Does not extract landmarks, measure cycles, determine task compliance, infer
MDS-UPDRS scores, or replace the established hawkeye-labeling-tool trace QC.
Keep this module off patient-facing paths until owners coordinate integration.
"""
from __future__ import annotations

import math
from typing import Any

TRACE_QC_OWNER = "hawkeye-labeling-tool/scripts/analyze_finger_tapping_trace.py"


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def inspect_finger_sampling(
    *,
    task_type: str,
    source_fps: float,
    source_frame_count: int,
    configured_stride: int,
    retained_frames: list[dict[str, Any]],
    time_source: str = "source_fps_estimate",
) -> dict[str, Any]:
    """Check *structural* sampling evidence; clinical scores always stay null.

    The current MediaPipe extractor uses frame_number/source_fps for timestamps.
    That is not authoritative presentation-time evidence for variable-FPS media.
    No sampling policy is applied to the runtime by this function.
    """
    reasons: list[str] = []
    warnings: list[str] = []

    valid_meta = (
        task_type == "finger_tapping"
        and _finite_number(source_fps) and source_fps > 0
        and isinstance(source_frame_count, int)
        and not isinstance(source_frame_count, bool)
        and source_frame_count > 0
        and isinstance(configured_stride, int)
        and not isinstance(configured_stride, bool)
        and configured_stride >= 1
    )
    if task_type != "finger_tapping":
        reasons.append("wrong_task")
    if not valid_meta and task_type == "finger_tapping":
        reasons.append("invalid_video_sampling_metadata")

    source_duration = (
        source_frame_count / source_fps if valid_meta else None
    )
    expected_indices = (
        list(range(0, source_frame_count, configured_stride))
        if valid_meta else []
    )
    observed_indices: list[int] = []
    observed_times: list[float] = []

    if not isinstance(retained_frames, list):
        reasons.append("invalid_retained_trace")
    elif valid_meta:
        for sample in retained_frames:
            if not isinstance(sample, dict):
                reasons.append("invalid_sample")
                break
            frame_idx = sample.get("frame_number")
            timestamp = sample.get("timestamp")
            if (
                not isinstance(frame_idx, int) or isinstance(frame_idx, bool)
                or frame_idx < 0 or frame_idx >= source_frame_count
                or not _finite_number(timestamp) or timestamp < 0
            ):
                reasons.append("invalid_frame_index_or_timestamp")
                break
            if (
                observed_indices and frame_idx <= observed_indices[-1]
                or observed_times and timestamp <= observed_times[-1]
            ):
                reasons.append("nonmonotonic_frame_timebase")
                break
            landmarks = sample.get("landmarks")
            if not isinstance(landmarks, list) or not landmarks:
                reasons.append("unavailable_pose_landmarks")
                break
            observed_indices.append(frame_idx)
            observed_times.append(float(timestamp))

        # Fail closed on absent samples; no dropped-frame interpolation.
        if not reasons and observed_indices != expected_indices:
            reasons.append("missing_or_unexpected_retained_frames")
        if not reasons and len(observed_times) < 2:
            reasons.append("insufficient_temporal_samples")

    if valid_meta and configured_stride > 1:
        warnings.append("temporal_decimation_risk")
    if time_source != "pts_observed":
        warnings.append("source_presentation_timestamps_unverified")
    if valid_meta and not reasons and time_source == "pts_observed":
        # This flag is caller-supplied provenance, not independent verification.
        warnings.append("pts_provenance_requires_independent_verification")

    status = "hold" if reasons else "review" if warnings else "structural_precheck_pass"
    return {
        "task": task_type,
        "policy": {
            "recommended_stride": 1 if valid_meta else None,
            "purpose": "research_full_rate_finger_trace",
            "applied_to_runtime": False,
            "requires_resource_and_aliasing_validation": True,
        },
        "sampling_evidence": {
            "source_fps": float(source_fps) if valid_meta else None,
            "source_frame_count": source_frame_count if valid_meta else None,
            "source_duration_s": source_duration,
            "configured_stride": configured_stride if valid_meta else None,
            "nominal_retained_fps": (source_fps / configured_stride) if valid_meta else None,
            "expected_processed_frames": len(expected_indices) if valid_meta else None,
            "retained_pose_frames": len(retained_frames) if isinstance(retained_frames, list) else None,
            "observed_time_span_s": (
                observed_times[-1] - observed_times[0]
                if len(observed_times) >= 2 else None
            ),
            "time_source": time_source,
        },
        "precheck": {
            "status": status,
            "reasons": reasons,
            "warnings": warnings,
            "scope": "sampling_structure_only_not_clinical_quality",
        },
        "next_quality_gate_owner": TRACE_QC_OWNER,
        "score_anchor": {"updrs_3_4": None, "source": "not_scored"},
    }
