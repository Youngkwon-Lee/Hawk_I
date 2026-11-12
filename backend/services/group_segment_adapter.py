"""Pure metadata adapter for synthetic group-session segment candidates.

This does not detect people, cut video, infer clinical scores, or write records.
Output is a review queue; it is NOT a prediction-ingest payload.
"""
from __future__ import annotations

from typing import Any

SUPPORTED_TASKS = frozenset({"finger_tapping", "gait"})
ALLOWED_CANDIDATE_FIELDS = frozenset({
    "start_ms", "end_ms", "task", "side", "track_id", "identity_status",
    "performed", "occluded", "quality_status",
})


def adapt_segments(session_id: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert already-detected candidates into deterministic clip metadata.

    IDs are session-local and input-order-based; callers must freeze candidate order
    and source revision before reuse. No participant names or patient IDs accepted.
    """
    if not isinstance(session_id, str) or not session_id or not session_id.isascii() or not all(
        c.isalnum() or c in "_-" for c in session_id
    ):
        raise ValueError("session_id must be a nonempty opaque ASCII identifier")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")

    output = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be an object")
        if set(candidate) - ALLOWED_CANDIDATE_FIELDS:
            raise ValueError("unsupported candidate fields (including identifiers or media locations)")
        start, end = candidate.get("start_ms"), candidate.get("end_ms")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)) or start < 0 or end <= start:
            raise ValueError("start_ms/end_ms must be valid nonnegative integer interval")
        task = candidate.get("task")
        if not isinstance(task, str):
            raise ValueError("task must be a string")
        side = candidate.get("side", "na")
        if side not in ("left", "right", "na"):
            raise ValueError("side must be left/right/na")
        if task == "gait" and side != "na":
            raise ValueError("gait is not side-scoped")
        if task == "finger_tapping" and side == "na":
            raise ValueError("finger_tapping requires left or right")
        track = candidate.get("track_id")
        identity_ok = isinstance(track, str) and bool(track) and track.isascii() and all(
            c.isalnum() or c in "_-" for c in track
        )
        reasons = set()
        if not identity_ok or candidate.get("identity_status") != "confirmed":
            reasons.add("id_unknown")
        if candidate.get("performed") is not True:
            reasons.add("not_performed")
        if candidate.get("occluded") is True:
            reasons.add("occluded")
        if candidate.get("quality_status") != "pass":
            reasons.add("poor_quality")
        if task not in SUPPORTED_TASKS:
            reasons.add("unsupported_task")
        held = bool(reasons)
        output.append({
            "clip_id": f"{session_id}_seg_{index:04d}",
            "start_ms": start,
            "end_ms": end,
            "task": task if task in SUPPORTED_TASKS else "unknown",
            "side": side,
            "track_id": track if identity_ok and not held else None,
            "quality_gate": {"status": "hold" if held else "pass", "reasons": sorted(reasons)},
            "review_status": "pending_review",
            "model_input_eligible": not held,
            "handoff_status": "blocked_quality" if held else "awaiting_review_and_permission",
            "score_anchor": None,
            "part_iii_total": None,
            "hy_stage": None,
        })
    return output


def prepare_research_handoff(
    segments: list[dict[str, Any]],
    *,
    reviewed_clip_ids: set[str],
    approved_clip_ids: set[str],
) -> list[dict[str, Any]]:
    """Prepare metadata-only research handoff; never emit scoring or media.

    Review and permission sets are unverified caller assertions. They must NOT
    be interpreted as consent verification or authorization to send media, run
    inference, promote ground truth, or publish results.
    """
    if not isinstance(reviewed_clip_ids, set) or not isinstance(approved_clip_ids, set):
        raise ValueError("review and permission identifiers must be sets")
    result = []
    seen = set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("segment must be an object")
        clip_id = segment.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id or clip_id in seen:
            raise ValueError("clip_id must be unique and nonempty")
        seen.add(clip_id)
        if not isinstance(segment.get("quality_gate"), dict) or segment["quality_gate"].get("status") != "pass":
            continue
        if segment["quality_gate"].get("reasons") != []:
            continue
        if segment.get("model_input_eligible") is not True:
            continue
        if segment.get("review_status") != "pending_review":
            continue
        if segment.get("handoff_status") != "awaiting_review_and_permission":
            continue
        if any(segment.get(field) is not None for field in ("score_anchor", "part_iii_total", "hy_stage")):
            continue
        if clip_id not in reviewed_clip_ids or clip_id not in approved_clip_ids:
            continue
        task, side = segment.get("task"), segment.get("side")
        if task not in SUPPORTED_TASKS:
            continue
        if (task == "gait" and side != "na") or (task == "finger_tapping" and side not in ("left", "right")):
            continue
        start, end = segment.get("start_ms"), segment.get("end_ms")
        if any(type(v) is not int for v in (start, end)) or start < 0 or end <= start:
            continue
        result.append({
            "clip_id": clip_id,
            "task": segment["task"],
            "side": segment["side"],
            "start_ms": segment["start_ms"],
            "end_ms": segment["end_ms"],
            "handoff_status": "reviewed_permission_asserted_unverified",
        })
    return result


def candidates_from_observation_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group pre-annotated synthetic windows; NOT automatic video understanding.

    No interpolation across gaps or ambiguous identity/task transitions. A caller
    must provide explicit time windows and classifications; this function never
    estimates person identity, exercise type, or clinical features.
    """
    allowed = ALLOWED_CANDIDATE_FIELDS
    if not isinstance(windows, list):
        raise ValueError("windows must be a list")
    ordered = []
    for window in windows:
        if not isinstance(window, dict) or set(window) - allowed:
            raise ValueError("window has unsupported fields")
        start, end = window.get("start_ms"), window.get("end_ms")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("invalid window timing")
        ordered.append(dict(window))
    # Simultaneous participants are expected in group sessions. Reject overlap
    # only within the same known track; unknown tracks are never merged.
    ordered.sort(key=lambda w: (w["start_ms"], w["end_ms"], str(w.get("track_id"))))
    result: list[dict[str, Any]] = []
    last_by_track: dict[str, dict[str, Any]] = {}
    for window in ordered:
        track = window.get("track_id")
        known_track = isinstance(track, str) and bool(track) and track.isascii() and all(
            c.isalnum() or c in "_-" for c in track
        ) and window.get("identity_status") == "confirmed"
        if known_track:
            previous = last_by_track.get(track)
            if previous and window["start_ms"] < previous["end_ms"]:
                raise ValueError("overlapping windows on the same track require review")
            comparable = {key: value for key, value in window.items() if key not in ("start_ms", "end_ms")}
            if previous and previous["end_ms"] == window["start_ms"] and {
                key: value for key, value in previous.items() if key not in ("start_ms", "end_ms")
            } == comparable:
                previous["end_ms"] = window["end_ms"]
                continue
            last_by_track[track] = window
        result.append(window)
    return result


def windows_from_tracker_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map external tracker *claims* to held review windows, without inference.

    Tracker IDs are ephemeral and do not prove a person's identity. Even a
    stable track is held until a separate, trusted review confirms identity,
    task, and quality; this adapter never supplies those confirmations.
    """
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    windows = []
    allowed = {"start_ms", "end_ms", "track_id", "track_state", "task", "side"}
    for obs in observations:
        if not isinstance(obs, dict) or set(obs) - allowed:
            raise ValueError("unsupported tracker observation fields")
        start, end = obs.get("start_ms"), obs.get("end_ms")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("invalid tracker observation interval")
        if obs.get("track_state") not in ("stable", "lost", "switched", "ambiguous"):
            raise ValueError("unknown tracker state")
        task = obs.get("task", "unknown")
        side = obs.get("side", "na")
        if not isinstance(task, str) or side not in ("left", "right", "na"):
            raise ValueError("invalid task or side")
        # Never treat external tracker state as identity verification.
        windows.append({
            "start_ms": start, "end_ms": end, "track_id": obs.get("track_id"),
            "identity_status": "unverified", "task": task, "side": side,
            "performed": None, "occluded": obs["track_state"] != "stable",
            "quality_status": "hold",
        })
    return windows


def tracker_frames_to_observations(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize precomputed per-frame tracker outputs into held time windows.

    No model is loaded and no identity, activity, or video content is inferred.
    Frame intervals must be supplied by the upstream tracker (milliseconds).
    """
    if not isinstance(frames, list):
        raise ValueError("frames must be a list")
    allowed_frame = {"start_ms", "end_ms", "tracks"}
    allowed_track = {"track_id", "state"}
    observations: list[dict[str, Any]] = []
    previous_end = None
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) - allowed_frame:
            raise ValueError("unsupported frame fields")
        start, end, tracks = frame.get("start_ms"), frame.get("end_ms"), frame.get("tracks")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("invalid frame interval")
        if previous_end is not None and start < previous_end:
            raise ValueError("frame intervals must be ordered and nonoverlapping")
        previous_end = end
        if not isinstance(tracks, list):
            raise ValueError("tracks must be a list")
        seen_tracks: set[str] = set()
        for track in tracks:
            if not isinstance(track, dict) or set(track) - allowed_track:
                raise ValueError("unsupported track fields")
            track_id, state = track.get("track_id"), track.get("state")
            if not isinstance(track_id, str) or not track_id or not track_id.isascii() or not all(
                c.isalnum() or c in "_-" for c in track_id
            ):
                raise ValueError("track_id must be an opaque ASCII token")
            if track_id in seen_tracks:
                raise ValueError("duplicate track in one frame")
            seen_tracks.add(track_id)
            if state not in ("stable", "lost", "switched", "ambiguous"):
                raise ValueError("unknown track state")
            observations.append({
                "start_ms": start, "end_ms": end, "track_id": track_id,
                "track_state": state, "task": "unknown", "side": "na",
            })
    return observations


def build_tracker_review_queue(session_id: str, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """End-to-end *metadata only* tracker-output -> held review queue.

    No video decoding, identity verification, task inference, model invocation,
    approval, or persistence occurs here. All tracker-derived rows are held.
    """
    observations = tracker_frames_to_observations(frames)
    windows = windows_from_tracker_observations(observations)
    candidates = candidates_from_observation_windows(windows)
    rows = adapt_segments(session_id, candidates)
    if any(row["quality_gate"]["status"] != "hold" for row in rows):
        raise ValueError("tracker-only inputs cannot pass the review gate")
    return rows
