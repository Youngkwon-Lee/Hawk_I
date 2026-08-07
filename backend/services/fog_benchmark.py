"""Ask a vision-language model when freezing of gait occurs, and grade the timing.

Freezing is too rare in this cohort to train on - 28 positive clips out of 426 -
but that is enough to *benchmark*. And unlike the turn markings, freezing spans
are real durations (median 6.3s), so temporal overlap is a meaningful score.

The request builder and the response parser are pure functions so they can be
tested without a network call. Sending clinical video to a third-party API is a
data-governance decision, not a technical one, so the caller must pass explicit
approval before anything leaves the machine.
"""

from __future__ import annotations

import json
import re
from typing import Any

FREEZING_PROMPT = """You are assisting a movement-disorders research study on gait video.

Watch the frames in order. They are sampled uniformly from a single walking clip
of {duration:.1f} seconds, so frame i corresponds to approximately
i * {duration:.1f} / {n_frames} seconds.

Identify every episode of FREEZING OF GAIT: a sudden, transient inability to
start or continue stepping, with the feet appearing stuck while the intention to
walk continues. Trembling in place and very short shuffling steps that do not
advance count as freezing. Ordinary slowness, small steps that still advance,
and deliberate stopping do NOT count.

Report only what is visible. If you see no freezing, return an empty list.

Respond with JSON only, no prose:
{{"freezing_intervals": [{{"start": <seconds>, "end": <seconds>, "note": "<short observation>"}}]}}"""


class DataUseNotApproved(RuntimeError):
    """Raised when a request would send clinical video to a third party without approval."""


def build_prompt(duration_sec: float, n_frames: int) -> str:
    """The prompt carries the timebase, since the model only sees ordered frames."""
    if duration_sec <= 0 or n_frames <= 0:
        raise ValueError("duration and frame count must be positive")
    return FREEZING_PROMPT.format(duration=duration_sec, n_frames=n_frames)


def _coerce_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("s", "")
    if not text:
        return None
    # accept "1:05" as well as "65"
    if ":" in text:
        parts = text.split(":")
        try:
            return sum(float(p) * 60 ** (len(parts) - 1 - i) for i, p in enumerate(parts))
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_freezing_response(text: str, duration_sec: float | None = None) -> list[dict[str, Any]]:
    """Pull intervals out of a model reply, tolerating code fences and stray prose.

    Anything unparseable is dropped rather than guessed at: a benchmark that
    invents intervals when the model rambles would flatter the model.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace:
            candidate = brace.group(0)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    raw = payload.get("freezing_intervals")
    if not isinstance(raw, list):
        return []

    intervals: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        start = _coerce_seconds(entry.get("start"))
        end = _coerce_seconds(entry.get("end"))
        if start is None or end is None or end <= start or start < 0:
            continue
        if duration_sec is not None:
            if start >= duration_sec:
                continue
            end = min(end, duration_sec)
        interval: dict[str, Any] = {"start": round(start, 2), "end": round(end, 2)}
        note = entry.get("note")
        if isinstance(note, str) and note.strip():
            interval["note"] = note.strip()
        intervals.append(interval)

    return sorted(intervals, key=lambda i: i["start"])


def build_prediction_record(
    clip_id: str,
    intervals: list[dict[str, Any]],
    model: str,
    dataset: str = "PD4T",
    split: str | None = None,
) -> dict[str, Any]:
    """Emit the shape the existing evaluator reads, so grading needs no new code."""
    record: dict[str, Any] = {
        "clip_id": clip_id,
        "task": "gait",
        "dataset": dataset,
        "model": model,
        "condition": "commercial_zero_shot",
        "gait_events": {"freezing_intervals": intervals},
        "primitives": {
            "freezing_of_gait": {
                "observability": "observed",
                "severity": 1 if intervals else 0,
                "confidence": "low",
                "evidence": [
                    {"start_sec": i["start"], "end_sec": i["end"]} for i in intervals
                ],
            }
        },
    }
    if split:
        record["split"] = split
    return record


def require_data_use_approval(approved: bool, destination: str) -> None:
    """Third-party inference is a disclosure. Make the caller say so out loud."""
    if not approved:
        raise DataUseNotApproved(
            f"sending clip frames to {destination} would disclose study video to a third party. "
            "Confirm the dataset's terms permit it, then re-run with the approval flag."
        )
