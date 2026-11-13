"""Read Kinelo Research's ``hawkeye-training/v1`` JSONL export.

Kinelo exports reviewer-approved gait labels (single rater, unanimous pair or
adjudicated) one clip per line. This module turns each line into the same shape
``label_export.to_training_record()`` produces, so ``summarise()`` and
``patient_overlap()`` work on Kinelo data unchanged.

The rules mirror label_export:

- A null primitive stays null. In the export null means "not observable or not
  rated", never 0. A trainable primitive the export did not carry at all (TULIP
  only rates turning) is also null, for the same reason.
- turning_impairment is dropped, because turning_quality is not a trainable field.
- Null interval lists become ``[]`` so trainers can iterate them, but each record
  keeps ``assessable`` and ``unassessable_clips()`` lists the clips whose empty
  spans mean "could not be assessed" rather than "none marked".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from services.label_export import TRAINABLE_FIELDS, _intervals
from services.primitive_eval import GAIT_LABEL_TO_PRIMITIVE

FORMAT = "hawkeye-training/v1"

TRAINABLE_PRIMITIVES = [GAIT_LABEL_TO_PRIMITIVE[f] for f in TRAINABLE_FIELDS]

PROVENANCE_FIELDS = (
    "annotators",
    "adjudicated",
    "adjudicator_id",
    "resolution",
    "dataset_schema",
    "schema_version",
    "schema_sha256",
    "media_sha256",
)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"primitive value must be a number or null, got {value!r}")
    return float(value)


def from_kinelo_record(line: dict[str, Any]) -> dict[str, Any]:
    """One Kinelo export line in the to_training_record() shape, plus provenance."""
    fmt = line.get("format")
    if fmt != FORMAT:
        raise ValueError(f"unsupported export format {fmt!r}; expected {FORMAT!r}")

    source = line.get("primitives") or {}
    primitives = {name: _float_or_none(source.get(name)) for name in TRAINABLE_PRIMITIVES}

    score = line.get("updrs_3_10")
    if score is not None and (isinstance(score, bool) or not isinstance(score, int)):
        raise ValueError(f"updrs_3_10 must be an int or null, got {score!r}")

    annotators = list(line.get("annotators") or [])
    events = {
        "turning_intervals": line.get("turning_intervals"),
        "freezing_intervals": line.get("freezing_intervals"),
    }

    record = {
        "clip_id": line.get("clip_id"),
        "patient_id": str(line.get("subject_id") or ""),
        "split": line.get("split"),
        "task": line.get("task") or "gait",
        "media_path": None,
        "updrs_3_10": score,
        "primitives": primitives,
        "turning_intervals": _intervals(events, "turning_intervals"),
        "freezing_intervals": _intervals(events, "freezing_intervals"),
        "annotator": annotators[0] if len(annotators) == 1 else None,
        "annotator_confidence": line.get("annotator_confidence") or None,
        "note": None,
        "assessable": bool(line.get("assessable", True)),
        "video_quality": line.get("video_quality"),
        "view": line.get("view"),
    }
    for field in PROVENANCE_FIELDS:
        record[field] = line.get(field)
    record["annotators"] = annotators
    return record


def load_kinelo_export(path: str | Path) -> list[dict[str, Any]]:
    """Load a hawkeye-training/v1 JSONL file. Any line in another format fails the load."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                records.append(from_kinelo_record(json.loads(raw)))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return records


def unassessable_clips(records: Iterable[dict[str, Any]]) -> list[str]:
    """Clips whose empty interval lists mean 'not assessable', not 'none marked'."""
    return sorted(r["clip_id"] for r in records if not r.get("assessable", True))
