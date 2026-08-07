"""Turn workbench label records into a training-ready shape.

The labeling UI stores severities as text in two vocabularies and keeps turn and
freezing spans under gait_events. Training code should not have to know that, so
the conversion happens once, here, with the ambiguous cases made explicit rather
than smoothed over:

- A blank severity stays null. The ontology treats "not observed" as different
  from "normal", and collapsing the two would teach the model that an occluded
  turn is a healthy turn.
- turning_quality is dropped from the training fields entirely: it is blank in
  effectively every clip, so it carries no signal. Its spans are kept, because
  those are 96% populated and are the real asset.
"""

from __future__ import annotations

from typing import Any

from services.primitive_eval import GAIT_LABEL_TO_PRIMITIVE, parse_severity

# turning_quality is excluded: 0.4% filled in training, so there is nothing to learn.
UNUSABLE_FIELDS = {"turning_quality"}

TRAINABLE_FIELDS = [f for f in GAIT_LABEL_TO_PRIMITIVE if f not in UNUSABLE_FIELDS]

SPLIT_BY_PROJECT = {
    "pd4t_train": "train",
    "pd4t_validation": "validation",
    "pd4t": "test",
}


def _intervals(events: Any, key: str) -> list[dict[str, float]]:
    if not isinstance(events, dict):
        return []
    spans = events.get(key)
    if not isinstance(spans, list):
        return []
    out = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        start, end = parse_severity(span.get("start")), parse_severity(span.get("end"))
        if start is None or end is None or end < start:
            continue
        out.append({"start": round(start, 2), "end": round(end, 2)})
    return sorted(out, key=lambda s: s["start"])


def to_training_record(label: dict[str, Any], split: str) -> dict[str, Any]:
    """One labeled clip, with text severities resolved and spans normalised."""
    events = label.get("gait_events")

    primitives: dict[str, Any] = {}
    for field in TRAINABLE_FIELDS:
        primitive = GAIT_LABEL_TO_PRIMITIVE[field]
        # None means not observed, and must not be read as a zero.
        primitives[primitive] = parse_severity(label.get(field))

    score = parse_severity(label.get("updrs_score"))

    return {
        "clip_id": label.get("clip_id") or label.get("id"),
        "patient_id": str(label.get("patient_id") or ""),
        "split": split,
        "task": "gait",
        "media_path": label.get("source_clip_path"),
        "updrs_3_10": None if score is None else int(score),
        "primitives": primitives,
        "turning_intervals": _intervals(events, "turning_intervals"),
        "freezing_intervals": _intervals(events, "freezing_intervals"),
        "annotator": label.get("annotator"),
        "annotator_confidence": label.get("confidence") or None,
        "note": (label.get("clinical_note") or label.get("memo") or "") or None,
    }


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts a trainer needs before choosing a loss: class balance and coverage."""
    by_split: dict[str, dict[str, Any]] = {}
    for record in records:
        split = record["split"]
        bucket = by_split.setdefault(split, {
            "clips": 0, "patients": set(), "score_counts": {},
            "primitive_filled": {p: 0 for p in GAIT_LABEL_TO_PRIMITIVE.values() if p in record["primitives"]},
            "clips_with_turning": 0, "clips_with_freezing": 0,
        })
        bucket["clips"] += 1
        bucket["patients"].add(record["patient_id"])
        score = record["updrs_3_10"]
        key = "null" if score is None else str(score)
        bucket["score_counts"][key] = bucket["score_counts"].get(key, 0) + 1
        for primitive, value in record["primitives"].items():
            if value is not None:
                bucket["primitive_filled"][primitive] = bucket["primitive_filled"].get(primitive, 0) + 1
        if record["turning_intervals"]:
            bucket["clips_with_turning"] += 1
        if record["freezing_intervals"]:
            bucket["clips_with_freezing"] += 1

    for bucket in by_split.values():
        bucket["patients"] = len(bucket["patients"])
    return by_split


def patient_overlap(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Any patient in two splits invalidates the evaluation, so report it loudly."""
    patients_by_split: dict[str, set[str]] = {}
    for record in records:
        patients_by_split.setdefault(record["split"], set()).add(record["patient_id"])

    overlaps: dict[str, list[str]] = {}
    splits = sorted(patients_by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            shared = sorted(patients_by_split[left] & patients_by_split[right])
            if shared:
                overlaps[f"{left}|{right}"] = shared
    return overlaps
