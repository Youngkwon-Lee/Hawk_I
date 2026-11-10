"""Conservative, dependency-free association of synthetic detection boxes.

Not a detector, biometric identity system, or clinically validated tracker.
Output states are *claims* and downstream review always holds them.
"""
from __future__ import annotations

from typing import Any


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def associate_synthetic_boxes(frames: list[dict[str, Any]], *, min_iou: float = 0.3) -> list[dict[str, Any]]:
    """Associate adjacent-frame bounding boxes; ambiguous matches are held.

    Boxes use normalized (x_min, y_min, x_max, y_max) coordinates. An absent
    detection or frame gap invalidates continuity. Never infer real identity.
    """
    if type(min_iou) is not float or not 0.0 < min_iou < 1.0:
        raise ValueError("min_iou must be a float between zero and one")
    if not isinstance(frames, list):
        raise ValueError("frames must be a list")
    previous: list[tuple[str, tuple[float, ...]]] = []
    previous_end = None
    next_id = 1
    output = []
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {"start_ms", "end_ms", "boxes"}:
            raise ValueError("frame requires only start_ms, end_ms, boxes")
        start, end, boxes = frame["start_ms"], frame["end_ms"], frame["boxes"]
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise ValueError("invalid frame interval")
        if previous_end is not None and start < previous_end:
            raise ValueError("overlapping frame intervals")
        if not isinstance(boxes, list):
            raise ValueError("boxes must be a list")
        validated = []
        for box in boxes:
            if not isinstance(box, (tuple, list)) or len(box) != 4 or any(
                type(v) not in (int, float) or not 0 <= v <= 1 for v in box
            ) or not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError("invalid normalized box")
            validated.append(tuple(float(v) for v in box))
        if start != previous_end:
            previous = []
        similarities = [[_iou(old_box, box) for _, old_box in previous] for box in validated]
        choices = []
        for scores in similarities:
            matches = [i for i, value in enumerate(scores) if value >= min_iou]
            choices.append(matches[0] if len(matches) == 1 else None)
        claimed = [choice for choice in choices if choice is not None]
        duplicated = {choice for choice in claimed if claimed.count(choice) > 1}
        tracks = []
        current = []
        for box, scores, choice in zip(validated, similarities, choices):
            uncertain = choice is None and any(score >= min_iou for score in scores)
            if choice in duplicated:
                uncertain = True
            if choice is not None and choice not in duplicated:
                track_id = previous[choice][0]
                state = "stable"
            else:
                track_id = f"synthetic_track_{next_id}"
                next_id += 1
                state = "ambiguous" if uncertain else "stable"
            tracks.append({"track_id": track_id, "state": state})
            current.append((track_id, box))
        output.append({"start_ms": start, "end_ms": end, "tracks": tracks})
        previous = current
        previous_end = end
    return output
