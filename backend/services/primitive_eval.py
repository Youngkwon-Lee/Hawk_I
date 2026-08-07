"""Score a model's structured observations against human labels.

Most explainable-AI work generates explanations but never grades them. Because
the labeling tool records the same primitives the model is asked to emit, the
explanation itself can be scored - and a score that is right for the wrong
reason can be separated from one that is right for the right reason.

Three outputs, matching the evaluation design:
1. per-primitive agreement (exact, within-1, rank correlation)
2. a cross table of score-correct x observation-correct
3. trivial baselines, because a primitive that is almost always 0 makes
   "always predict 0" look excellent
"""

from __future__ import annotations

from typing import Any, Iterable

# The gait labeling UI already collects all nine ontology primitives under
# legacy field names. This is the mapping recorded in the ontology document.
GAIT_LABEL_TO_PRIMITIVE = {
    "gait_speed": "gait_speed_reduction",
    "stride_length": "shortened_stride",
    "left_right_asymmetry": "step_length_asymmetry",
    "arm_swing_asymmetry": "arm_swing_asymmetry",
    "festination": "festination",
    "freezing": "freezing_of_gait",
    "turning_quality": "turning_impairment",
    "stooped_posture": "trunk_flexion",
    "postural_stability": "postural_instability",
}


# The gait UI stores ordinal severity as text, and two vocabularies are in use:
# Korean arrows for magnitude fields and English words for the rest. Freezing is
# recorded as presence rather than severity.
SEVERITY_WORDS = {
    "정상": 0.0, "경미↓": 1.0, "중등↓": 2.0, "심함↓": 3.0,
    "none": 0.0, "mild": 1.0, "moderate": 2.0, "severe": 3.0,
    "normal": 0.0,
    "observed": 1.0,
}


def parse_severity(value: Any) -> float | None:
    """Read a severity that may be a number or one of the ordinal words.

    Returns None for blanks, which the ontology treats as "not observed" rather
    than as normal - so a blank must never become a zero.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text in SEVERITY_WORDS:
        return SEVERITY_WORDS[text]
    lowered = text.lower()
    if lowered in SEVERITY_WORDS:
        return SEVERITY_WORDS[lowered]
    try:
        return float(text)
    except ValueError:
        return None


def _as_number(value: Any) -> float | None:
    return parse_severity(value)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties do not distort the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def spearman(left: list[float], right: list[float]) -> float | None:
    """Rank correlation, or None when it is undefined (too few points, no variance)."""
    if len(left) != len(right) or len(left) < 3:
        return None
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    n = len(left_ranks)
    mean_left = sum(left_ranks) / n
    mean_right = sum(right_ranks) / n
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left_ranks, right_ranks))
    left_var = sum((a - mean_left) ** 2 for a in left_ranks)
    right_var = sum((b - mean_right) ** 2 for b in right_ranks)
    if left_var == 0 or right_var == 0:
        return None
    return round(numerator / (left_var * right_var) ** 0.5, 4)


def extract_label_primitives(label: dict[str, Any]) -> dict[str, float]:
    """Read the nine gait primitives out of a label record's legacy field names."""
    found: dict[str, float] = {}
    for field, primitive in GAIT_LABEL_TO_PRIMITIVE.items():
        value = _as_number(label.get(field))
        if value is not None:
            found[primitive] = value
    return found


def extract_predicted_primitives(prediction: dict[str, Any]) -> dict[str, float]:
    """Read severities from an ontology prediction, skipping anything not observed."""
    primitives = prediction.get("primitives")
    if not isinstance(primitives, dict):
        return {}
    found: dict[str, float] = {}
    for name, rating in primitives.items():
        if not isinstance(rating, dict):
            continue
        if rating.get("observability") != "observed":
            continue
        value = _as_number(rating.get("severity"))
        if value is not None:
            found[name] = value
    return found


# Turn and freezing spans are recorded under gait_events, not as ontology
# evidence spans, but they carry the same information: when the finding happens.
INTERVAL_SOURCES = {
    "turning_intervals": "turning_impairment",
    "freezing_intervals": "freezing_of_gait",
}


def extract_intervals(record: dict[str, Any], kind: str = "turning_intervals") -> list[tuple[float, float]]:
    """Read (start, end) spans, dropping entries that are not usable numbers."""
    events = record.get("gait_events")
    if not isinstance(events, dict):
        events = record.get("label") if isinstance(record.get("label"), dict) else {}
        events = events.get("gait_events") if isinstance(events.get("gait_events"), dict) else {}
    spans = events.get(kind) if isinstance(events, dict) else None
    if not isinstance(spans, list):
        return []

    parsed: list[tuple[float, float]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        start, end = _as_number(span.get("start")), _as_number(span.get("end"))
        if start is None or end is None or end < start:
            continue
        parsed.append((start, end))
    return sorted(parsed)


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union if union > 0 else 0.0


def temporal_event_agreement(
    pairs: Iterable[tuple[list[tuple[float, float]], list[tuple[float, float]]]],
    tolerance_sec: float = 1.0,
) -> dict[str, Any]:
    """Match predicted spans to labeled spans by midpoint, and report IoU too.

    Most labeled turn spans are well under a second, so they behave like event
    markers rather than durations. Overlap-based matching would then fail on
    near-misses that are clinically the same event, which is why midpoint
    distance within a tolerance is the primary match and IoU is reported
    alongside rather than used as the criterion.
    """
    matched = 0
    total_true = 0
    total_pred = 0
    ious: list[float] = []

    for truths, predictions in pairs:
        total_true += len(truths)
        total_pred += len(predictions)
        available = list(predictions)
        for truth in truths:
            truth_mid = (truth[0] + truth[1]) / 2
            best_index, best_distance = None, None
            for index, prediction in enumerate(available):
                distance = abs((prediction[0] + prediction[1]) / 2 - truth_mid)
                if distance <= tolerance_sec and (best_distance is None or distance < best_distance):
                    best_index, best_distance = index, distance
            if best_index is not None:
                ious.append(_iou(truth, available.pop(best_index)))
                matched += 1

    precision = matched / total_pred if total_pred else None
    recall = matched / total_true if total_true else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )
    return {
        "labeled_events": total_true,
        "predicted_events": total_pred,
        "matched": matched,
        "tolerance_sec": tolerance_sec,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "mean_iou_of_matched": round(sum(ious) / len(ious), 4) if ious else None,
    }


def per_primitive_agreement(pairs: Iterable[tuple[dict[str, float], dict[str, float]]]) -> dict[str, dict[str, Any]]:
    """Agreement for each primitive separately.

    Averaging across primitives would hide that walking speed is easy and
    freezing is not, which is exactly the difference a clinician cares about.
    """
    collected: dict[str, list[tuple[float, float]]] = {}
    for label, prediction in pairs:
        for primitive, truth in label.items():
            if primitive in prediction:
                collected.setdefault(primitive, []).append((truth, prediction[primitive]))

    report: dict[str, dict[str, Any]] = {}
    for primitive, values in sorted(collected.items()):
        truths = [t for t, _ in values]
        predictions = [p for _, p in values]
        n = len(values)
        exact = sum(1 for t, p in values if t == p)
        within_one = sum(1 for t, p in values if abs(t - p) <= 1)
        report[primitive] = {
            "n": n,
            "exact_match": round(exact / n, 4),
            "within_1": round(within_one / n, 4),
            "spearman": spearman(truths, predictions),
        }
    return report


def label_distribution(pairs: Iterable[tuple[dict[str, float], dict[str, float]]]) -> dict[str, dict[str, Any]]:
    """Share of zeros per primitive, and how a trivial predictor would score.

    A primitive that is 95% zero makes "always predict 0" look like 95%
    accuracy, so the trivial baselines are reported next to the distribution
    rather than left for the reader to infer.
    """
    collected: dict[str, list[float]] = {}
    for label, _ in pairs:
        for primitive, truth in label.items():
            collected.setdefault(primitive, []).append(truth)

    report: dict[str, dict[str, Any]] = {}
    for primitive, truths in sorted(collected.items()):
        n = len(truths)
        zeros = sum(1 for t in truths if t == 0)
        counts: dict[float, int] = {}
        for t in truths:
            counts[t] = counts.get(t, 0) + 1
        majority_value, majority_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
        report[primitive] = {
            "n": n,
            "zero_share": round(zeros / n, 4),
            "always_zero_accuracy": round(zeros / n, 4),
            "majority_value": majority_value,
            "majority_accuracy": round(majority_count / n, 4),
        }
    return report


def top_finding(primitives: dict[str, float]) -> str | None:
    """The most severe finding; ties break by name so the result is deterministic."""
    positive = {name: value for name, value in primitives.items() if value > 0}
    if not positive:
        return None
    return min(positive.items(), key=lambda item: (-item[1], item[0]))[0]


def score_vs_observation(
    records: Iterable[dict[str, Any]],
    score_tolerance: float = 0,
) -> dict[str, Any]:
    """Cross-tabulate score correctness against observation correctness.

    The interesting cell is right-score/wrong-observation: a model that lands on
    the right number for the wrong reason. Score-only evaluation cannot see it.
    """
    table = {
        "score_right_observation_right": 0,
        "score_right_observation_wrong": 0,
        "score_wrong_observation_right": 0,
        "score_wrong_observation_wrong": 0,
        "skipped_missing_data": 0,
    }

    for record in records:
        true_score = _as_number(record.get("true_score"))
        predicted_score = _as_number(record.get("predicted_score"))
        label_primitives = record.get("label_primitives") or {}
        predicted_primitives = record.get("predicted_primitives") or {}

        if true_score is None or predicted_score is None or not label_primitives or not predicted_primitives:
            table["skipped_missing_data"] += 1
            continue

        score_right = abs(true_score - predicted_score) <= score_tolerance
        observation_right = top_finding(label_primitives) == top_finding(predicted_primitives)

        key = (
            f"score_{'right' if score_right else 'wrong'}"
            f"_observation_{'right' if observation_right else 'wrong'}"
        )
        table[key] += 1

    scored = sum(
        table[name] for name in table if name != "skipped_missing_data"
    )
    right_for_wrong_reason = table["score_right_observation_wrong"]
    table["evaluated"] = scored
    table["lucky_hit_share"] = round(right_for_wrong_reason / scored, 4) if scored else None
    return table
