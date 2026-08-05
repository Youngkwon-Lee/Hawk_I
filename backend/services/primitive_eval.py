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


def _as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


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
