"""Pure scoring helpers for the five-grade clinician C0B candidate."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np


ANSWER_RE = re.compile(r"answer:\s*([0-4])", re.IGNORECASE)


def parse_c0b_answer(text: str) -> int | None:
    matches = ANSWER_RE.findall(text or "")
    return int(matches[-1]) if matches else None


def quadratic_weighted_kappa(
    truth: Sequence[int], predictions: Sequence[int], *, n_classes: int = 5
) -> float:
    y = np.asarray(truth, dtype=int)
    p = np.asarray(predictions, dtype=int)
    if len(y) != len(p) or not len(y):
        raise ValueError("truth and predictions must be non-empty and the same length")
    observed = np.zeros((n_classes, n_classes), dtype=float)
    for actual, predicted in zip(y, p):
        observed[actual, predicted] += 1
    weights = np.fromfunction(
        lambda i, j: ((i - j) ** 2) / ((n_classes - 1) ** 2),
        (n_classes, n_classes),
        dtype=float,
    )
    expected = np.outer(
        np.bincount(y, minlength=n_classes), np.bincount(p, minlength=n_classes)
    ) / len(y)
    denominator = float((weights * expected).sum())
    return float("nan") if denominator == 0 else 1.0 - float((weights * observed).sum()) / denominator


def score_predictions(
    truth: Sequence[int],
    predictions: Sequence[int | None],
    *,
    fallback: int = 0,
) -> dict[str, Any]:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("truth and predictions must be non-empty and the same length")
    resolved = [fallback if value is None else int(value) for value in predictions]
    y = np.asarray(truth, dtype=int)
    p = np.asarray(resolved, dtype=int)
    confusion = np.zeros((5, 5), dtype=int)
    for actual, predicted in zip(y, p):
        confusion[actual, predicted] += 1
    recalls = [
        float(confusion[grade, grade] / confusion[grade].sum())
        for grade in range(5)
        if confusion[grade].sum()
    ]
    gt2 = y >= 2
    majority = int(np.bincount(y, minlength=5).argmax())
    return {
        "n": len(y),
        "parse_failures": sum(value is None for value in predictions),
        "fallback": fallback,
        "exact": float((y == p).mean()),
        "within_1": float((np.abs(y - p) <= 1).mean()),
        "mae": float(np.abs(y - p).mean()),
        "qwk": quadratic_weighted_kappa(y.tolist(), p.tolist()),
        "balanced_accuracy": float(np.mean(recalls)),
        "gt2_sensitivity": float((p[gt2] >= 2).mean()) if gt2.any() else None,
        "majority_class": majority,
        "majority_exact_baseline": float((y == majority).mean()),
        "confusion_matrix": confusion.tolist(),
        "truth_counts": {str(i): int((y == i).sum()) for i in range(5)},
        "prediction_counts": {str(i): int((p == i).sum()) for i in range(5)},
    }
