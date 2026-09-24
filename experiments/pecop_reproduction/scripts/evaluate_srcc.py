#!/usr/bin/env python3
"""Evaluate PD4T predictions with Spearman rank correlation (SRCC).

The implementation is dependency-free and uses average ranks for ties.
Input CSV must contain ground-truth and prediction columns.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        raise ValueError("at least two samples are required")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    numerator = sum(a * b for a, b in zip(dx, dy))
    denominator = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if denominator == 0:
        raise ValueError("correlation is undefined for constant input")
    return numerator / denominator


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def read_columns(path: Path, truth_col: str, pred_col: str) -> tuple[list[float], list[float]]:
    truth: list[float] = []
    pred: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV header is required")
        missing = [name for name in (truth_col, pred_col) if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing CSV columns: {missing}")
        for line_no, row in enumerate(reader, start=2):
            try:
                truth.append(float(row[truth_col]))
                pred.append(float(row[pred_col]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid numeric value") from exc
    return truth, pred


def run_self_test() -> None:
    assert abs(spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12
    tied = spearman([0, 0, 1, 2], [0, 1, 1, 2])
    assert 0.8 < tied < 1.0
    print("self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="?")
    parser.add_argument("--truth-col", default="y_true")
    parser.add_argument("--pred-col", default="y_pred")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if args.predictions is None:
        parser.error("predictions CSV is required unless --self-test is used")

    truth, pred = read_columns(args.predictions, args.truth_col, args.pred_col)
    value = spearman(truth, pred)
    print(f"n={len(truth)}")
    print(f"SRCC={value:.6f}")
    print(f"SRCC_percent={value * 100:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
