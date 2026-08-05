#!/usr/bin/env python3
"""Grade model predictions against human labels and print the three result tables.

    python backend/scripts/evaluate_primitive_predictions.py \\
        --labels labels_gait.jsonl --predictions preds_c3.jsonl

Inputs
------
--labels       Records with `clip_id` and either a flat set of gait fields or a
               nested `label` object (the labeling tool's export shape).
--predictions  The v2 prediction contract: `clip_id` plus `primitives`, and a
               score from `predicted_score` or `score_anchor`.

Outputs
-------
1. Label distribution and trivial baselines - read this first. A primitive that
   is 95% zero makes "always predict 0" score 95%, so any accuracy below the
   baseline column means the model learned nothing for that primitive.
2. Per-primitive agreement - exact, within-1, rank correlation.
3. Score x observation cross table - separates a right answer from a right
   answer for the right reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.primitive_eval import (  # noqa: E402
    INTERVAL_SOURCES,
    extract_intervals,
    extract_label_primitives,
    extract_predicted_primitives,
    label_distribution,
    per_primitive_agreement,
    score_vs_observation,
    temporal_event_agreement,
)
from services.prediction_ingest import score_from  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            print(f"  skipped {path.name} line {index}: {exc.msg}", file=sys.stderr)
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def label_fields(record: dict) -> dict:
    """Labels may be flat or nested under `label`, depending on the export."""
    nested = record.get("label")
    return nested if isinstance(nested, dict) else record


def print_table(title: str, rows: list[list[str]], headers: list[str]) -> None:
    print(f"\n{title}")
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    print("  " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("  " + "-+-".join("-" * w for w in widths))
    for row in rows:
        print("  " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--score-tolerance", type=float, default=0,
        help="0 grades the score as exact (default); 1 counts within-one as correct",
    )
    parser.add_argument(
        "--event-tolerance", type=float, default=1.0,
        help="seconds of midpoint distance allowed when matching predicted spans (default 1.0)",
    )
    parser.add_argument("--json-out", type=Path, help="also write the report as JSON")
    args = parser.parse_args(argv)

    for path in (args.labels, args.predictions):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    labels = {str(r.get("clip_id")): r for r in load_jsonl(args.labels) if r.get("clip_id")}
    predictions = {str(r.get("clip_id")): r for r in load_jsonl(args.predictions) if r.get("clip_id")}

    shared = sorted(set(labels) & set(predictions))
    print(f"labels {len(labels)} | predictions {len(predictions)} | matched clips {len(shared)}")
    if not shared:
        print("error: no clip_id appears in both files", file=sys.stderr)
        return 1
    if len(shared) < len(predictions):
        print(f"  note: {len(predictions) - len(shared)} prediction(s) had no matching label")

    pairs = []
    cross_records = []
    for clip_id in shared:
        label_primitives = extract_label_primitives(label_fields(labels[clip_id]))
        predicted_primitives = extract_predicted_primitives(predictions[clip_id])
        pairs.append((label_primitives, predicted_primitives))
        cross_records.append({
            "true_score": label_fields(labels[clip_id]).get("updrs_score"),
            "predicted_score": score_from(predictions[clip_id]),
            "label_primitives": label_primitives,
            "predicted_primitives": predicted_primitives,
        })

    distribution = label_distribution(pairs)
    agreement = per_primitive_agreement(pairs)
    cross = score_vs_observation(cross_records, score_tolerance=args.score_tolerance)

    print_table(
        "1) 라벨 분포와 기준선 — 이걸 먼저 보세요",
        [
            [name, d["n"], f"{d['zero_share']:.1%}", f"{d['always_zero_accuracy']:.1%}", f"{d['majority_accuracy']:.1%}"]
            for name, d in distribution.items()
        ],
        ["primitive", "n", "0의 비율", "'항상 0' 정확도", "최빈값 정확도"],
    )
    print("  → 아래 표의 정확도가 이 기준선을 못 넘으면 그 항목은 학습되지 않은 것입니다.")

    if agreement:
        print_table(
            "2) 항목별 일치도",
            [
                [name, a["n"], f"{a['exact_match']:.1%}", f"{a['within_1']:.1%}",
                 "-" if a["spearman"] is None else f"{a['spearman']:.3f}"]
                for name, a in agreement.items()
            ],
            ["primitive", "n", "정확 일치", "1단계 이내", "순위상관"],
        )
    else:
        print("\n2) 항목별 일치도 — 예측에 primitive가 없어 계산할 수 없습니다 (C0 조건이면 정상입니다)")

    tolerance_note = "정확 일치" if args.score_tolerance == 0 else f"±{args.score_tolerance:g} 허용"
    print_table(
        f"3) 점수 × 관찰 교차표 (점수 기준: {tolerance_note})",
        [
            ["점수 맞음", cross["score_right_observation_right"], cross["score_right_observation_wrong"]],
            ["점수 틀림", cross["score_wrong_observation_right"], cross["score_wrong_observation_wrong"]],
        ],
        ["", "관찰 맞음", "관찰 틀림"],
    )
    if cross["lucky_hit_share"] is not None:
        print(f"  운 좋게 맞춤(점수는 맞고 이유는 틀림): {cross['lucky_hit_share']:.1%} of {cross['evaluated']}건")
    if cross["skipped_missing_data"]:
        print(f"  데이터 부족으로 제외: {cross['skipped_missing_data']}건")

    temporal: dict[str, Any] = {}
    for source, primitive in INTERVAL_SOURCES.items():
        span_pairs = [
            (extract_intervals(labels[c], source), extract_intervals(predictions[c], source))
            for c in shared
        ]
        if not any(truth for truth, _ in span_pairs):
            continue
        temporal[primitive] = temporal_event_agreement(span_pairs, tolerance_sec=args.event_tolerance)

    if temporal:
        print_table(
            f"4) 시간 구간 예측 (매칭 기준: 중점 ±{args.event_tolerance:g}초)",
            [
                [primitive, r["labeled_events"], r["predicted_events"], r["matched"],
                 "-" if r["precision"] is None else f"{r['precision']:.1%}",
                 "-" if r["recall"] is None else f"{r['recall']:.1%}",
                 "-" if r["f1"] is None else f"{r['f1']:.3f}",
                 "-" if r["mean_iou_of_matched"] is None else f"{r['mean_iou_of_matched']:.3f}"]
                for primitive, r in temporal.items()
            ],
            ["primitive", "정답구간", "예측구간", "매칭", "정밀도", "재현율", "F1", "평균 IoU"],
        )
        print("  → 라벨 구간 대부분이 1초 미만이라 지속시간이 아닌 사건 표시에 가깝습니다.")
        print("     그래서 중점 거리로 매칭하고 IoU는 참고로만 표시합니다.")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {"matched_clips": len(shared), "distribution": distribution,
                 "agreement": agreement, "score_vs_observation": cross,
                 "temporal_events": temporal},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nJSON 저장: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
