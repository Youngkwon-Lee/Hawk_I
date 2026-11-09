#!/usr/bin/env python3
"""Grade model predictions against human labels and print the three result tables.

    python backend/scripts/evaluate_primitive_predictions.py \\
        --labels labels_gait.jsonl --predictions preds_c3.jsonl \
        --task gait --dataset PD4T

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
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.primitive_eval import (  # noqa: E402
    INTERVAL_SOURCES,
    extract_label_primitives,
    extract_predicted_primitives,
    label_distribution,
    per_primitive_agreement,
    score_vs_observation,
    temporal_event_agreement,
)
from services.prediction_ingest import score_from, validate_prediction  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{index}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path.name}:{index}: expected a JSON object")
        records.append(record)
    return records


def label_fields(record: dict) -> dict:
    """Labels may be flat or nested under `label`, depending on the export."""
    nested = record.get("label")
    return nested if isinstance(nested, dict) else record


def _field_for(record: dict, role: str, name: str) -> Any:
    """Resolve nested export fields without trusting conflicting values."""
    outer = record.get(name)
    if role != "label":
        return outer
    nested = label_fields(record)
    if nested is record:
        return outer
    inner = nested.get(name)
    if outer is not None and inner is not None and outer != inner:
        raise ValueError(f"label: conflicting nested/top-level {name}")
    return inner if inner is not None else outer


def label_score(record: dict, task: str) -> Any:
    value = _field_for(record, "label", "updrs_score")
    if value not in (None, ""):
        return value
    anchor = _field_for(record, "label", "score_anchor") or {}
    if not isinstance(anchor, dict):
        raise ValueError("label: score_anchor must be an object")
    return anchor.get("updrs_3_4" if task == "finger_tapping" else "updrs_3_10")


def validated_score(record: dict, *, role: str, task: str) -> float | None:
    """Validate score ranges and cross-field agreement without inventing missing scores."""
    expected = "updrs_3_4" if task == "finger_tapping" else "updrs_3_10"
    other = "updrs_3_10" if task == "finger_tapping" else "updrs_3_4"
    field = "updrs_score" if role == "label" else "predicted_score"
    flat = _field_for(record, role, field)
    anchor = _field_for(record, role, "score_anchor")
    if anchor is not None and not isinstance(anchor, dict):
        raise ValueError(f"{role}: score_anchor must be an object")
    anchor = anchor or {}
    if anchor.get(other) not in (None, ""):
        raise ValueError(f"{role}: score_anchor has the wrong task-specific UPDRS item")

    supplied = [(field, flat), (f"score_anchor.{expected}", anchor.get(expected))]
    if role == "prediction":
        supplied.append(("score_anchor.score", anchor.get("score")))
    numbers = []
    for source, raw in supplied:
        if raw is None or raw == "":
            continue
        if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
            raise ValueError(f"{role}: {source} is not a numeric score")
        try:
            score = float(raw)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{role}: {source} is not a numeric score") from exc
        if not math.isfinite(score) or not 0 <= score <= 4 or not score.is_integer():
            raise ValueError(f"{role}: {source} must be an integer score in 0-4")
        numbers.append(score)
    if any(value != numbers[0] for value in numbers[1:]):
        raise ValueError(f"{role}: conflicting ordinal score fields")
    return numbers[0] if numbers else None


def label_primitives(record: dict, task: str) -> dict[str, float]:
    fields = label_fields(record)
    structured = _field_for(record, "label", "primitives")
    if isinstance(structured, dict):
        if all(isinstance(p, dict) for p in structured.values()):
            return extract_predicted_primitives({"primitives": structured})
        # Existing gait training exports contain plain severity values.
        if all(value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 3)
               for value in structured.values()):
            return {key: float(value) for key, value in structured.items() if value is not None}
        raise ValueError("mixed or invalid label primitive representation")
    if task != "gait":
        raise ValueError("finger tapping requires versioned structured primitives")
    return extract_label_primitives(fields)


def _score_held(record: dict, role: str, task: str) -> bool:
    gate = _field_for(record, role, "quality_gate")
    if gate is None:
        return False  # Legacy labels have no certified quality gate.
    if not isinstance(gate, dict) or gate.get("status") not in ("pass", "hold"):
        raise ValueError(f"{role}: invalid quality_gate")
    reasons = gate.get("reasons")
    if not isinstance(reasons, list) or (gate["status"] == "hold" and not reasons) or (
        gate["status"] == "pass" and reasons
    ):
        raise ValueError(f"{role}: quality_gate reasons do not match status")
    if gate["status"] != "hold":
        return False
    score = label_score(record, task) if role == "label" else score_from(record)
    if score is not None:
        raise ValueError(f"{role}: held clip must not contain a score")
    if any((isinstance(p, dict) and p.get("severity") is not None)
           or (type(p) in (int, float) and math.isfinite(p))
           for p in (_field_for(record, role, "primitives") or {}).values()):
        raise ValueError(f"{role}: held clip must not contain primitive severities")
    return True


def _check_primitive_evidence(record: dict, role: str) -> None:
    primitives = _field_for(record, role, "primitives")
    if primitives is None:
        return
    if not isinstance(primitives, dict):
        raise ValueError(f"{role}: primitives must be a mapping")
    for name, rating in primitives.items():
        if rating is None or (type(rating) in (int, float) and math.isfinite(rating) and 0 <= rating <= 3):
            if role != "label":
                raise ValueError(f"{role}: primitive {name} must be structured")
            continue  # Historical numeric training labels, never model output.
        if not isinstance(rating, dict):
            raise ValueError(f"{role}: invalid primitive {name}")
        observed = rating.get("observability")
        severity = rating.get("severity")
        if observed not in ("observed", "unobservable", "uncertain"):
            raise ValueError(f"{role}: invalid observability for {name}")
        if observed == "observed":
            if type(severity) is not int or severity not in (0, 1, 2, 3):
                raise ValueError(f"{role}: invalid observed severity for {name}")
        elif severity is not None:
            raise ValueError(f"{role}: unobserved {name} requires null severity")
        evidence = rating.get("evidence", [])
        if not isinstance(evidence, list) or (severity is not None and severity > 0 and not evidence):
            raise ValueError(f"{role}: positive {name} requires evidence")
        for span in evidence:
            if not isinstance(span, dict):
                raise ValueError(f"{role}: invalid evidence for {name}")
            seconds = (("start_s", "end_s"), ("start_sec", "end_sec"))
            has_time = [any(k in span for k in pair) for pair in seconds]
            has_cycle = any(k in span for k in ("cycle_start", "cycle_end"))
            if sum(has_time) + int(has_cycle) != 1:
                raise ValueError(f"{role}: ambiguous time/cycle evidence for {name}")
            if has_cycle:
                start, end = span.get("cycle_start"), span.get("cycle_end")
                if type(start) is not int or type(end) is not int or start < 1 or end < start:
                    raise ValueError(f"{role}: invalid cycle interval for {name}")
            else:
                start_key, end_key = seconds[has_time.index(True)]
                start, end = span.get(start_key), span.get(end_key)
                if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end <= start):
                    raise ValueError(f"{role}: invalid seconds interval for {name}")



def _legacy_event_intervals(record: dict, source: str) -> tuple[list[tuple[float, float]], bool]:
    """Validate raw legacy event intervals and retain explicit empty annotations.

    Missing/null lists are unannotated, not proof of an event-free recording.
    """
    nested = label_fields(record)
    export_events = record.get("events")
    if export_events is not None and not isinstance(export_events, dict):
        raise ValueError("invalid legacy events container")
    candidates = [
        ("gait_events", record.get("gait_events")),
        ("label.gait_events", nested.get("gait_events") if nested is not record else None),
        ("events.gait_events", export_events.get("gait_events") if export_events else None),
    ]
    declared = None
    for location, events in candidates:
        if events is None:
            continue
        if not isinstance(events, dict):
            raise ValueError(f"invalid legacy {location} container")
        if source not in events or events[source] is None:
            continue
        spans = events[source]
        if not isinstance(spans, list):
            raise ValueError(f"invalid legacy {location}.{source} intervals")
        checked = []
        for span in spans:
            if not isinstance(span, dict):
                raise ValueError(f"invalid legacy {source} span")
            start, end = span.get("start"), span.get("end")
            if any(isinstance(v, bool) or not isinstance(v, (str, int, float))
                   for v in (start, end)):
                raise ValueError(f"invalid legacy {source} interval values")
            try:
                begin, finish = float(start), float(end)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"invalid legacy {source} interval values") from exc
            if not math.isfinite(begin) or not math.isfinite(finish) or begin < 0 or finish <= begin:
                raise ValueError(f"invalid legacy {source} interval bounds")
            checked.append((begin, finish))
        checked.sort()
        if declared is not None and declared != checked:
            raise ValueError(f"conflicting legacy {source} event annotations")
        declared = checked
    return (declared if declared is not None else [], declared is not None)


def _quality_state(record: dict, role: str, task: str) -> str:
    """Separate certified pass, explicit hold, and missing quality evidence."""
    if _score_held(record, role, task):
        return "hold"
    return "pass" if _field_for(record, role, "quality_gate") is not None else "unverified"


def _temporal_label_annotated(record: dict, source: str, primitive: str) -> bool:
    """Count predicted-only events only against explicitly annotated references."""
    _, has_legacy_list = _legacy_event_intervals(record, source)
    if has_legacy_list:
        return True
    primitives = _field_for(record, "label", "primitives")
    rating = (primitives or {}).get(primitive) if isinstance(primitives, dict) else None
    return (isinstance(rating, dict) and rating.get("observability") == "observed"
            and isinstance(rating.get("evidence"), list))


def index_by_identity(records: list[dict], *, role: str, task: str, dataset: str) -> dict[tuple[str, str, str, str], dict]:
    """Never silently overwrite a different side, dataset, or model condition."""
    indexed = {}
    signatures = set()
    for record in records:
        clip_id = record.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError(f"{role}: missing clip_id")
        declared_task = record.get("task") or (record.get("item") if role == "label" else None)
        if declared_task is not None and declared_task != task:
            raise ValueError(f"{role}: wrong task for {clip_id}")
        if role == "prediction" and declared_task is None:
            raise ValueError(f"{role}: missing task for {clip_id}")
        declared_dataset = record.get("dataset")
        if declared_dataset is not None and declared_dataset != dataset:
            raise ValueError(f"{role}: dataset mismatch for {clip_id}")
        if role == "prediction" and declared_dataset is None:
            raise ValueError(f"{role}: dataset is required for {clip_id}")
        side = record.get("side")
        if task == "finger_tapping" and side not in ("left", "right"):
            raise ValueError(f"{role}: finger tapping side required for {clip_id}")
        if task == "gait" and side not in (None, "na"):
            raise ValueError(f"{role}: unexpected gait side for {clip_id}")
        key = (dataset, task, clip_id, side if task == "finger_tapping" else "na")
        if key in indexed:
            raise ValueError(f"{role}: duplicate evaluation key {key}")
        _check_primitive_evidence(record, role)
        declared_split(record, role)
        if task == "gait":
            for event_source in INTERVAL_SOURCES:
                _legacy_event_intervals(record, event_source)
        validated_score(record, role=role, task=task)
        _score_held(record, role, task)
        if role == "prediction":
            validate_prediction(record)
            signatures.add((record.get("model"), record.get("condition"),
                            record.get("model_version"), record.get("model_sha256")))
        else:
            review_status = _field_for(record, role, "review_status")
            if review_status is not None and review_status not in ("reviewed", "adjudicated"):
                raise ValueError(f"{role}: incomplete reviewer decision for {clip_id}")
            if task == "finger_tapping" and not isinstance(_field_for(record, role, "primitives"), dict):
                raise ValueError(f"{role}: missing finger tapping primitives for {clip_id}")
        indexed[key] = record
    if role == "prediction" and len(signatures) > 1:
        raise ValueError("mixed model/condition predictions: evaluate each candidate separately")
    return indexed


def declared_split(record: dict, role: str) -> str | None:
    """Extract declared split from flat/nested label and legacy export metadata.

    Compare claims only; a matching split declaration does not certify
    training-run membership, a frozen subject split, or held-out status.
    """
    declared = [("split", _field_for(record, role, "split"))]
    if role == "label":
        meta = record.get("meta")
        if meta is not None and not isinstance(meta, dict):
            raise ValueError("label: meta must be an object")
        if isinstance(meta, dict):
            declared.append(("meta.split", meta.get("split")))
    split = None
    for name, raw in declared:
        if raw is None or raw == "":
            continue
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{role}: invalid {name} identity")
        value = raw.strip().lower()
        if split is not None and value != split:
            raise ValueError(f"{role}: conflicting declared split identity")
        split = value
    return split


def pair_provenance_unverified(label: dict, prediction: dict) -> bool:
    """Check declared metadata, but never certify an unverified source binding.

    These fields are caller-supplied. Matching clip IDs, ontology versions and
    source digests are not evidence that both files came from the same approved
    media artifact. Independent source receipts and model-lineage verification
    are outside this evaluator and have not been supplied here.
    """
    left_dataset, right_dataset = label.get("dataset"), prediction.get("dataset")
    if left_dataset and right_dataset and left_dataset != right_dataset:
        raise ValueError("model/label provenance mismatch: dataset")
    left_split, right_split = declared_split(label, "label"), declared_split(prediction, "prediction")
    if left_split is not None and right_split is not None and left_split != right_split:
        raise ValueError("model/label provenance mismatch: split")
    for key in ("ontology_version", "source_sha256"):
        left, right = _field_for(label, "label", key), prediction.get(key)
        if left and right and left != right:
            raise ValueError(f"model/label provenance mismatch: {key}")
    return True  # Never promote self-declared matching values into verified provenance.


def temporal_evidence(record: dict, source: str, primitive: str) -> tuple[list[tuple[float, float]], int]:
    """Return seconds only. Cycle-number spans are kept separate, never treated as seconds."""
    legacy, _ = _legacy_event_intervals(record, source)
    if legacy:
        return legacy, 0
    primitives = _field_for(record, "label", "primitives")
    rating = (primitives or {}).get(primitive)
    if not isinstance(rating, dict):
        return [], 0
    intervals, cycle_only = [], 0
    for span in rating.get("evidence") or []:
        if "cycle_start" in span:
            cycle_only += 1
        elif "start_s" in span:
            intervals.append((float(span["start_s"]), float(span["end_s"])))
        elif "start_sec" in span:
            intervals.append((float(span["start_sec"]), float(span["end_sec"])))
    return intervals, cycle_only


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
    parser.add_argument("--task", choices=("gait", "finger_tapping"), required=True)
    parser.add_argument("--dataset", required=True, help="operator-confirmed common dataset identity")
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

    if not args.dataset.strip():
        print("error: dataset cannot be empty", file=sys.stderr)
        return 2
    if (not math.isfinite(args.score_tolerance) or args.score_tolerance < 0
            or not math.isfinite(args.event_tolerance) or args.event_tolerance < 0):
        print("error: evaluation tolerances must be finite and nonnegative", file=sys.stderr)
        return 2
    try:
        labels = index_by_identity(load_jsonl(args.labels), role="label", task=args.task, dataset=args.dataset)
        predictions = index_by_identity(load_jsonl(args.predictions), role="prediction", task=args.task, dataset=args.dataset)
        shared = sorted(set(labels) & set(predictions))
        if not shared:
            raise ValueError("no identical dataset/task/clip/side keys in both files")
        incomplete_provenance = sum(pair_provenance_unverified(labels[key], predictions[key]) for key in shared)
        split_claims = [
            (declared_split(labels[key], "label"), declared_split(predictions[key], "prediction"))
            for key in shared
        ]
        declared_splits = {value for pair in split_claims for value in pair if value is not None}
        if len(declared_splits) > 1:
            raise ValueError("mixed split claims: evaluate each split separately")
        split_claim_matching_pairs = sum(left is not None and right is not None
                                         for left, right in split_claims)
        split_claim_missing_pairs = len(shared) - split_claim_matching_pairs
    except (ValueError, OSError) as exc:
        print(f"error: evaluation input rejected: {exc}", file=sys.stderr)
        return 2

    print(f"labels {len(labels)} | predictions {len(predictions)} | matched identity keys {len(shared)}")
    print(f"unmatched labels {len(labels)-len(shared)} | unmatched predictions {len(predictions)-len(shared)}")
    print(f"provenance not fully bound for {incomplete_provenance}/{len(shared)} pair(s); results are research-only")
    print(f"split claims matching {split_claim_matching_pairs} | incomplete {split_claim_missing_pairs}; "
          "held-out split provenance remains unverified")
    comparable, held_count, uncertified_count = [], 0, 0
    for key in shared:
        label_state = _quality_state(labels[key], "label", args.task)
        prediction_state = _quality_state(predictions[key], "prediction", args.task)
        if "hold" in (label_state, prediction_state):
            held_count += 1
        elif label_state == prediction_state == "pass":
            comparable.append(key)
        else:
            uncertified_count += 1
    print(f"certified quality-pass pairs {len(comparable)} | held {held_count} | "
          f"missing quality gate {uncertified_count}")
    if not comparable:
        print("error: no certified quality-passing matched records to score", file=sys.stderr)
        return 1

    pairs = []
    cross_records = []
    for key in comparable:
        label_primitives_for_key = label_primitives(labels[key], args.task)
        predicted_primitives = extract_predicted_primitives(predictions[key])
        pairs.append((label_primitives_for_key, predicted_primitives))
        cross_records.append({
            "true_score": validated_score(labels[key], role="label", task=args.task),
            "predicted_score": validated_score(predictions[key], role="prediction", task=args.task),
            "label_primitives": label_primitives_for_key,
            "predicted_primitives": predicted_primitives,
        })

    ordinal_pairs = sum(row["true_score"] is not None and row["predicted_score"] is not None
                        for row in cross_records)
    print(f"quality-passing matched {len(comparable)} | pairs with both ordinal scores {ordinal_pairs}")
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
    unannotated_temporal_pairs: dict[str, int] = {}
    cycle_only_evidence = 0
    if args.task == "gait":
        for source, primitive in INTERVAL_SOURCES.items():
            span_pairs = []
            skipped = 0
            for key in comparable:
                if not _temporal_label_annotated(labels[key], source, primitive):
                    skipped += 1
                    continue
                truth, t_cycles = temporal_evidence(labels[key], source, primitive)
                predicted, p_cycles = temporal_evidence(predictions[key], source, primitive)
                cycle_only_evidence += t_cycles + p_cycles
                span_pairs.append((truth, predicted))
            unannotated_temporal_pairs[primitive] = skipped
            if span_pairs:
                result = temporal_event_agreement(span_pairs, tolerance_sec=args.event_tolerance)
                result["false_positive_events"] = result["predicted_events"] - result["matched"]
                result["false_negative_events"] = result["labeled_events"] - result["matched"]
                result["annotated_pairs"] = len(span_pairs)
                result["unannotated_pairs_excluded"] = skipped
                temporal[primitive] = result
    if cycle_only_evidence:
        print(f"  note: {cycle_only_evidence} cycle-number evidence span(s) excluded from seconds-based event matching")

    if temporal:
        print_table(
            f"4) 시간 구간 예측 (매칭 기준: 중점 ±{args.event_tolerance:g}초)",
            [
                [primitive, r["labeled_events"], r["predicted_events"], r["matched"], r["false_positive_events"],
                 "-" if r["precision"] is None else f"{r['precision']:.1%}",
                 "-" if r["recall"] is None else f"{r['recall']:.1%}",
                 "-" if r["f1"] is None else f"{r['f1']:.3f}",
                 "-" if r["mean_iou_of_matched"] is None else f"{r['mean_iou_of_matched']:.3f}"]
                for primitive, r in temporal.items()
            ],
            ["primitive", "정답구간", "예측구간", "매칭", "오탐", "정밀도", "재현율", "F1", "평균 IoU"],
        )
        if "turning_impairment" in temporal:
            print("  → 회전은 구간이 짧아(중앙값 0.7초) IoU가 작은 어긋남에도 무너집니다.")
            print("     그래서 중점 거리로 매칭하고 IoU는 참고로만 봅니다.")
        if "freezing_of_gait" in temporal:
            print("  → 동결은 실제 지속시간(중앙값 6.3초)이라 IoU도 의미 있게 읽힙니다.")

    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {"matched_identity_keys": len(shared), "scored_pairs": ordinal_pairs,
                 "quality_passing_pairs": len(comparable),
                 "score_missing_pairs": len(comparable) - ordinal_pairs,
                 "held_pairs_excluded": held_count,
                 "quality_unverified_pairs_excluded": uncertified_count,
                 "provenance_unverified_pairs": incomplete_provenance,
                 "split_claim_matching_pairs": split_claim_matching_pairs,
                 "split_claim_missing_pairs": split_claim_missing_pairs,
                 "split_binding_verified": False,
                 "held_out_verified": False,
                 "unmatched_labels": len(labels)-len(shared),
                 "unmatched_predictions": len(predictions)-len(shared),
                 "dataset": args.dataset, "task": args.task, "research_only": True,
                 "cycle_only_time_evidence_excluded": cycle_only_evidence,
                 "unannotated_temporal_pairs_excluded": unannotated_temporal_pairs,
                 "distribution": distribution, "agreement": agreement,
                 "score_vs_observation": cross, "temporal_events": temporal},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nJSON 저장: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
