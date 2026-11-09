"""Synthetic, identity-only regressions for the research evaluation boundary."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_primitive_predictions import (  # noqa: E402
    index_by_identity,
    label_primitives,
    label_score,
    pair_provenance_unverified,
    temporal_evidence,
)


def _prediction(**updates):
    result = {"clip_id": "synthetic-001", "task": "finger_tapping", "side": "left",
              "dataset": "SYNTHETIC", "model": "demo", "condition": "C3",
              "quality_gate": {"status": "pass", "reasons": []},
              "primitives": {
                  "tapping_speed": {"observability": "observed", "severity": 0, "evidence": []}
              }, "score_anchor": {"updrs_3_4": None, "source": "not_scored"}}
    result.update(updates)
    return result


def _label(**updates):
    result = {"clip_id": "synthetic-001", "task": "finger_tapping", "side": "left",
              "ontology_version": "finger_tapping_primitive_v0.1",
              "review_status": "reviewed", "quality_gate": {"status": "pass", "reasons": []},
              "primitives": {
                  "tapping_speed": {"observability": "observed", "severity": 0, "evidence": []}
              }, "score_anchor": {"updrs_3_4": None}}
    result.update(updates)
    return result


def _index(rows, role="label", task="finger_tapping"):
    return index_by_identity(rows, role=role, task=task, dataset="SYNTHETIC")


def test_two_sides_are_separate_identity_keys():
    indexed = _index([_label(), _label(side="right")])
    assert len(indexed) == 2


def test_duplicate_clip_side_is_rejected_instead_of_overwritten():
    with pytest.raises(ValueError, match="duplicate evaluation key"):
        _index([_label(), _label()])


def test_mismatched_dataset_and_task_rejected():
    with pytest.raises(ValueError, match="dataset mismatch"):
        _index([_label(dataset="OTHER")])
    with pytest.raises(ValueError, match="wrong task"):
        _index([_label(task="gait")])


def test_missing_hand_side_and_mixed_model_conditions_rejected():
    with pytest.raises(ValueError, match="side required"):
        _index([_prediction(side=None)], role="prediction")
    with pytest.raises(ValueError, match="mixed model/condition"):
        _index([_prediction(), _prediction(clip_id="synthetic-002", model="other")], role="prediction")


def test_unreviewed_label_and_held_score_rejected():
    with pytest.raises(ValueError, match="incomplete reviewer"):
        _index([_label(review_status="draft")])
    with pytest.raises(ValueError, match="held clip"):
        _index([_label(quality_gate={"status": "hold", "reasons": ["hand_occluded"]},
                       score_anchor={"updrs_3_4": 2})])


def test_held_prediction_cannot_retain_score_or_primitive_severity():
    with pytest.raises(ValueError, match="held clip"):
        _index([_prediction(quality_gate={"status": "hold", "reasons": ["hand_occluded"]},
                            score_anchor={"updrs_3_4": 1})], role="prediction")
    with pytest.raises(ValueError, match="held clip"):
        _index([_prediction(quality_gate={"status": "hold", "reasons": ["hand_occluded"]})],
               role="prediction")


def test_structured_labels_and_task_specific_anchor():
    label = _label(score_anchor={"updrs_3_4": 3}, primitives={
        "opening_amplitude": {"observability": "observed", "severity": 1,
                              "evidence": [{"cycle_start": 2, "cycle_end": 3}]}
    })
    assert label_primitives(label, "finger_tapping") == {"opening_amplitude": 1.0}
    assert label_score(label, "finger_tapping") == 3


def test_seconds_alias_and_cycles_are_not_interchanged():
    pred = _prediction(primitives={
        "turning_impairment": {"observability": "observed", "severity": 1,
                               "evidence": [{"start_sec": 1.5, "end_sec": 2.5},
                                            {"cycle_start": 3, "cycle_end": 4}]}
    })
    spans, cycle_count = temporal_evidence(pred, "turning_intervals", "turning_impairment")
    assert spans == [(1.5, 2.5)]
    assert cycle_count == 1


def test_provenance_disagreement_rejected_and_missing_both_not_verified():
    with pytest.raises(ValueError, match="ontology_version"):
        pair_provenance_unverified(_label(ontology_version="v1"), _prediction(ontology_version="v2"))
    assert pair_provenance_unverified(_label(), _prediction()) is True


def test_malformed_evidence_is_rejected():
    with pytest.raises(ValueError, match="ambiguous time/cycle"):
        _index([_label(primitives={
            "halt": {"observability": "observed", "severity": 1, "evidence": [
                {"start_s": 1.2, "cycle_start": 2, "cycle_end": 3}
            ]}
        })])


def test_held_numeric_training_label_not_scored():
    record = {"clip_id": "synthetic-gait", "task": "gait",
              "quality_gate": {"status": "hold", "reasons": ["body_occluded"]},
              "primitives": {"gait_speed_reduction": 0},
              "score_anchor": {"updrs_3_10": None}}
    with pytest.raises(ValueError, match="held clip"):
        _index([record], task="gait")


def test_mixed_model_revisions_are_rejected():
    a = _prediction(clip_id="synthetic-101", model_version="v1")
    b = _prediction(clip_id="synthetic-102", model_version="v2")
    with pytest.raises(ValueError, match="mixed model/condition"):
        _index([a, b], role="prediction")


def test_equal_declared_hash_is_not_independent_source_proof():
    # Both parties can copy the same digest without reading the actual media.
    claim = "a" * 64
    label = _label(dataset="SYNTHETIC", source_sha256=claim)
    prediction = _prediction(
        ontology_version="finger_tapping_primitive_v0.1",
        source_sha256=claim,
    )
    assert pair_provenance_unverified(label, prediction) is True


def test_conflicting_claimed_hash_and_dataset_are_rejected():
    with pytest.raises(ValueError, match="source_sha256"):
        pair_provenance_unverified(
            _label(source_sha256="a" * 64),
            _prediction(source_sha256="b" * 64),
        )
    with pytest.raises(ValueError, match="dataset"):
        pair_provenance_unverified(
            _label(dataset="OTHER"),
            _prediction(),
        )


def test_nested_gait_hold_cannot_be_scored_as_legacy_pass():
    row = {"clip_id": "synthetic-gait", "item": "gait",
           "label": {"quality_gate": {"status": "hold", "reasons": ["body_occluded"]},
                     "updrs_score": 2,
                     "primitives": {"gait_speed_reduction": 2}}}
    with pytest.raises(ValueError, match="held clip"):
        _index([row], task="gait")


def test_nested_gait_hold_with_null_score_is_accepted_and_excluded():
    from evaluate_primitive_predictions import _score_held

    row = {"clip_id": "synthetic-gait", "item": "gait",
           "label": {"quality_gate": {"status": "hold", "reasons": ["body_occluded"]},
                     "score_anchor": {"updrs_3_10": None},
                     "primitives": {"gait_speed_reduction": None}}}
    assert len(_index([row], task="gait")) == 1
    assert _score_held(row, "label", "gait") is True


def test_conflicting_nested_and_outer_gates_fail_closed():
    row = {"clip_id": "synthetic-gait", "item": "gait",
           "quality_gate": {"status": "pass", "reasons": []},
           "label": {"quality_gate": {"status": "hold", "reasons": ["body_occluded"]},
                     "updrs_score": 2}}
    with pytest.raises(ValueError, match="conflicting nested/top-level quality_gate"):
        _index([row], task="gait")


def test_nested_unreviewed_and_malformed_primitive_evidence_rejected():
    with pytest.raises(ValueError, match="incomplete reviewer"):
        _index([{"clip_id": "synthetic-gait", "item": "gait",
                 "label": {"review_status": "draft", "updrs_score": 2}}], task="gait")
    with pytest.raises(ValueError, match="invalid seconds interval"):
        _index([{"clip_id": "synthetic-gait", "item": "gait",
                 "label": {"quality_gate": {"status": "pass", "reasons": []},
                           "primitives": {"turning_impairment": {
                               "observability": "observed", "severity": 1,
                               "evidence": [{"start_s": 3, "end_s": 3}]}}}}], task="gait")


def test_inconsistent_label_gate_reasons_rejected():
    with pytest.raises(ValueError, match="quality_gate reasons"):
        _index([_label(quality_gate={"status": "pass", "reasons": ["hand_occluded"]})])
    with pytest.raises(ValueError, match="quality_gate reasons"):
        _index([_label(quality_gate={"status": "hold", "reasons": []},
                       primitives={"tapping_speed": {"observability": "unobservable",
                                                     "severity": None, "evidence": []}},
                       score_anchor={"updrs_3_4": None})])



def test_inconsistent_flat_score_and_anchor_rejected():
    with pytest.raises(ValueError, match="conflicting ordinal score fields"):
        _index([_label(updrs_score=2, score_anchor={"updrs_3_4": 4})])
    with pytest.raises(ValueError, match="conflicting ordinal score fields"):
        _index([_prediction(predicted_score=2, score_anchor={"updrs_3_4": 4})],
               role="prediction")


def test_matching_zero_score_and_anchor_are_valid():
    from evaluate_primitive_predictions import validated_score
    label = _label(updrs_score="0", score_anchor={"updrs_3_4": 0})
    _index([label])
    assert validated_score(label, role="label", task="finger_tapping") == 0


def test_invalid_human_scores_are_not_grading_inputs():
    for value in (True, -1, 5, float("nan"), float("inf"), "not-a-number"):
        with pytest.raises(ValueError, match="score"):
            _index([_label(updrs_score=value)])


def test_scoreless_primitive_pair_not_claimed_scored(tmp_path):
    import json
    from evaluate_primitive_predictions import main

    label = _label(dataset="SYNTHETIC")
    pred = _prediction()
    labels = tmp_path / "labels.jsonl"
    preds = tmp_path / "preds.jsonl"
    report = tmp_path / "report.json"
    labels.write_text(json.dumps(label) + "\n", encoding="utf-8")
    preds.write_text(json.dumps(pred) + "\n", encoding="utf-8")
    code = main(["--labels", str(labels), "--predictions", str(preds),
                 "--task", "finger_tapping", "--dataset", "SYNTHETIC",
                 "--json-out", str(report)])
    assert code == 0
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["quality_passing_pairs"] == 1
    assert summary["scored_pairs"] == 0
    assert summary["score_missing_pairs"] == 1
    assert summary["score_vs_observation"]["evaluated"] == 0


def test_nonfinite_evaluation_tolerances_rejected(tmp_path):
    from evaluate_primitive_predictions import main
    labels = tmp_path / "labels.jsonl"
    preds = tmp_path / "preds.jsonl"
    labels.write_text("", encoding="utf-8")
    preds.write_text("", encoding="utf-8")
    for arg, value in (("--score-tolerance", "nan"),
                       ("--event-tolerance", "-1"),
                       ("--event-tolerance", "inf")):
        code = main(["--labels", str(labels), "--predictions", str(preds),
                     "--task", "gait", "--dataset", "SYNTHETIC", arg, value])
        assert code == 2


def test_exported_legacy_gait_events_are_not_silently_dropped():
    exported = {"clip_id": "SYNTHETIC_GAIT",
                "events": {"gait_events": {"turning_intervals": [
                    {"start": 1.0, "end": 2.0}]}}}
    assert temporal_evidence(exported, "turning_intervals", "turning_impairment") == (
        [(1.0, 2.0)], 0
    )


def test_nested_label_primitive_seconds_are_evaluated():
    exported = {"clip_id": "SYNTHETIC_GAIT", "label": {"primitives": {
        "turning_impairment": {"observability": "observed", "severity": 1,
                               "evidence": [{"start_s": 1.0, "end_s": 2.0}]}}}}
    assert temporal_evidence(exported, "turning_intervals", "turning_impairment") == (
        [(1.0, 2.0)], 0
    )


def _synthetic_gait_score_pair(clip_id, *, quality_gate=True, event_reference=None):
    label = {"clip_id": clip_id, "item": "gait", "dataset": "SYNTHETIC",
             "label": {"updrs_score": 1, "gait_speed": "mild"}}
    if quality_gate:
        label["quality_gate"] = {"status": "pass", "reasons": []}
    if event_reference is not None:
        label["events"] = {"gait_events": {"turning_intervals": event_reference}}
    pred = {"clip_id": clip_id, "task": "gait", "dataset": "SYNTHETIC",
            "model": "SYNTHETIC_MODEL", "condition": "C3",
            "quality_gate": {"status": "pass", "reasons": []},
            "predicted_score": 1,
            "primitives": {"turning_impairment": {
                "observability": "observed", "severity": 1,
                "evidence": [{"start_s": 1.0, "end_s": 2.0}]}}}
    return label, pred


def _score_gait_jsonl(tmp_path, labels, predictions):
    import json
    from evaluate_primitive_predictions import main

    labels_path = tmp_path / "labels.jsonl"
    preds_path = tmp_path / "preds.jsonl"
    report_path = tmp_path / "report.json"
    labels_path.write_text("".join(json.dumps(v) + "\n" for v in labels), encoding="utf-8")
    preds_path.write_text("".join(json.dumps(v) + "\n" for v in predictions), encoding="utf-8")
    status = main(["--labels", str(labels_path), "--predictions", str(preds_path),
                   "--task", "gait", "--dataset", "SYNTHETIC",
                   "--json-out", str(report_path)])
    return status, json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None


def test_fractional_updrs_scores_rejected_for_both_roles():
    for invalid in (2.5, "1.5", 3.1):
        with pytest.raises(ValueError, match="integer score"):
            _index([_label(updrs_score=invalid)])
        with pytest.raises(ValueError, match="integer score"):
            _index([_prediction(predicted_score=invalid)], role="prediction")
    assert len(_index([_label(updrs_score=2.0)])) == 1


def test_legacy_quality_unknown_does_not_enter_certified_score(tmp_path):
    approved_label, approved_pred = _synthetic_gait_score_pair("SYN_CERTIFIED", event_reference=[])
    unknown_label, unknown_pred = _synthetic_gait_score_pair("SYN_UNKNOWN", quality_gate=False)
    status, report = _score_gait_jsonl(
        tmp_path, [approved_label, unknown_label], [approved_pred, unknown_pred])
    assert status == 0
    assert report["matched_identity_keys"] == 2
    assert report["quality_passing_pairs"] == 1
    assert report["quality_unverified_pairs_excluded"] == 1
    assert report["held_pairs_excluded"] == 0
    assert report["scored_pairs"] == 1


def test_only_uncertified_legacy_labels_do_not_claim_pass(tmp_path):
    label, pred = _synthetic_gait_score_pair("SYN_UNCERTIFIED", quality_gate=False)
    status, report = _score_gait_jsonl(tmp_path, [label], [pred])
    assert status == 1
    assert report is None


def test_legacy_invalid_interval_values_rejected_before_grading():
    for start, end in ((-2.0, -1.0), (float("nan"), 2.0), (1.0, float("inf")),
                       (3.0, 2.0), (1.0, 1.0), (True, 2.0), ("oops", 2.0)):
        label, _ = _synthetic_gait_score_pair("SYN_BAD_EVENT", event_reference=[
            {"start": start, "end": end}])
        with pytest.raises(ValueError, match="invalid legacy"):
            _index([label], task="gait")


def test_conflicting_legacy_event_sources_rejected():
    label, _ = _synthetic_gait_score_pair(
        "SYN_CONFLICT", event_reference=[{"start": 1.0, "end": 2.0}])
    label["gait_events"] = {"turning_intervals": [{"start": 3.0, "end": 4.0}]}
    with pytest.raises(ValueError, match="conflicting legacy"):
        _index([label], task="gait")


def test_explicit_zero_reference_events_report_model_false_positives(tmp_path):
    label, prediction = _synthetic_gait_score_pair("SYN_EVENT_NEGATIVE", event_reference=[])
    status, report = _score_gait_jsonl(tmp_path, [label], [prediction])
    assert status == 0
    turns = report["temporal_events"]["turning_impairment"]
    assert turns["labeled_events"] == 0
    assert turns["predicted_events"] == 1
    assert turns["matched"] == 0
    assert turns["false_positive_events"] == 1
    assert turns["annotated_pairs"] == 1
    assert turns["unannotated_pairs_excluded"] == 0


def test_missing_temporal_annotation_is_not_assumed_negative(tmp_path):
    label, prediction = _synthetic_gait_score_pair("SYN_EVENT_UNANNOTATED")
    status, report = _score_gait_jsonl(tmp_path, [label], [prediction])
    assert status == 0
    assert "turning_impairment" not in report["temporal_events"]
    assert report["unannotated_temporal_pairs_excluded"]["turning_impairment"] == 1


def test_declared_split_mismatch_rejected():
    from evaluate_primitive_predictions import declared_split
    label = _label(meta={"split": "train"})
    prediction = _prediction(split="test")
    with pytest.raises(ValueError, match="provenance mismatch: split"):
        pair_provenance_unverified(label, prediction)
    assert declared_split(label, "label") == "train"


def test_split_identity_conflict_within_label_is_rejected():
    label = _label(split="test", meta={"split": "train"})
    with pytest.raises(ValueError, match="conflicting declared split"):
        _index([label])


def test_matching_split_claims_remain_unverified(tmp_path):
    label = _label(dataset="SYNTHETIC", meta={"split": "test"})
    prediction = _prediction(split="test")
    from evaluate_primitive_predictions import main
    import json

    label_path, pred_path, report_path = (tmp_path / name for name in ("labels.jsonl", "preds.jsonl", "report.json"))
    label_path.write_text(json.dumps(label) + "\n", encoding="utf-8")
    pred_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    assert main(["--labels", str(label_path), "--predictions", str(pred_path),
                 "--task", "finger_tapping", "--dataset", "SYNTHETIC",
                 "--json-out", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split_claim_matching_pairs"] == 1
    assert report["split_claim_missing_pairs"] == 0
    assert report["split_binding_verified"] is False
    assert report["held_out_verified"] is False
    assert report["provenance_unverified_pairs"] == 1


def test_missing_split_claim_is_counted_not_inferred(tmp_path):
    label = _label(dataset="SYNTHETIC", meta={"split": "test"})
    pred = _prediction()
    from evaluate_primitive_predictions import main
    import json

    label_path, pred_path, report_path = (tmp_path / name for name in ("labels.jsonl", "preds.jsonl", "report.json"))
    label_path.write_text(json.dumps(label) + "\n", encoding="utf-8")
    pred_path.write_text(json.dumps(pred) + "\n", encoding="utf-8")
    assert main(["--labels", str(label_path), "--predictions", str(pred_path),
                 "--task", "finger_tapping", "--dataset", "SYNTHETIC",
                 "--json-out", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split_claim_matching_pairs"] == 0
    assert report["split_claim_missing_pairs"] == 1
    assert report["held_out_verified"] is False


def test_multiple_declared_splits_cannot_share_one_report(tmp_path):
    from evaluate_primitive_predictions import main
    import json

    labels = [_label(clip_id=f"SYNTH_{split}", dataset="SYNTHETIC", meta={"split": split})
              for split in ("train", "test")]
    predictions = [_prediction(clip_id=f"SYNTH_{split}", split=split)
                   for split in ("train", "test")]
    label_path, pred_path = (tmp_path / name for name in ("labels.jsonl", "preds.jsonl"))
    label_path.write_text("".join(json.dumps(x) + "\n" for x in labels), encoding="utf-8")
    pred_path.write_text("".join(json.dumps(x) + "\n" for x in predictions), encoding="utf-8")
    assert main(["--labels", str(label_path), "--predictions", str(pred_path),
                 "--task", "finger_tapping", "--dataset", "SYNTHETIC"]) == 2
