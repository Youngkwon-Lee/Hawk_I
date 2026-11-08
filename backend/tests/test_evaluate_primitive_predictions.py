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
