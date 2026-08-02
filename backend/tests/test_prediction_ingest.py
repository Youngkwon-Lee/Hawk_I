"""
Tests for offline VLM prediction ingestion into the shared patient timeline.
"""

import pytest

from services import prediction_ingest
from services.prediction_ingest import (
    PredictionValidationError,
    analysis_id_for,
    attach_research_provenance,
    load_predictions,
    prediction_to_result,
)


def _prediction(**overrides):
    base = {
        "clip_id": "PD4T_S12_gait_03",
        "task": "gait",
        "predicted_score": 2,
        "rationale": "보폭 감소와 팔 흔들림 저하가 관찰됨",
        "dataset": "PD4T",
        "split": "test",
        "model": "qwen3-vl-4b-c3",
        "condition": "C3",
        "confidence": 0.81,
        "true_score": 2,
        "subject_ref": "S12",
    }
    base.update(overrides)
    return base


def test_prediction_converts_to_analysis_result_shape():
    result = prediction_to_result(_prediction())

    assert result["video_type"] == "gait"
    assert result["updrs_score"]["total_score"] == 2
    assert result["updrs_score"]["severity"] == "Mild"
    assert result["scoring_method"] == "vlm_offline"
    assert result["ml_model_type"] == "qwen3-vl-4b-c3"
    assert result["ai_interpretation"]["summary"].startswith("보폭")
    assert result["updrs_score"]["details"]["reference_score"] == 2


def test_analysis_id_is_stable_for_reingest():
    prediction = _prediction()
    assert analysis_id_for(prediction) == analysis_id_for(dict(prediction))
    assert analysis_id_for(prediction) != analysis_id_for(_prediction(condition="C0"))


def test_rationale_absent_leaves_interpretation_out():
    result = prediction_to_result(_prediction(rationale=""))
    assert "ai_interpretation" not in result


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"clip_id": ""}, "missing required field"),
        ({"predicted_score": "high"}, "numeric"),
        ({"predicted_score": 9}, "0-4"),
        ({"dataset": ""}, "dataset is required"),
    ],
)
def test_invalid_predictions_are_rejected(overrides, message):
    with pytest.raises(PredictionValidationError) as exc:
        prediction_to_result(_prediction(**overrides))
    assert message in str(exc.value)


def test_research_provenance_marks_row_as_non_clinical():
    row = attach_research_provenance(
        {"measurement_context": {"app_source": "hawk_i"}, "category": ["motor-assessment"]},
        _prediction(),
    )

    provenance = row["measurement_context"]["research_provenance"]
    assert provenance["is_research_prediction"] is True
    assert provenance["is_clinical_record"] is False
    assert provenance["serving_mode"] == "offline_batch"
    assert provenance["dataset"] == "PD4T"
    assert provenance["split"] == "test"
    assert provenance["research_subject_ref"] == "S12"
    assert provenance["reference_score"] == 2
    assert "research-prediction" in row["category"]
    assert row["measurement_context"]["app_source"] == "hawk_i"


def test_provenance_uses_observed_at_when_supplied():
    row = attach_research_provenance(
        {"measurement_context": {}},
        _prediction(observed_at="2026-08-09T10:00:00Z"),
    )
    assert row["effective_datetime"] == "2026-08-09T10:00:00Z"


def test_load_predictions_reports_bad_lines_without_dropping_good_ones():
    lines = [
        '{"clip_id":"a","task":"gait","predicted_score":1,"dataset":"PD4T"}',
        "# comment line",
        "",
        "{not json}",
        '{"clip_id":"b","task":"gait","predicted_score":9,"dataset":"PD4T"}',
        '{"clip_id":"c","task":"finger_tapping","predicted_score":3,"dataset":"PD4T"}',
    ]

    predictions, summary = load_predictions(lines)

    assert [p["clip_id"] for p in predictions] == ["a", "c"]
    assert summary.total == 4
    assert summary.converted == 2
    assert len(summary.skipped) == 2
    reasons = " ".join(reason for _, reason in summary.skipped)
    assert "invalid JSON" in reasons
    assert "0-4" in reasons


def test_severity_mapping_covers_updrs_range():
    for score, expected in [(0, "Normal"), (1, "Slight"), (2, "Mild"), (3, "Moderate"), (4, "Severe")]:
        assert prediction_ingest._severity(score) == expected


def _ontology_prediction(**overrides):
    base = {
        "clip_id": "PD4T_S12_gait_03",
        "task": "gait",
        "dataset": "PD4T",
        "split": "test",
        "model": "qwen3-vl-4b-c3",
        "condition": "C3",
        "quality_gate": {"status": "pass", "reasons": []},
        "primitives": {
            "gait_speed_reduction": {
                "observability": "observed",
                "severity": 2,
                "confidence": "medium",
                "evidence": [{"start_sec": 2.1, "end_sec": 5.4}],
            },
            "freezing_of_gait": {
                "observability": "unobservable",
                "severity": None,
                "confidence": "high",
            },
        },
        "score_anchor": {"updrs_3_10": 2, "source": "trusted_import", "confidence": "medium"},
    }
    base.update(overrides)
    return base


def test_ontology_prediction_carries_primitives_and_anchor_score():
    result = prediction_to_result(_ontology_prediction())

    assert result["updrs_score"]["total_score"] == 2
    assert result["updrs_score"]["details"]["score_anchor_source"] == "trusted_import"
    assert result["primitives"]["gait_speed_reduction"]["severity"] == 2
    assert "gait_speed_reduction 2단계" in result["ai_interpretation"]["summary"]


def test_unobserved_primitive_must_not_carry_severity():
    bad = _ontology_prediction()
    bad["primitives"]["freezing_of_gait"]["severity"] = 0
    with pytest.raises(PredictionValidationError) as exc:
        prediction_to_result(bad)
    assert "not visible is not the same as normal" in str(exc.value)


def test_positive_finding_requires_evidence_span():
    bad = _ontology_prediction()
    bad["primitives"]["gait_speed_reduction"].pop("evidence")
    with pytest.raises(PredictionValidationError) as exc:
        prediction_to_result(bad)
    assert "evidence span" in str(exc.value)


def test_held_clip_is_accepted_without_a_score():
    held = _ontology_prediction(
        quality_gate={"status": "hold", "reasons": ["body_occluded"], "note": "하반신 가려짐"},
        primitives={"gait_speed_reduction": {"observability": "uncertain", "severity": None, "confidence": "low"}},
        score_anchor={"updrs_3_10": None, "source": "not_scored", "confidence": "not_scored"},
    )
    result = prediction_to_result(held)

    assert result["updrs_score"]["total_score"] is None
    assert result["performability_assessment"]["status"] == "hold"
    assert result["score_advisory"]["summary"] == "body_occluded"


def test_v1_flat_format_still_works():
    result = prediction_to_result(_prediction())
    assert result["updrs_score"]["total_score"] == 2
    assert result["ai_interpretation"]["summary"].startswith("보폭")


def test_primitives_and_quality_gate_reach_the_observation_row():
    row = attach_research_provenance({"measurement_context": {}}, _ontology_prediction())
    assert row["measurement_context"]["primitives"]["gait_speed_reduction"]["severity"] == 2
    assert row["measurement_context"]["quality_gate"]["status"] == "pass"
