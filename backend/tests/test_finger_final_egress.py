"""Synthetic-only safety tests for final Finger Tapping result egress."""
import json
from copy import deepcopy

import pytest

from services.finger_final_egress import (
    build_finger_research_hold_result, is_finger_task,
)
from services.analysis_media import _is_patient_linked


def _result():
    return {
        "success": True, "id": "synthetic-001", "video_type": "finger_tapping",
        "physio_context": {
            "subject_person_id": "synthetic-person",
            "organization_id": "synthetic-org",
            "activity_session_id": "synthetic-session",
            "persistence_owner": "parkicheck",
            "contract_version": "synthetic-v1",
            "extra_score_note": "SECRET_SCORE_IN_CONTEXT",
        },
        "patient_id": "SECRET_PATIENT_ALIAS",
        "medication_context": {"dose_note": "SECRET_MEDICATION"},
        "assessment_session_id": "synthetic-session",
        "timeline_contract_version": "synthetic-v1",
        "video_metadata": {"fps": 80.0, "total_frames": 400},
        "updrs_score": {"total_score": 0, "details": {"pipeline_reference": {"score": 2}}},
        "ai_interpretation": {"summary": "SECRET_SCORE_INTERPRETATION"},
        "score_advisory": {"summary": "자동 점수 사용 가능"},
        "reasoning_log": [{"content": "UPDRS Estimation: Score 0", "meta": {"total_score": 0}}],
        "vlm_results": [{"score": 3.0, "reasoning": "SECRET_VLM"}],
        "primitives": {"tapping_severity": 4},
        "metrics": {"tapping_speed": 0.3, "total_score": 4},
        "visualization_urls": {"heatmap": "/files/synthetic_heatmap.jpg"},
        "skeleton_data": {
            "mode": "hand", "total_frames": 134,
            "original_video_url": "/files/synthetic.mp4",
            "keypoints": [{"secret": "SECRET_LANDMARK"}],
        },
    }


def test_post_model_score_and_nested_narrative_are_not_serialized():
    raw = _result()
    original = deepcopy(raw)
    projected = build_finger_research_hold_result(raw)
    serial = json.dumps(projected, ensure_ascii=False)
    for forbidden in (
        "SECRET_SCORE_IN_CONTEXT", "SECRET_PATIENT_ALIAS", "SECRET_MEDICATION",
        "SECRET_SCORE_INTERPRETATION", "UPDRS Estimation: Score 0",
        "SECRET_VLM", "SECRET_LANDMARK", "자동 점수 사용 가능",
        "pipeline_reference", "tapping_severity", "total_score",
    ):
        assert forbidden not in serial
    assert projected["updrs_score"] is None
    assert projected["clinical_score"] is None
    assert projected["ai_interpretation"] is None
    assert projected["reasoning_log"] == []
    assert raw == original


def test_patient_linked_auth_context_survives_score_hold():
    projected = build_finger_research_hold_result(_result())
    assert _is_patient_linked(projected) is True
    assert projected["physio_context"]["subject_person_id"] == "synthetic-person"
    assert projected["physio_context"]["persistence_owner"] == "parkicheck"
    assert projected["physio_context"]["activity_session_id"] == "synthetic-session"
    assert projected["skeleton_data"]["original_video_url"].endswith("synthetic.mp4")
    assert projected["visualization_urls"]["heatmap"].endswith(".jpg")


def test_unlinked_result_does_not_gain_synthetic_identity_or_score():
    source = _result()
    source["physio_context"] = None
    projected = build_finger_research_hold_result(source)
    assert _is_patient_linked(projected) is False
    assert projected["physio_context"] is None
    assert projected["updrs_score"] is None
    assert projected["research_review"]["state"] == "hold"


@pytest.mark.parametrize("task", ["Finger Tapping", "finger-tapping",
                                     "finger_tapping_right", "finger_tapping"])
def test_finger_task_aliases_are_held(task):
    source = _result()
    source["video_type"] = task
    out = build_finger_research_hold_result(source)
    assert out["video_type"] == "finger_tapping"
    assert out["research_review"]["state"] == "hold"
    assert is_finger_task(task)


def test_nonfinger_response_not_changed_by_candidate():
    source = {"video_type": "gait", "updrs_score": {"total_score": 3}}
    assert build_finger_research_hold_result(source) is source


def test_invalid_source_is_rejected():
    with pytest.raises(ValueError):
        build_finger_research_hold_result(None)


def test_background_egress_after_vlm_clears_score_before_persistence_and_file(tmp_path, monkeypatch):
    """Mock heavy CV/inference; exercise the actual PR20 background result path."""
    import json
    from types import SimpleNamespace

    from flask import Flask
    from domain.context import Report
    from routes import analyze as route
    from services import finetuned_vlm as ft
    from services.analysis_media import write_analysis_access_record, classify_direct_file_access

    analysis_id = "synthetic-research-hold"
    video_path = tmp_path / "synthetic.mp4"
    video_path.write_bytes(b"synthetic-not-a-real-video")
    physio_context = {
        "subject_person_id": "22222222-2222-4222-8222-222222222222",
        "organization_id": "11111111-1111-4111-8111-111111111111",
        "activity_session_id": "33333333-3333-4333-8333-333333333333",
        "persistence_owner": "parkicheck",
    }
    write_analysis_access_record(str(tmp_path), analysis_id, physio_context)

    class FakeOrchestrator:
        def process(self, ctx, on_progress_update=None):
            ctx.status = "report_done"
            ctx.task_type = "finger_tapping"
            ctx.skeleton_data = {
                "landmarks": [{
                    "frame_number": 0,
                    "timestamp": 0.0,
                    "landmarks": [{"id": 4, "x": 0.5, "y": 0.3, "z": 0.0}],
                }],
            }
            ctx.kinematic_metrics = {"tapping_speed": 2.0, "total_score": 4}
            ctx.clinical_scores = {"total_score": 0.0, "severity": "UNREVIEWED_SEVERITY"}
            ctx.vision_meta = {
                "fps": 80.0, "frame_count": 400, "frame_skip": 3,
                "roi": (0, 0, 96, 64), "confidence": .9,
                "quality_flag": "ok",
            }
            ctx.report = Report(
                summary_for_patient="UNREVIEWED_CLINICAL_SUMMARY",
                summary_for_clinician="UNREVIEWED_EXPLANATION",
                recommendations=["UNREVIEWED_RECOMMENDATION"],
            )
            ctx.log("clinical", "UPDRS Estimation: Score 0",
                    meta={"total_score": 0, "private": "UNREVIEWED_SCORE_LOG"})
            return ctx

    monkeypatch.setattr(route, "OrchestratorAgent", FakeOrchestrator)
    monkeypatch.setattr(route, "update_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(route, "complete_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(route, "generate_visualization_data",
                        lambda **kwargs: {"leak": "UNREVIEWED_METRIC_CHART"})
    monkeypatch.setattr(route, "detect_events",
                        lambda **kwargs: [{"label": "UNREVIEWED_EVENT"}])
    monkeypatch.setattr(route, "get_finger_performability_gate",
                        lambda: SimpleNamespace(
                            assess=lambda *args, **kwargs: SimpleNamespace(
                                status="performable",
                                summary="UNREVIEWED_ADVISORY",
                            )
                        ))
    monkeypatch.setattr(ft, "frames_to_base64", lambda *args, **kwargs: (["fake-frame"], 5.0))
    monkeypatch.setattr(ft, "analyze", lambda *args, **kwargs: {
        "scoring_method": "finetuned_vlm",
        "ml_model_type": "UNREVIEWED_MODEL",
        "updrs_score": {
            "total_score": 3.0, "score": 3.0,
            "details": {"model": "UNREVIEWED_MODEL"},
        },
        "ai_interpretation": {"summary": "UNREVIEWED_VLM_SUMMARY"},
        "primitives": {"score_label": "UNREVIEWED_PRIMITIVE"},
    })

    def no_db_write(*args, **kwargs):
        raise AssertionError("No scored Supabase or ParkiCheck persistence permitted")

    monkeypatch.setattr(route, "persist_analysis_observation", no_db_write)
    route.process_video_background(
        str(video_path), analysis_id, "UNREVIEWED_PATIENT_ALIAS",
        "finger_tapping", {"UPLOAD_FOLDER": str(tmp_path)},
        physio_context=physio_context,
    )
    out_path = tmp_path / (analysis_id + "_result.json")
    assert out_path.is_file(), "Local result serialization did not complete"
    result = json.loads(out_path.read_text())
    assert result["research_review"]["state"] == "hold"
    assert result["updrs_score"] is None
    assert result["integrations"]["supabase_observation"]["saved"] is False
    assert result["clinical_score"] is None
    assert result["reasoning_log"] == []
    assert _is_patient_linked(result) is True
    assert result["physio_context"]["activity_session_id"] == physio_context["activity_session_id"]

    wire = json.dumps(result, ensure_ascii=False)
    for forbidden in (
        "UNREVIEWED_SEVERITY", "UNREVIEWED_CLINICAL_SUMMARY",
        "UNREVIEWED_EXPLANATION", "UNREVIEWED_RECOMMENDATION",
        "UNREVIEWED_SCORE_LOG", "UNREVIEWED_ADVISORY",
        "UNREVIEWED_METRIC_CHART", "UNREVIEWED_EVENT",
        "UNREVIEWED_MODEL", "UNREVIEWED_VLM_SUMMARY",
        "UNREVIEWED_PRIMITIVE", "UNREVIEWED_PATIENT_ALIAS",
    ):
        assert forbidden not in wire

    protected = classify_direct_file_access(str(tmp_path), out_path.name)
    assert protected.protected_result is not None
    app = Flask(__name__)
    app.config.update(TESTING=True, UPLOAD_FOLDER=str(tmp_path))
    app.register_blueprint(route.bp)
    unauthorized = app.test_client().get("/api/analysis/result/" + analysis_id)
    assert unauthorized.status_code in (401, 403, 503)



@pytest.mark.parametrize("bad_context", [
    {"persistence_owner": "self", "activity_session_id": "synthetic-session"},
    {"persistence_owner": "parkicheck", "subject_person_id": "synthetic-person"},
    "synthetic-invalid-context",
])
def test_incomplete_patient_linkage_fails_closed(bad_context):
    source = _result()
    source["physio_context"] = bad_context
    with pytest.raises(ValueError, match="authorization"):
        build_finger_research_hold_result(source)
