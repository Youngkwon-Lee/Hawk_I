"""Synthetic pre-LLM hold regressions; no patient video or paid inference."""
import json
from types import SimpleNamespace

from domain.context import AnalysisContext
from agents.orchestrator import OrchestratorAgent
from services.analysis_media import _is_patient_linked


def forbidden(message):
    def raise_error(*args, **kwargs):
        raise AssertionError(message)
    return raise_error


def fake_orchestrator(task, *, allow_report=False):
    obj = object.__new__(OrchestratorAgent)

    def vision(ctx):
        ctx.task_type = task
        ctx.vision_meta = {"confidence": 1.0, "frame_count": 400}
        ctx.skeleton_data = {"landmarks": [{"frame_number": 0, "timestamp": 0.0}]}
        ctx.status = "vision_done"
        return ctx

    def clinical(ctx):
        ctx.status = "clinical_done"
        ctx.kinematic_metrics = {"tapping_speed": 2.0}
        ctx.clinical_scores = {"total_score": 0.0}
        return ctx

    def validate(ctx):
        ctx.validation_result = {"is_valid": True}
        return ctx

    report_calls = []

    def report(ctx):
        report_calls.append(task)
        ctx.status = "report_done"
        return ctx

    obj.vision_agent = SimpleNamespace(process=vision)
    obj.clinical_agent = SimpleNamespace(process=clinical)
    obj.gait_cycle_agent = SimpleNamespace(process=lambda ctx: ctx)
    obj.validation_agent = SimpleNamespace(process=validate)
    obj.report_agent = SimpleNamespace(
        process=report if allow_report else forbidden("ReportAgent called for held Finger Tapping")
    )
    return obj, report_calls


def test_orchestrator_never_sends_finger_result_to_interpreter(monkeypatch):
    import agents.orchestrator as orchestration
    monkeypatch.setattr(orchestration.time, "sleep", lambda *_: None)
    agent, calls = fake_orchestrator("finger_tapping")
    steps = []
    result = agent.process(
        AnalysisContext(video_path="synthetic-not-real.mp4"),
        lambda name, status: steps.append((name, status)),
    )
    assert result.error is None
    assert result.report is None
    assert calls == []
    assert ("ai_interpretation", "skipped") in steps
    assert ("updrs_calculation", "skipped") in steps
    assert ("ai_interpretation", "completed") not in steps


def test_gait_report_call_remains_available(monkeypatch):
    import agents.orchestrator as orchestration
    monkeypatch.setattr(orchestration.time, "sleep", lambda *_: None)
    agent, calls = fake_orchestrator("gait", allow_report=True)
    steps = []
    result = agent.process(
        AnalysisContext(video_path="synthetic-gait.mp4"),
        lambda name, status: steps.append((name, status)),
    )
    assert result.error is None
    assert calls == ["gait"]
    assert ("ai_interpretation", "completed") in steps


def test_background_finger_does_not_call_finetuned_vlm(tmp_path, monkeypatch):
    from routes import analyze as route
    from services import finetuned_vlm as ft

    fake_video = tmp_path / "synthetic.mp4"
    fake_video.write_bytes(b"synthetic-not-real-video")
    progress = []

    class FakeOrchestrator:
        def process(self, ctx, on_progress_update=None):
            ctx.status = "clinical_done"
            ctx.task_type = "finger_tapping"
            ctx.skeleton_data = {"landmarks": [
                {"frame_number": 0, "timestamp": 0.0, "landmarks": [
                    {"id": 4, "x": .5, "y": .5, "z": .0},
                ]}
            ]}
            ctx.kinematic_metrics = {"tapping_speed": 2.0}
            ctx.clinical_scores = {"total_score": 0.0}
            ctx.vision_meta = {
                "fps": 80, "frame_count": 400, "frame_skip": 3,
                "roi": (0, 0, 64, 64), "confidence": 1.0,
            }
            ctx.log("clinical", "UPDRS Estimation: Score 0",
                    meta={"total_score": 0})
            return ctx

    monkeypatch.setattr(route, "OrchestratorAgent", FakeOrchestrator)
    monkeypatch.setattr(route, "update_step",
                        lambda *args, **kwargs: progress.append((args[1], args[2])))
    monkeypatch.setattr(route, "complete_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(route, "generate_visualization_data", lambda **kwargs: {})
    monkeypatch.setattr(route, "detect_events", lambda **kwargs: [])
    monkeypatch.setattr(route, "get_finger_performability_gate",
                        lambda: SimpleNamespace(
                            assess=lambda *args, **kwargs: SimpleNamespace(status="uncertain")))
    monkeypatch.setattr(ft, "frames_to_base64",
                        forbidden("Video frame extraction before authorization"))
    monkeypatch.setattr(ft, "analyze",
                        forbidden("External model inference before authorization"))
    monkeypatch.setattr(route, "persist_analysis_observation",
                        forbidden("Scored DB persistence on a held result"))

    route.process_video_background(
        str(fake_video), "synthetic-finger-hold", None, "finger_tapping",
        {"UPLOAD_FOLDER": str(tmp_path)},
        physio_context={
            "persistence_owner": "parkicheck",
            "activity_session_id": "33333333-3333-4333-8333-333333333333",
        },
    )
    saved = json.loads((tmp_path / "synthetic-finger-hold_result.json").read_text())
    assert saved["updrs_score"] is None
    assert saved["ai_interpretation"] is None
    assert saved["reasoning_log"] == []
    assert saved["research_review"]["state"] == "hold"
    assert _is_patient_linked(saved) is True
    assert ("updrs_calculation", "skipped") in progress
    assert ("ai_interpretation", "skipped") in progress
    assert ("updrs_calculation", "completed") not in progress
    assert ("ai_interpretation", "completed") not in progress
