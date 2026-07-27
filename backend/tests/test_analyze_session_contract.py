from io import BytesIO
import json


def test_analyze_start_echoes_and_forwards_assessment_session(monkeypatch, tmp_path):
    from app import app
    from routes import analyze
    from services import analysis_job_store

    captured = {}

    class FakeThread:
        daemon = False

        def __init__(self, target, args):
            captured["target"] = target
            captured["args"] = args

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(analyze.threading, "Thread", FakeThread)
    monkeypatch.setattr(
        analysis_job_store,
        "JOB_FILE",
        tmp_path / "analysis_jobs.json",
    )
    app.config["UPLOAD_FOLDER"] = str(tmp_path)

    response = app.test_client().post(
        "/api/analyze",
        data={
            "video_file": (BytesIO(b"synthetic video"), "finger.mp4"),
            "patient_id": "person-123",
            "assessment_session_id": "assessment-123",
            "test_type": "finger_tapping",
            "medication_context": json.dumps({
                "available": True,
                "medication": "레보도파",
                "dose_mg": 100,
                "taken_at": "2026-07-27T00:00:00Z",
                "assessment_at": "2026-07-27T01:30:00Z",
                "hours_before_assessment": 1.5,
            }),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    assert response.get_json()["assessment_session_id"] == "assessment-123"
    video_id = response.get_json()["id"]
    job = analysis_job_store.get_job(video_id)
    assert captured["target"] is analyze._run_persisted_analysis_job
    assert captured["args"][0] == video_id
    assert job["payload"]["assessment_session_id"] == "assessment-123"
    assert job["payload"]["medication_context"] == {
        "available": True,
        "source": "patient_reported_local",
        "assessment_at": "2026-07-27T01:30:00Z",
        "taken_at": "2026-07-27T00:00:00Z",
        "medication": "레보도파",
        "dose_mg": 100.0,
        "hours_before_assessment": 1.5,
    }
    assert job["status"] == "queued"
    assert captured["started"] is True


def test_analyze_rejects_untrusted_physio_persistence_context(monkeypatch, tmp_path):
    monkeypatch.delenv("HAWKEYE_PHYSIO_CONTEXT_TOKEN", raising=False)

    from app import app

    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    response = app.test_client().post(
        "/api/analyze",
        data={
            "video_file": (BytesIO(b"synthetic video"), "finger.mp4"),
            "physio_subject_person_id": "person-forged",
            "physio_organization_id": "org-forged",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 403
    assert list(tmp_path.iterdir()) == []


def test_analyze_rejects_invalid_medication_context_before_saving_video(tmp_path):
    from app import app

    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    response = app.test_client().post(
        "/api/analyze",
        data={
            "video_file": (BytesIO(b"synthetic video"), "finger.mp4"),
            "medication_context": "{",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []
