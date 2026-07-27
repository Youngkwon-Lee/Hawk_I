from io import BytesIO


def test_analyze_start_echoes_and_forwards_assessment_session(monkeypatch, tmp_path):
    from app import app
    from routes import analyze

    captured = {}

    class FakeThread:
        daemon = False

        def __init__(self, target, args):
            captured["target"] = target
            captured["args"] = args

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(analyze.threading, "Thread", FakeThread)
    app.config["UPLOAD_FOLDER"] = str(tmp_path)

    response = app.test_client().post(
        "/api/analyze",
        data={
            "video_file": (BytesIO(b"synthetic video"), "finger.mp4"),
            "patient_id": "person-123",
            "assessment_session_id": "assessment-123",
            "test_type": "finger_tapping",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    assert response.get_json()["assessment_session_id"] == "assessment-123"
    assert captured["args"][-1] == "assessment-123"
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
