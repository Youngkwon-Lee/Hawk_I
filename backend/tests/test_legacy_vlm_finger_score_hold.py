"""Synthetic-only legacy score-route guard. No real media or VLM calls."""
from io import BytesIO

from flask import Flask
from werkzeug.datastructures import FileStorage

from routes import vlm


def _app(tmp_path):
    app = Flask(__name__)
    app.config.update(TESTING=True, UPLOAD_FOLDER=str(tmp_path))
    app.register_blueprint(vlm.bp)
    return app


def _deny_network(monkeypatch):
    monkeypatch.setattr(vlm.vlm_scorer, "is_available", lambda: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy scorer was called on held finger result")

    monkeypatch.setattr(vlm.vlm_scorer, "analyze_video", forbidden)
    return forbidden


def test_previous_finger_result_never_scores_or_echoes_ml_score(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    monkeypatch.setattr(
        vlm, "_load_completed_analysis",
        lambda _: ({"video_path": "/synthetic.mp4",
                    "task_type": "finger_tapping", "ml_score": 4}, None, None),
    )
    response = _app(tmp_path).test_client().post("/api/vlm/analyze/synthetic-id")
    data = response.get_json()
    assert response.status_code == 409
    assert data["state"] == "human_review_required"
    assert data["release_status"] == "not_authorized"
    assert data["score"] is None
    assert "ml_score" not in data
    assert "data" not in data


def test_finger_task_aliases_cannot_bypass_review(tmp_path, monkeypatch):
    _deny_network(monkeypatch)
    for task in ("Finger Tapping", "finger-tapping", "finger_tapping_right"):
        monkeypatch.setattr(
            vlm, "_load_completed_analysis",
            lambda _, task=task: ({"video_path": "/synthetic.mp4",
                                  "task_type": task, "ml_score": 0}, None, None),
        )
        response = _app(tmp_path).test_client().post("/api/vlm/analyze/synthetic-id")
        assert response.status_code == 409


def test_direct_upload_defaults_finger_and_does_not_save_or_infer(tmp_path, monkeypatch):
    _deny_network(monkeypatch)

    def forbidden_save(*args, **kwargs):
        raise AssertionError("upload was saved before hold")

    monkeypatch.setattr(FileStorage, "save", forbidden_save)
    response = _app(tmp_path).test_client().post(
        "/api/vlm/analyze-direct",
        data={"video": (BytesIO(b"synthetic-video-only"), "fake.mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 409
    assert response.get_json()["score"] is None
    assert list(tmp_path.iterdir()) == []


def test_nonfinger_named_result_still_uses_existing_research_route(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm.vlm_scorer, "is_available", lambda: True)
    fake = tmp_path / "synthetic.mp4"
    fake.write_bytes(b"synthetic-not-a-real-video")
    monkeypatch.setattr(
        vlm, "_load_completed_analysis",
        lambda _: ({"video_path": str(fake), "task_type": "gait",
                    "ml_score": None}, None, None),
    )
    observed = []
    def stub_analyze(path, task):
        observed.append((path, task))
        return {"success": True, "score": 2, "confidence": 0.1}
    monkeypatch.setattr(vlm.vlm_scorer, "analyze_video", stub_analyze)
    monkeypatch.setattr(vlm.vlm_scorer, "format_result_for_chat", lambda *_: "research-only")
    response = _app(tmp_path).test_client().post("/api/vlm/analyze/synthetic-id")
    assert response.status_code == 200
    assert observed == [(str(fake), "gait")]
    assert response.get_json()["data"]["score"] == 2
