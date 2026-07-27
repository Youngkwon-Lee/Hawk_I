import json

import pytest

from services.json_store import atomic_write_json


def test_atomic_write_json_persists_result_for_route_after_restart(tmp_path):
    from app import app

    video_id = "restart-safe-analysis"
    payload = {
        "success": True,
        "id": video_id,
        "patient_id": "deidentified-subject",
        "video_type": "finger_tapping",
        "metrics": {"total_taps": 12},
    }
    atomic_write_json(tmp_path / f"{video_id}_result.json", payload)

    previous_upload_folder = app.config["UPLOAD_FOLDER"]
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    try:
        response = app.test_client().get(f"/api/analysis/result/{video_id}")
    finally:
        app.config["UPLOAD_FOLDER"] = previous_upload_folder

    assert response.status_code == 200
    assert response.get_json() == payload
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_json_preserves_previous_result_on_serialization_failure(tmp_path):
    destination = tmp_path / "analysis_result.json"
    original = {"status": "completed", "score": 2.0}
    atomic_write_json(destination, original)

    with pytest.raises(TypeError):
        atomic_write_json(destination, {"not_json": {object()}})

    assert json.loads(destination.read_text(encoding="utf-8")) == original
    assert list(tmp_path.glob(".*.tmp")) == []
