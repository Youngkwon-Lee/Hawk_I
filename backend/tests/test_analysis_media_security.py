"""Security and streaming tests for analysis-owned media."""

import json

from flask import Flask

from routes import analyze
from services import analysis_media
from services.supabase_auth import AuthenticatedClinician
from services.supabase_observations import SupabaseObservationConfig


ORG_ID = "11111111-1111-4111-8111-111111111111"
SUBJECT_ID = "22222222-2222-4222-8222-222222222222"


def _config() -> SupabaseObservationConfig:
    return SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="server-key",
        subject_person_id=None,
        organization_id=ORG_ID,
        created_by="33333333-3333-4333-8333-333333333333",
        performer_person_id="44444444-4444-4444-8444-444444444444",
    )


def _clinician() -> AuthenticatedClinician:
    return AuthenticatedClinician(
        user_id="55555555-5555-4555-8555-555555555555",
        person_id="33333333-3333-4333-8333-333333333333",
        organization_id=ORG_ID,
        role="provider",
        access_token="caller-token",
    )


def _app(upload_folder) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = str(upload_folder)
    app.register_blueprint(analyze.bp)
    return app


def _write_result(tmp_path, analysis_id: str, filename: str, private: bool = True):
    context = {"subject_person_id": SUBJECT_ID} if private else None
    payload = {
        "success": True,
        "id": analysis_id,
        "physio_context": context,
        "skeleton_data": {"skeleton_video_url": f"/files/{filename}"},
    }
    (tmp_path / f"{analysis_id}_result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return payload


def _authorize(monkeypatch):
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    monkeypatch.setattr(analyze, "authenticate_clinician", lambda token, config: _clinician())
    monkeypatch.setattr(analyze, "authorize_physio_subject", lambda *args, **kwargs: {})


def test_media_filename_rejects_remote_and_nested_paths():
    remote = {"skeleton_data": {"skeleton_video_url": "https://evil.example/video.mp4"}}
    nested = {"skeleton_data": {"skeleton_video_url": "/files/private/video.mp4"}}
    encoded = {"skeleton_data": {"skeleton_video_url": "/files/%2e%2e%2fvideo.mp4"}}

    assert analysis_media.media_filename(remote, "skeleton_video") is None
    assert analysis_media.media_filename(nested, "skeleton_video") is None
    assert analysis_media.media_filename(encoded, "skeleton_video") is None


def test_result_loader_rejects_unsafe_analysis_id(tmp_path):
    assert analysis_media.load_analysis_result(str(tmp_path), "../private") is None
    assert analysis_media.load_analysis_result(str(tmp_path), "bad..id") is None


def test_patient_media_endpoint_requires_authentication(tmp_path):
    _write_result(tmp_path, "private_123", "private_123_skeleton.mp4")
    (tmp_path / "private_123_skeleton.mp4").write_bytes(b"0123456789")

    response = _app(tmp_path).test_client().get(
        "/api/analysis/media/private_123/skeleton_video"
    )

    assert response.status_code == 401
    assert response.get_json()["error"] == "authentication required"


def test_authorized_media_endpoint_preserves_range_requests(tmp_path, monkeypatch):
    _write_result(tmp_path, "private_123", "private_123_skeleton.mp4")
    (tmp_path / "private_123_skeleton.mp4").write_bytes(b"0123456789")
    _authorize(monkeypatch)

    response = _app(tmp_path).test_client().get(
        "/api/analysis/media/private_123/skeleton_video",
        headers={"Authorization": "Bearer caller-token", "Range": "bytes=2-5"},
    )

    assert response.status_code == 206
    assert response.data == b"2345"
    assert response.headers["Content-Range"] == "bytes 2-5/10"
    assert response.headers["Accept-Ranges"] == "bytes"
    assert response.headers["Cache-Control"] == "no-store, private"


def test_media_endpoint_rejects_unknown_asset(tmp_path):
    _write_result(tmp_path, "private_123", "private_123_skeleton.mp4")
    response = _app(tmp_path).test_client().get(
        "/api/analysis/media/private_123/not-a-real-asset"
    )
    assert response.status_code == 400


def test_access_sidecar_protects_upload_before_result_exists(tmp_path):
    analysis_media.write_analysis_access_record(
        str(tmp_path),
        "private_123",
        {"subject_person_id": SUBJECT_ID, "organization_id": ORG_ID},
    )
    decision = analysis_media.classify_direct_file_access(
        str(tmp_path),
        "private_123_original.mp4",
    )

    assert decision.internal is False
    assert decision.protected_result is not None
    assert decision.protected_result["physio_context"]["subject_person_id"] == SUBJECT_ID


def test_access_sidecar_is_never_a_web_resource(tmp_path):
    analysis_media.write_analysis_access_record(
        str(tmp_path),
        "private_123",
        {"subject_person_id": SUBJECT_ID},
    )
    decision = analysis_media.classify_direct_file_access(
        str(tmp_path),
        "private_123_access.json",
    )
    assert decision.internal is True


def test_legacy_result_scan_protects_existing_patient_media(tmp_path):
    _write_result(tmp_path, "legacy_123", "legacy-skeleton.mp4")
    decision = analysis_media.classify_direct_file_access(
        str(tmp_path),
        "legacy-skeleton.mp4",
    )
    assert decision.protected_result is not None


def test_public_unlinked_media_remains_public(tmp_path):
    _write_result(tmp_path, "public_123", "public_123_skeleton.mp4", private=False)
    decision = analysis_media.classify_direct_file_access(
        str(tmp_path),
        "public_123_skeleton.mp4",
    )
    assert decision.internal is False
    assert decision.protected_result is None


def test_legacy_parkicheck_result_is_treated_as_patient_linked(tmp_path):
    analysis_id = "parkicheck_123"
    filename = f"{analysis_id}_skeleton.mp4"
    payload = {
        "success": True,
        "id": analysis_id,
        "physio_context": {
            "activity_session_id": "66666666-6666-4666-8666-666666666666",
            "persistence_owner": "parkicheck",
        },
        "skeleton_data": {"skeleton_video_url": f"/files/{filename}"},
    }
    (tmp_path / f"{analysis_id}_result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    decision = analysis_media.classify_direct_file_access(str(tmp_path), filename)
    assert decision.protected_result is not None


def test_patient_linked_progress_requires_authentication(tmp_path, monkeypatch):
    from app import app as full_app

    analysis_media.write_analysis_access_record(
        str(tmp_path),
        "private_123",
        {"subject_person_id": SUBJECT_ID, "organization_id": ORG_ID},
    )
    monkeypatch.setitem(full_app.config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setitem(full_app.config, "TESTING", True)

    response = full_app.test_client().get("/api/analysis/progress/private_123")
    assert response.status_code == 401


def test_direct_file_routes_cannot_bypass_patient_media_auth(tmp_path, monkeypatch):
    from app import app as full_app

    payload = _write_result(tmp_path, "private_123", "private_123_skeleton.mp4")
    (tmp_path / "private_123_skeleton.mp4").write_bytes(b"0123456789")
    analysis_media.write_analysis_access_record(
        str(tmp_path),
        "private_123",
        payload["physio_context"],
    )
    monkeypatch.setitem(full_app.config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setitem(full_app.config, "TESTING", True)

    client = full_app.test_client()
    assert client.get("/files/private_123_skeleton.mp4").status_code == 401
    assert client.get("/uploads/private_123_skeleton.mp4").status_code == 401
    assert client.get("/files/private_123_result.json").status_code == 401
    assert client.get("/files/private_123_access.json").status_code == 404


def test_direct_authorized_file_route_preserves_range(tmp_path, monkeypatch):
    from app import app as full_app

    payload = _write_result(tmp_path, "private_123", "private_123_skeleton.mp4")
    (tmp_path / "private_123_skeleton.mp4").write_bytes(b"0123456789")
    analysis_media.write_analysis_access_record(
        str(tmp_path),
        "private_123",
        payload["physio_context"],
    )
    _authorize(monkeypatch)
    monkeypatch.setitem(full_app.config, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setitem(full_app.config, "TESTING", True)

    response = full_app.test_client().get(
        "/files/private_123_skeleton.mp4",
        headers={"Authorization": "Bearer caller-token", "Range": "bytes=4-7"},
    )

    assert response.status_code == 206
    assert response.data == b"4567"
    assert response.headers["Content-Range"] == "bytes 4-7/10"
