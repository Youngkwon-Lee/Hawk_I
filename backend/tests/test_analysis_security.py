"""Security tests for patient-linked analysis creation and result access."""

from io import BytesIO
import json

from flask import Flask

from routes import analyze
from services.supabase_auth import (
    AuthenticatedClinician,
    AuthenticatedPerson,
    SupabaseClinicianForbidden,
)
from services.supabase_observations import SupabaseObservationConfig
from services.supabase_observations import SupabaseObservationResult


ORG_ID = "11111111-1111-4111-8111-111111111111"
SUBJECT_ID = "22222222-2222-4222-8222-222222222222"
CLINICIAN_ID = "33333333-3333-4333-8333-333333333333"
PERFORMER_ID = "44444444-4444-4444-8444-444444444444"


def _config() -> SupabaseObservationConfig:
    return SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="server-key",
        subject_person_id=None,
        organization_id=ORG_ID,
        created_by=CLINICIAN_ID,
        performer_person_id=PERFORMER_ID,
    )


def _clinician() -> AuthenticatedClinician:
    return AuthenticatedClinician(
        user_id="55555555-5555-4555-8555-555555555555",
        person_id=CLINICIAN_ID,
        organization_id=ORG_ID,
        role="provider",
        access_token="caller-token",
    )


def _person() -> AuthenticatedPerson:
    return AuthenticatedPerson(
        user_id="55555555-5555-4555-8555-555555555555",
        person_id=SUBJECT_ID,
        access_token="caller-token",
    )


def _app(upload_folder) -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = str(upload_folder)
    app.register_blueprint(analyze.bp)
    return app


def _video_form(**fields):
    return {
        "video_file": (BytesIO(b"synthetic-video"), "synthetic.mp4"),
        **fields,
    }


def test_patient_linked_analysis_requires_auth_before_file_save(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)

    response = _app(tmp_path).test_client().post(
        "/api/analyze",
        data=_video_form(
            physio_subject_person_id=SUBJECT_ID,
            physio_organization_id=ORG_ID,
        ),
        content_type="multipart/form-data",
    )

    assert response.status_code == 401
    assert response.get_json()["error"] == "authentication required"
    assert list(tmp_path.iterdir()) == []


def test_patient_linked_analysis_without_auth_fails_closed_when_config_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(analyze, "get_supabase_observation_config", lambda: None)

    response = _app(tmp_path).test_client().post(
        "/api/analyze",
        data=_video_form(
            physio_subject_person_id=SUBJECT_ID,
            physio_organization_id=ORG_ID,
        ),
        content_type="multipart/form-data",
    )

    assert response.status_code == 401
    assert list(tmp_path.iterdir()) == []


def test_patient_linked_analysis_uses_server_canonical_context(tmp_path, monkeypatch):
    captured = {}

    class FakeThread:
        daemon = False

        def __init__(self, target, args):
            captured["target"] = target
            captured["args"] = args

        def start(self):
            captured["started"] = True

    canonical_context = {
        "subject_person_id": SUBJECT_ID,
        "organization_id": ORG_ID,
        "created_by_person_id": CLINICIAN_ID,
        "performer_person_id": PERFORMER_ID,
        "subject_display_name": "Synthetic Patient",
        "organization_display_name": "Synthetic Clinic",
    }
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    monkeypatch.setattr(analyze, "authenticate_clinician", lambda token, config: _clinician())
    monkeypatch.setattr(
        analyze,
        "authorize_physio_subject",
        lambda clinician, subject_id, activity_session_id, config: canonical_context.copy(),
    )
    monkeypatch.setattr(analyze.threading, "Thread", FakeThread)
    monkeypatch.setattr(analyze, "init_analysis", lambda *args, **kwargs: None)

    response = _app(tmp_path).test_client().post(
        "/api/analyze",
        data=_video_form(
            patient_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            physio_subject_person_id=SUBJECT_ID,
            physio_organization_id=ORG_ID,
            physio_created_by_person_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            physio_performer_person_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            physio_subject_display_name="Forged Name",
            physio_organization_display_name="Forged Clinic",
        ),
        headers={"Authorization": "Bearer caller-token"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    assert captured["started"] is True
    assert captured["args"][2] == SUBJECT_ID
    assert captured["args"][7] == canonical_context
    assert "Forged Name" not in captured["args"][7].values()


def test_patient_linked_analysis_rejects_other_organization(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    response = _app(tmp_path).test_client().post(
        "/api/analyze",
        data=_video_form(
            physio_subject_person_id=SUBJECT_ID,
            physio_organization_id="99999999-9999-4999-8999-999999999999",
        ),
        headers={"Authorization": "Bearer caller-token"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 403
    assert list(tmp_path.iterdir()) == []


def test_public_unlinked_analysis_remains_available(tmp_path, monkeypatch):
    class FakeThread:
        daemon = False

        def __init__(self, target, args):
            self.args = args

        def start(self):
            return None

    monkeypatch.setattr(analyze.threading, "Thread", FakeThread)
    monkeypatch.setattr(analyze, "init_analysis", lambda *args, **kwargs: None)

    response = _app(tmp_path).test_client().post(
        "/api/analyze",
        data=_video_form(),
        content_type="multipart/form-data",
    )

    assert response.status_code == 202


def test_parkicheck_context_requires_auth_before_file_save(tmp_path, monkeypatch):
    session_id = "66666666-6666-4666-8666-666666666666"
    response = _app(tmp_path).test_client().post(
        "/api/analyze",
        data=_video_form(
            assessment_session_id=session_id,
            physio_subject_person_id=SUBJECT_ID,
            physio_organization_id=ORG_ID,
            physio_contract_version="parkicheck-hawk-i/v1",
            physio_persistence_owner="parkicheck",
        ),
        content_type="multipart/form-data",
    )

    assert response.status_code == 401
    assert list(tmp_path.iterdir()) == []


def test_parkicheck_context_uses_authorized_activity_session(tmp_path, monkeypatch):
    captured = {}

    class FakeThread:
        daemon = False

        def __init__(self, target, args):
            captured["args"] = args

        def start(self):
            return None

    session_id = "66666666-6666-4666-8666-666666666666"
    canonical_context = {
        "subject_person_id": SUBJECT_ID,
        "organization_id": ORG_ID,
        "created_by_person_id": SUBJECT_ID,
        "performer_person_id": SUBJECT_ID,
        "activity_session_id": session_id,
    }
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    monkeypatch.setattr(
        analyze,
        "authorize_parkicheck_session",
        lambda *args, **kwargs: canonical_context.copy(),
    )
    monkeypatch.setattr(analyze.threading, "Thread", FakeThread)
    monkeypatch.setattr(analyze, "init_analysis", lambda *args, **kwargs: None)

    response = _app(tmp_path).test_client().post(
        "/api/analyze",
        data=_video_form(
            assessment_session_id=session_id,
            physio_subject_person_id=SUBJECT_ID,
            physio_organization_id=ORG_ID,
            physio_contract_version="parkicheck-hawk-i/v1",
            physio_persistence_owner="parkicheck",
        ),
        headers={"Authorization": "Bearer caller-token"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    assert captured["args"][7] == {
        **canonical_context,
        "contract_version": "parkicheck-hawk-i/v1",
        "persistence_owner": "parkicheck",
    }


def test_self_context_uses_authenticated_person_and_canonical_context(tmp_path, monkeypatch):
    captured = {}

    class FakeThread:
        daemon = False

        def __init__(self, target, args):
            captured["args"] = args

        def start(self):
            return None

    canonical_context = {
        "subject_person_id": SUBJECT_ID,
        "organization_id": ORG_ID,
        "created_by_person_id": SUBJECT_ID,
        "performer_person_id": SUBJECT_ID,
        "subject_display_name": "Self Tester",
        "organization_display_name": "Personal Workspace",
    }
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    monkeypatch.setattr(analyze, "authenticate_person", lambda token, config: _person())
    monkeypatch.setattr(
        analyze,
        "authorize_physio_self",
        lambda person, subject_id, organization_id, config: canonical_context.copy(),
    )
    monkeypatch.setattr(analyze.threading, "Thread", FakeThread)
    monkeypatch.setattr(analyze, "init_analysis", lambda *args, **kwargs: None)

    response = _app(tmp_path).test_client().post(
        "/api/analyze",
        data=_video_form(
            physio_subject_person_id=SUBJECT_ID,
            physio_organization_id=ORG_ID,
            physio_contract_version="hawkeye-self/v1",
            physio_persistence_owner="self",
        ),
        headers={"Authorization": "Bearer caller-token"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    assert captured["args"][7] == {
        **canonical_context,
        "contract_version": "hawkeye-self/v1",
        "persistence_owner": "self",
    }


def test_self_context_rejects_other_subject_before_file_save(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    monkeypatch.setattr(analyze, "authenticate_person", lambda token, config: _person())
    monkeypatch.setattr(
        analyze,
        "authorize_physio_self",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SupabaseClinicianForbidden("self subject access denied")
        ),
    )

    response = _app(tmp_path).test_client().post(
        "/api/analyze",
        data=_video_form(
            physio_subject_person_id="77777777-7777-4777-8777-777777777777",
            physio_organization_id=ORG_ID,
            physio_contract_version="hawkeye-self/v1",
            physio_persistence_owner="self",
        ),
        headers={"Authorization": "Bearer caller-token"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 403
    assert list(tmp_path.iterdir()) == []


def _write_result(tmp_path, analysis_id: str, physio_context=None):
    (tmp_path / f"{analysis_id}_result.json").write_text(
        json.dumps({"success": True, "id": analysis_id, "physio_context": physio_context}),
        encoding="utf-8",
    )


def test_unlinked_result_remains_public_and_disables_cache(tmp_path):
    _write_result(tmp_path, "public_123")
    response = _app(tmp_path).test_client().get("/api/analysis/result/public_123")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, private"


def test_patient_linked_result_requires_authentication(tmp_path):
    _write_result(tmp_path, "private_123", {"subject_person_id": SUBJECT_ID})
    response = _app(tmp_path).test_client().get("/api/analysis/result/private_123")

    assert response.status_code == 401
    assert response.get_json()["error"] == "authentication required"


def test_patient_linked_result_checks_subject_access(tmp_path, monkeypatch):
    _write_result(tmp_path, "private_123", {"subject_person_id": SUBJECT_ID})
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    monkeypatch.setattr(analyze, "authenticate_clinician", lambda token, config: _clinician())
    monkeypatch.setattr(
        analyze,
        "authorize_physio_subject",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            SupabaseClinicianForbidden("subject access denied")
        ),
    )

    response = _app(tmp_path).test_client().get(
        "/api/analysis/result/private_123",
        headers={"Authorization": "Bearer caller-token"},
    )

    assert response.status_code == 403
    assert response.get_json()["error"] == "result access denied"


def test_patient_linked_result_allows_authorized_clinician(tmp_path, monkeypatch):
    _write_result(tmp_path, "private_123", {"subject_person_id": SUBJECT_ID})
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    monkeypatch.setattr(analyze, "authenticate_clinician", lambda token, config: _clinician())
    monkeypatch.setattr(analyze, "authorize_physio_subject", lambda *args, **kwargs: {})

    response = _app(tmp_path).test_client().get(
        "/api/analysis/result/private_123",
        headers={"Authorization": "Bearer caller-token"},
    )

    assert response.status_code == 200


def test_self_result_allows_only_authenticated_person(tmp_path, monkeypatch):
    _write_result(tmp_path, "self_123", {
        "subject_person_id": SUBJECT_ID,
        "organization_id": ORG_ID,
        "persistence_owner": "self",
    })
    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    monkeypatch.setattr(analyze, "authenticate_person", lambda token, config: _person())
    monkeypatch.setattr(analyze, "authorize_physio_self", lambda *args, **kwargs: {})

    response = _app(tmp_path).test_client().get(
        "/api/analysis/result/self_123",
        headers={"Authorization": "Bearer caller-token"},
    )

    assert response.status_code == 200


def test_legacy_parkicheck_result_requires_session_authorization(tmp_path, monkeypatch):
    session_id = "66666666-6666-4666-8666-666666666666"
    _write_result(tmp_path, "parkicheck_123", {
        "activity_session_id": session_id,
        "persistence_owner": "parkicheck",
    })

    unauthenticated = _app(tmp_path).test_client().get(
        "/api/analysis/result/parkicheck_123"
    )
    assert unauthenticated.status_code == 401

    monkeypatch.setattr(analyze, "get_supabase_observation_config", _config)
    monkeypatch.setattr(analyze, "authorize_parkicheck_session", lambda *args, **kwargs: {})
    authorized = _app(tmp_path).test_client().get(
        "/api/analysis/result/parkicheck_123",
        headers={"Authorization": "Bearer caller-token"},
    )
    assert authorized.status_code == 200


def test_result_rejects_unsafe_analysis_id(tmp_path):
    response = _app(tmp_path).test_client().get("/api/analysis/result/bad..id")
    assert response.status_code == 400


def test_parkicheck_trace_uses_shared_session_and_deterministic_fhir_id():
    session_id = "66666666-6666-4666-8666-666666666666"
    trace = analyze.build_analysis_trace(
        "analysis-123",
        {"assessment_session_id": session_id},
        SupabaseObservationResult(
            enabled=True,
            saved=False,
            activity_session_id=session_id,
            persistence_owner="parkicheck",
            delegated=True,
        ),
    )

    assert trace == {
        "analysis_id": "analysis-123",
        "activity_session_id": session_id,
        "observation_fhir_id": f"parkicheck-{session_id}",
        "persistence_owner": "parkicheck",
    }


def test_hawk_i_trace_includes_persisted_observation_id():
    trace = analyze.build_analysis_trace(
        "analysis-456",
        {},
        SupabaseObservationResult(
            enabled=True,
            saved=True,
            observation_id="observation-456",
            activity_session_id="session-456",
        ),
    )

    assert trace["analysis_id"] == "analysis-456"
    assert trace["activity_session_id"] == "session-456"
    assert trace["observation_id"] == "observation-456"
    assert trace["observation_fhir_id"] == "hawkeye-analysis-456"
