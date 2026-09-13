"""Protected route smoke tests that do not require live Supabase access."""

from flask import Flask

from routes import history, physio_context, timeline
from services.supabase_auth import AuthenticatedPerson


def _app() -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(history.bp)
    app.register_blueprint(physio_context.bp)
    app.register_blueprint(timeline.bp)
    return app


def test_history_routes_require_bearer_authentication():
    client = _app().test_client()

    assert client.get("/api/history/").status_code == 401
    assert client.get("/api/history").status_code == 401
    assert client.get("/api/history/stats").status_code == 401
    assert client.get("/api/history/timeline?subject_person_id=person-1").status_code == 401
    assert client.get("/api/history/analysis-1").status_code == 401
    assert client.delete("/api/history/analysis-1").status_code == 401


def test_physio_subjects_require_bearer_authentication():
    client = _app().test_client()
    assert client.get("/api/physio/subjects").status_code == 401
    assert client.get("/api/physio/self").status_code == 401


def test_physio_self_returns_only_authenticated_person(monkeypatch):
    monkeypatch.setattr(
        "services.supabase_auth.authenticate_person",
        lambda access_token: AuthenticatedPerson(
            user_id="auth-user-1", person_id="person-1", access_token=access_token
        ),
    )
    monkeypatch.setattr(
        physio_context,
        "load_physio_self_context",
        lambda person: {
            "subject": {
                "id": person.person_id,
                "display_name": "내 기록",
                "organization_id": "org-personal",
                "is_default": True,
            },
            "organization": {"id": "org-personal", "display_name": "개인 기록"},
        },
    )

    response = _app().test_client().get(
        "/api/physio/self", headers={"Authorization": "Bearer caller-token"}
    )

    assert response.status_code == 200
    assert response.get_json()["subject"]["id"] == "person-1"
    assert response.get_json()["subject"]["display_name"] == "내 기록"
    assert response.get_json()["organization"]["id"] == "org-personal"
    assert response.get_json()["contract_version"] == "hawkeye-self/v1"
    assert response.get_json()["persistence_owner"] == "self"


def test_legacy_simulated_medication_timeline_is_retired():
    response = _app().test_client().get("/api/timeline/patient-1")

    assert response.status_code == 410
    assert response.get_json() == {
        "success": False,
        "error": "The simulated medication timeline has been removed.",
        "replacement": "/api/history/timeline",
        "requires_authentication": True,
        "patient_id": "patient-1",
    }
