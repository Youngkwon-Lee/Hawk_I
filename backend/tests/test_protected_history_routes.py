"""Protected route smoke tests that do not require live Supabase access."""

from flask import Flask

from routes import history, physio_context, timeline


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
