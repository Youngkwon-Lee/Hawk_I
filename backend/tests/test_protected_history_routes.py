"""Protected route smoke tests that do not require live Supabase access."""

from flask import Flask

from routes import history, physio_context


def _app() -> Flask:
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(history.bp)
    app.register_blueprint(physio_context.bp)
    return app


def test_history_routes_require_bearer_authentication():
    client = _app().test_client()

    assert client.get("/api/history/").status_code == 401
    assert client.get("/api/history/stats").status_code == 401
    assert client.get("/api/history/timeline?subject_person_id=person-1").status_code == 401
    assert client.get("/api/history/analysis-1").status_code == 401
    assert client.delete("/api/history/analysis-1").status_code == 401


def test_physio_subjects_require_bearer_authentication():
    client = _app().test_client()
    assert client.get("/api/physio/subjects").status_code == 401
