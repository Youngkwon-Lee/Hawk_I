def test_subject_listing_fails_closed_without_operator_token(monkeypatch):
    monkeypatch.delenv("HAWKEYE_PHYSIO_CONTEXT_TOKEN", raising=False)

    from app import app

    response = app.test_client().get("/api/physio/subjects")

    assert response.status_code == 403
    assert response.get_json()["success"] is False


def test_subject_listing_requires_matching_bearer_token(monkeypatch):
    monkeypatch.setenv("HAWKEYE_PHYSIO_CONTEXT_TOKEN", "operator-secret")

    from app import app

    response = app.test_client().get(
        "/api/physio/subjects",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert response.status_code == 403


def test_subject_listing_allows_configured_operator_token(monkeypatch):
    monkeypatch.setenv("HAWKEYE_PHYSIO_CONTEXT_TOKEN", "operator-secret")

    from app import app
    from routes import physio_context

    expected = {
        "success": True,
        "enabled": True,
        "organization": None,
        "subjects": [],
    }
    monkeypatch.setattr(
        physio_context,
        "load_physio_subject_context",
        lambda limit: expected,
    )

    response = app.test_client().get(
        "/api/physio/subjects?limit=10",
        headers={"Authorization": "Bearer operator-secret"},
    )

    assert response.status_code == 200
    assert response.get_json() == expected
