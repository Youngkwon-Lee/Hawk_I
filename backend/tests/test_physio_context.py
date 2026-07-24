"""
Tests for server-side physio_app context lookup.
"""


def test_load_physio_subject_context_returns_active_clients(monkeypatch):
    from services import physio_context

    monkeypatch.setenv("HAWKEYE_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("HAWKEYE_SUPABASE_SERVICE_KEY", "server-secret")
    monkeypatch.delenv("HAWKEYE_SUPABASE_SUBJECT_PERSON_ID", raising=False)
    monkeypatch.setenv("HAWKEYE_SUPABASE_ORGANIZATION_ID", "org-1")
    monkeypatch.setenv("HAWKEYE_SUPABASE_CREATED_BY_PERSON_ID", "person-default")
    monkeypatch.setenv("HAWKEYE_SUPABASE_PERFORMER_PERSON_ID", "person-default")

    class FakeResponse:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    calls = []

    def fake_get(url, headers, params, timeout):
        calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        if url.endswith("/organizations"):
            return FakeResponse([{
                "id": "org-1",
                "display_name": "Test Clinic",
                "status": "active",
            }])
        if url.endswith("/org_clients"):
            return FakeResponse([
                {"person_id": "person-2", "status": "active"},
            ])
        if url.endswith("/persons"):
            return FakeResponse([
                {
                    "id": "person-2",
                    "display_name": "Client Two",
                    "email": None,
                    "user_type": "client",
                    "source_type": "manual",
                },
            ])
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(physio_context.requests, "get", fake_get)

    result = physio_context.load_physio_subject_context()

    assert result["success"] is True
    assert result["enabled"] is True
    assert result["organization"]["display_name"] == "Test Clinic"
    assert result["default_subject_id"] == "person-2"
    assert [subject["id"] for subject in result["subjects"]] == ["person-2"]
    assert result["subjects"][0]["is_default"] is True
    assert result["subjects"][0]["role"] == "client"
    assert calls[0]["headers"]["Authorization"] == "Bearer server-secret"


def test_load_physio_subject_context_returns_no_subjects_without_active_clients(monkeypatch):
    from services import physio_context

    monkeypatch.setenv("HAWKEYE_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("HAWKEYE_SUPABASE_SERVICE_KEY", "server-secret")
    monkeypatch.setenv("HAWKEYE_SUPABASE_SUBJECT_PERSON_ID", "provider-person")
    monkeypatch.setenv("HAWKEYE_SUPABASE_ORGANIZATION_ID", "org-1")
    monkeypatch.setenv("HAWKEYE_SUPABASE_CREATED_BY_PERSON_ID", "provider-person")
    monkeypatch.setenv("HAWKEYE_SUPABASE_PERFORMER_PERSON_ID", "provider-person")

    class FakeResponse:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_get(url, headers, params, timeout):
        if url.endswith("/organizations"):
            return FakeResponse([{"id": "org-1", "display_name": "Test Clinic"}])
        if url.endswith("/org_clients"):
            return FakeResponse([])
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(physio_context.requests, "get", fake_get)

    result = physio_context.load_physio_subject_context()

    assert result["success"] is True
    assert result["enabled"] is True
    assert result["subjects"] == []
    assert result["default_subject_id"] is None
    assert "no active physio_app clients" in result["reason"]


def test_load_physio_subject_context_disabled_without_env(monkeypatch):
    from services.physio_context import load_physio_subject_context

    for name in [
        "HAWKEYE_SUPABASE_URL",
        "SUPABASE_URL",
        "HAWKEYE_SUPABASE_SERVICE_KEY",
        "SUPABASE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "HAWKEYE_SUPABASE_SUBJECT_PERSON_ID",
        "HAWKEYE_SUPABASE_PERSON_ID",
        "HAWKEYE_SUPABASE_ORGANIZATION_ID",
        "HAWKEYE_SUPABASE_ORG_ID",
    ]:
        monkeypatch.delenv(name, raising=False)

    result = load_physio_subject_context()

    assert result["success"] is True
    assert result["enabled"] is False
    assert result["subjects"] == []
