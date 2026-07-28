"""Tests for Supabase JWT validation and clinician authorization."""

import pytest

from services.supabase_auth import (
    SupabaseClinicianForbidden,
    SupabaseInvalidToken,
    authenticate_clinician,
    extract_bearer_token,
)
from services.supabase_observations import SupabaseObservationConfig


def _config() -> SupabaseObservationConfig:
    return SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="server-key",
        subject_person_id=None,
        organization_id="org-1",
        created_by="creator-1",
        performer_person_id="performer-1",
    )


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_extract_bearer_token_requires_standard_header():
    assert extract_bearer_token("Bearer caller-token") == "caller-token"
    with pytest.raises(SupabaseInvalidToken):
        extract_bearer_token(None)
    with pytest.raises(SupabaseInvalidToken):
        extract_bearer_token("Basic credentials")


def test_authenticate_clinician_uses_caller_jwt_for_rls(monkeypatch):
    calls = []

    def fake_get(url, headers, params=None, timeout=None):
        calls.append({"url": url, "headers": headers, "params": params})
        if url.endswith("/auth/v1/user"):
            return FakeResponse(200, {"id": "auth-user-1"})
        if url.endswith("/rest/v1/persons"):
            assert params["auth_user_id"] == "eq.auth-user-1"
            return FakeResponse(200, [{"id": "person-1"}])
        if url.endswith("/rest/v1/organization_members"):
            return FakeResponse(200, [{
                "organization_id": "org-1",
                "person_id": "person-1",
                "role": "owner",
                "status": "active",
            }])
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("services.supabase_auth.requests.get", fake_get)

    clinician = authenticate_clinician("caller-token", config=_config())

    assert clinician.user_id == "auth-user-1"
    assert clinician.person_id == "person-1"
    assert clinician.organization_id == "org-1"
    assert clinician.role == "owner"
    assert all(call["headers"]["apikey"] == "server-key" for call in calls)
    assert all(
        call["headers"]["Authorization"] == "Bearer caller-token" for call in calls
    )


def test_authenticate_clinician_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(
        "services.supabase_auth.requests.get",
        lambda *args, **kwargs: FakeResponse(401, {"message": "invalid"}),
    )
    with pytest.raises(SupabaseInvalidToken):
        authenticate_clinician("expired-token", config=_config())


def test_authenticate_clinician_rejects_non_clinical_role(monkeypatch):
    def fake_get(url, headers, params=None, timeout=None):
        if url.endswith("/auth/v1/user"):
            return FakeResponse(200, {"id": "auth-user-1"})
        if url.endswith("/rest/v1/persons"):
            return FakeResponse(200, [{"id": "person-1"}])
        if url.endswith("/rest/v1/organization_members"):
            return FakeResponse(200, [{"role": "client"}])
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr("services.supabase_auth.requests.get", fake_get)
    with pytest.raises(SupabaseClinicianForbidden):
        authenticate_clinician("caller-token", config=_config())
