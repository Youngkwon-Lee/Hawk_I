"""
Tests for the unified patient timeline read (ParkiCheck + Hawk I observations).
"""

from services.supabase_observations import SupabaseObservationConfig
from services import supabase_timeline


def _config():
    return SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="service-key",
        subject_person_id=None,
        organization_id="org-1",
        created_by="creator-1",
        performer_person_id="performer-1",
        activity_session_id=None,
    )


def _parkicheck_row():
    return {
        "fhir_id": "parkicheck-session-1",
        "code": "UPDRS_3_4",
        "status": "final",
        "source_type": "device",
        "value_integer": 2,
        "value_quantity": None,
        "effective_datetime": "2026-07-28T10:00:00Z",
        "activity_session_id": "session-1",
        "subject_person_id": "person-1",
        "measurement_context": {
            "confidence": "HIGH",
            "medication_context": {"available": True, "medication": "levodopa"},
            "hawk_i": {"analysis_id": "ft_123", "score": 2},
        },
    }


def _hawk_i_row():
    return {
        "fhir_id": "hawkeye-ft_456",
        "code": "UPDRS_3_9",
        "status": "final",
        "source_type": "ai",
        "value_integer": None,
        "value_quantity": 1.5,
        "effective_datetime": "2026-07-27T09:00:00Z",
        "activity_session_id": "session-2",
        "subject_person_id": "person-1",
        "measurement_context": {
            "app_source": "hawk_i",
            "analysis_id": "gait_456",
            "confidence": 0.87,
        },
    }


def test_normalize_parkicheck_row():
    item = supabase_timeline.normalize_observation(_parkicheck_row())
    assert item["app_source"] == "parkicheck"
    assert item["source_type"] == "device"
    assert item["score"] == 2
    assert item["confidence"] == "HIGH"
    assert item["has_medication_context"] is True
    assert item["has_hawk_i_review"] is True
    assert item["observed_at"] == "2026-07-28T10:00:00Z"


def test_normalize_hawk_i_row_uses_quantity_and_app_source():
    item = supabase_timeline.normalize_observation(_hawk_i_row())
    assert item["app_source"] == "hawk_i"
    assert item["score"] == 1.5
    assert item["analysis_id"] == "gait_456"
    assert item["has_medication_context"] is False
    assert item["has_hawk_i_review"] is False


def test_normalize_ai_source_without_app_source_falls_back_to_hawk_i():
    row = _hawk_i_row()
    row["fhir_id"] = "obs-1"
    row["measurement_context"] = {}
    item = supabase_timeline.normalize_observation(row)
    assert item["app_source"] == "hawk_i"


def test_fetch_timeline_returns_none_when_not_configured(monkeypatch):
    monkeypatch.setattr(
        supabase_timeline, "get_supabase_observation_config", lambda: None
    )
    assert supabase_timeline.fetch_timeline("person-1") is None


def test_fetch_timeline_queries_subject_and_normalizes(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [_parkicheck_row(), _hawk_i_row()]

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(supabase_timeline.requests, "get", fake_get)

    items = supabase_timeline.fetch_timeline("person-1", limit=50, config=_config())

    assert captured["url"] == "https://example.supabase.co/rest/v1/observations"
    assert captured["params"]["subject_person_id"] == "eq.person-1"
    assert captured["params"]["limit"] == "50"
    assert captured["headers"]["apikey"] == "service-key"
    assert len(items) == 2
    assert {item["app_source"] for item in items} == {"parkicheck", "hawk_i"}
