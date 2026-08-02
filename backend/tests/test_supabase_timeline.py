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
        "id": "observation-1",
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


def _medication_row():
    return {
        "fhir_id": "parkicheck-medication-med-1",
        "status": "completed",
        "medication_code": "LEVODOPA",
        "medication_display": "레보도파",
        "effective_start": "2026-07-28T08:00:00Z",
        "date_asserted": "2026-07-28T08:01:00Z",
        "dosage": {
            "dose_mg": 125,
            "unit": "mg",
            "app_source": "parkicheck",
        },
        "information_source_type": "patient",
        "subject_person_id": "person-1",
    }


def test_normalize_parkicheck_row():
    item = supabase_timeline.normalize_observation(_parkicheck_row())
    assert item["app_source"] == "parkicheck"
    assert item["source_type"] == "device"
    assert item["score"] == 2
    assert item["confidence"] == "HIGH"
    assert item["has_medication_context"] is True
    assert item["medication_name"] == "levodopa"
    assert item["medication_dose_mg"] is None
    assert item["has_hawk_i_review"] is True
    assert item["analysis_id"] == "ft_123"
    assert item["observation_id"] == "observation-1"
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
    assert supabase_timeline.fetch_timeline("person-1", "caller-token") is None


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

    items = supabase_timeline.fetch_timeline(
        "person-1",
        "caller-token",
        limit=50,
        config=_config(),
    )

    assert captured["url"] == "https://example.supabase.co/rest/v1/observations"
    assert captured["params"]["subject_person_id"] == "eq.person-1"
    assert captured["params"]["organization_id"] == "eq.org-1"
    assert captured["params"]["limit"] == "50"
    assert captured["headers"]["apikey"] == "service-key"
    assert captured["headers"]["Authorization"] == "Bearer caller-token"
    assert len(items) == 2
    assert {item["app_source"] for item in items} == {"parkicheck", "hawk_i"}


def test_normalize_medication_statement_uses_patient_reported_dose():
    item = supabase_timeline.normalize_medication_statement(_medication_row())
    assert item == {
        "event_id": "parkicheck-medication-med-1",
        "observed_at": "2026-07-28T08:00:00Z",
        "status": "completed",
        "event_kind": "taken",
        "medication_code": "LEVODOPA",
        "medication_display": "레보도파",
        "dose_mg": 125,
        "dose_unit": "mg",
        "information_source_type": "patient",
        "subject_person_id": "person-1",
        "app_source": "parkicheck",
    }


def test_fetch_medication_statements_queries_subject_and_normalizes(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [_medication_row()]

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(supabase_timeline.requests, "get", fake_get)
    items = supabase_timeline.fetch_medication_statements(
        "person-1", "caller-token", limit=20, config=_config()
    )

    assert captured["url"].endswith("/rest/v1/medication_statements")
    assert captured["params"]["subject_person_id"] == "eq.person-1"
    assert captured["params"]["organization_id"] == "eq.org-1"
    assert captured["headers"]["Authorization"] == "Bearer caller-token"
    assert items[0]["medication_display"] == "레보도파"


def _rich_hawk_i_row():
    row = _hawk_i_row()
    row["measurement_context"].update(
        {
            "severity": "Mild",
            "score_confidence": 0.7,
            "scoring_method": "coral",
            "ml_model_type": "rf",
            "metrics": {"gait_speed": 0.82, "stride_length": 0.94},
            "ai_interpretation": {
                "summary": "보폭 감소와 팔 흔들림 저하가 관찰됨",
                "explanation": "긴 설명",
            },
            "score_advisory": {"level": "review_recommended", "summary": "조명 불량"},
            "performability_assessment": {"status": "analyzable"},
        }
    )
    return row


def test_normalize_exposes_quantitative_and_qualitative_evidence():
    item = supabase_timeline.normalize_observation(_rich_hawk_i_row())

    assert item["metrics"] == {"gait_speed": 0.82, "stride_length": 0.94}
    assert item["rationale"] == "보폭 감소와 팔 흔들림 저하가 관찰됨"
    assert item["severity"] == "Mild"
    assert item["score_confidence"] == 0.7
    assert item["score_advisory_level"] == "review_recommended"
    assert item["score_advisory_summary"] == "조명 불량"
    assert item["performability_status"] == "analyzable"
    assert item["scoring_method"] == "coral"
    assert item["model_type"] == "rf"


def test_normalize_falls_back_to_explanation_and_tolerates_missing_evidence():
    row = _rich_hawk_i_row()
    row["measurement_context"]["ai_interpretation"] = {"explanation": "설명만 있음"}
    assert supabase_timeline.normalize_observation(row)["rationale"] == "설명만 있음"

    bare = supabase_timeline.normalize_observation(_hawk_i_row())
    assert bare["metrics"] == {}
    assert bare["rationale"] is None
    assert bare["score_advisory_level"] is None


def test_attach_dose_context_links_most_recent_prior_dose():
    observations = [
        {"observed_at": "2026-08-03T09:30:00Z"},
        {"observed_at": "2026-08-03T14:00:00Z"},
    ]
    medications = [
        {"observed_at": "2026-08-03T13:00:00Z", "medication_display": "레보도파", "dose_mg": 125},
        {"observed_at": "2026-08-03T08:00:00Z", "medication_display": "레보도파", "dose_mg": 100},
    ]

    supabase_timeline.attach_dose_context(observations, medications)

    assert observations[0]["last_dose_at"] == "2026-08-03T08:00:00Z"
    assert observations[0]["hours_since_last_dose"] == 1.5
    assert observations[0]["last_dose_mg"] == 100
    assert observations[1]["last_dose_at"] == "2026-08-03T13:00:00Z"
    assert observations[1]["hours_since_last_dose"] == 1.0
    assert observations[1]["last_dose_medication"] == "레보도파"


def test_attach_dose_context_leaves_null_when_no_prior_dose():
    observations = [{"observed_at": "2026-08-03T07:00:00Z"}, {"observed_at": None}]
    medications = [{"observed_at": "2026-08-03T08:00:00Z", "dose_mg": 100}]

    supabase_timeline.attach_dose_context(observations, medications)

    for observation in observations:
        assert observation["last_dose_at"] is None
        assert observation["hours_since_last_dose"] is None


def test_attach_dose_context_ignores_unparseable_timestamps():
    observations = [{"observed_at": "2026-08-03T10:00:00Z"}]
    medications = [
        {"observed_at": "not-a-date", "dose_mg": 50},
        {"observed_at": "2026-08-03T09:00:00Z", "dose_mg": 100},
    ]

    supabase_timeline.attach_dose_context(observations, medications)

    assert observations[0]["hours_since_last_dose"] == 1.0
    assert observations[0]["last_dose_mg"] == 100


def test_medication_event_kind_distinguishes_scheduled_from_taken():
    scheduled = _medication_row()
    scheduled["status"] = "intended"
    scheduled["dosage"] = {**scheduled["dosage"], "event_type": "scheduled_dose"}
    assert supabase_timeline.normalize_medication_statement(scheduled)["event_kind"] == "scheduled"

    taken = _medication_row()
    assert supabase_timeline.normalize_medication_statement(taken)["event_kind"] == "taken"


def test_medication_event_kind_falls_back_to_status_then_unknown():
    by_status = _medication_row()
    by_status["dosage"] = {"dose_mg": 100, "unit": "mg"}
    by_status["status"] = "intended"
    assert supabase_timeline.normalize_medication_statement(by_status)["event_kind"] == "scheduled"

    unknown = _medication_row()
    unknown["dosage"] = {"dose_mg": 100, "unit": "mg"}
    unknown["status"] = "entered-in-error"
    assert supabase_timeline.normalize_medication_statement(unknown)["event_kind"] == "unknown"
