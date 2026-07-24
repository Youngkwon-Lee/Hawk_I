"""
Tests for optional Supabase observation persistence.
"""


def _sample_result(score=2.5):
    return {
        "success": True,
        "id": "sample_video_123",
        "patient_id": "local-test",
        "video_type": "finger_tapping",
        "auto_detected": False,
        "confidence": 0.91,
        "scoring_method": "coral",
        "ml_model_type": "rf",
        "updrs_score": {
            "total_score": score,
            "severity": "Mild",
            "confidence": 0.83,
            "details": {"model": "coral"},
        },
        "metrics": {
            "tapping_speed": 3.2,
            "fatigue_rate": 12.4,
        },
        "skeleton_data": {
            "total_frames": 180,
            "detection_rate": 0.94,
            "mode": "hand",
            "skeleton_video_url": "/files/sample_skeleton.mp4",
            "original_video_url": "/files/sample.mp4",
            "fps": 30,
            "keypoints": [{"frame": 1, "keypoints": [{"x": 0.1, "y": 0.2}]}],
        },
        "events": [{"timestamp": 1.2, "type": "hesitation"}],
    }


def _sample_physio_context():
    return {
        "subject_person_id": "person-1",
        "organization_id": "org-1",
        "created_by_person_id": "person-1",
        "performer_person_id": "person-1",
    }


def test_supabase_observation_disabled_without_required_env(monkeypatch):
    from services.supabase_observations import save_analysis_observation

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
        "HAWKEYE_SUPABASE_CREATED_BY_PERSON_ID",
        "HAWKEYE_SUPABASE_PERFORMER_PERSON_ID",
        "HAWKEYE_SUPABASE_ACTIVITY_SESSION_ID",
    ]:
        monkeypatch.delenv(name, raising=False)

    result = save_analysis_observation(_sample_result())

    assert result.enabled is False
    assert result.saved is False


def test_save_analysis_observation_requires_explicit_physio_context(monkeypatch):
    from services import supabase_observations

    monkeypatch.setenv("HAWKEYE_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("HAWKEYE_SUPABASE_SERVICE_KEY", "server-secret")
    monkeypatch.setenv("HAWKEYE_SUPABASE_SUBJECT_PERSON_ID", "person-default")
    monkeypatch.setenv("HAWKEYE_SUPABASE_ORGANIZATION_ID", "org-1")
    monkeypatch.setenv("HAWKEYE_SUPABASE_CREATED_BY_PERSON_ID", "person-default")
    monkeypatch.setenv("HAWKEYE_SUPABASE_PERFORMER_PERSON_ID", "person-default")

    def fake_post(*args, **kwargs):
        raise AssertionError("Supabase should not be called without explicit subject context")

    monkeypatch.setattr(supabase_observations.requests, "post", fake_post)

    result = supabase_observations.save_analysis_observation(_sample_result(score=2))

    assert result.enabled is True
    assert result.saved is False
    assert result.reason == "missing explicit physio_app subject context"


def test_build_observation_row_uses_quantity_for_fractional_scores():
    from services.supabase_observations import (
        SupabaseObservationConfig,
        build_observation_row,
    )

    config = SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="secret",
        subject_person_id="person-1",
        organization_id="org-1",
        created_by="person-1",
        performer_person_id="person-1",
        activity_session_id="session-1",
    )

    row = build_observation_row(_sample_result(score=2.5), config)

    assert row["code"] == "UPDRS_3_4"
    assert row["performer_person_id"] == "person-1"
    assert row["activity_session_id"] == "session-1"
    assert row["value_type"] == "quantity"
    assert row["value_quantity"] == 2.5
    assert row["value_unit"] == "score"
    assert row["measurement_context"]["app_source"] == "hawk_i"
    assert row["measurement_context"]["score_raw"] == 2.5
    assert "keypoints" not in row["measurement_context"]["skeleton_summary"]


def test_build_observation_row_uses_integer_for_integral_scores():
    from services.supabase_observations import (
        SupabaseObservationConfig,
        build_observation_row,
    )

    config = SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="secret",
        subject_person_id="person-1",
        organization_id="org-1",
        created_by="person-1",
        performer_person_id="person-1",
        activity_session_id="session-1",
    )

    row = build_observation_row(_sample_result(score=2), config)

    assert row["value_type"] == "integer"
    assert row["value_integer"] == 2


def test_build_observation_row_normalizes_non_finite_context_numbers():
    from services.supabase_observations import (
        SupabaseObservationConfig,
        build_observation_row,
    )

    result = _sample_result(score=2)
    result["metrics"]["tapping_speed"] = float("nan")
    config = SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="secret",
        subject_person_id="person-1",
        organization_id="org-1",
        created_by="person-1",
        performer_person_id="person-1",
        activity_session_id="session-1",
    )

    row = build_observation_row(result, config)

    assert row["measurement_context"]["metrics"]["tapping_speed"] is None


def test_build_activity_session_row_uses_assessment_context():
    from services.supabase_observations import (
        SupabaseObservationConfig,
        build_activity_session_row,
    )

    config = SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="secret",
        subject_person_id="person-1",
        organization_id="org-1",
        created_by="person-1",
        performer_person_id="person-1",
    )

    row = build_activity_session_row(_sample_result(score=3), config)

    assert row["activity_type"] == "assessment"
    assert row["source"] == "camera"
    assert row["status"] == "completed"
    assert row["metrics"]["app_source"] == "hawk_i"
    assert row["metrics"]["score_raw"] == 3
    assert row["exercise_log"]["task_code"] == "UPDRS_3_4"


def test_config_for_analysis_result_uses_physio_context_overrides():
    from services.supabase_observations import (
        SupabaseObservationConfig,
        build_observation_row,
        config_for_analysis_result,
    )

    base = SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="secret",
        subject_person_id="person-default",
        organization_id="org-default",
        created_by="creator-default",
        performer_person_id="performer-default",
        activity_session_id="session-1",
    )
    result = _sample_result(score=2)
    result["physio_context"] = {
        "subject_person_id": "person-selected",
        "organization_id": "org-selected",
        "created_by_person_id": "creator-selected",
        "performer_person_id": "performer-selected",
    }

    config = config_for_analysis_result(base, result)
    row = build_observation_row(result, config)

    assert row["subject_person_id"] == "person-selected"
    assert row["organization_id"] == "org-selected"
    assert row["created_by"] == "creator-selected"
    assert row["performer_person_id"] == "performer-selected"
    assert row["measurement_context"]["physio_context"]["subject_person_id"] == "person-selected"


def test_save_analysis_observation_posts_to_supabase(monkeypatch):
    from services import supabase_observations

    monkeypatch.setenv("HAWKEYE_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("HAWKEYE_SUPABASE_SERVICE_KEY", "server-secret")
    monkeypatch.delenv("HAWKEYE_SUPABASE_SUBJECT_PERSON_ID", raising=False)
    monkeypatch.setenv("HAWKEYE_SUPABASE_ORGANIZATION_ID", "org-1")
    monkeypatch.setenv("HAWKEYE_SUPABASE_CREATED_BY_PERSON_ID", "person-1")
    monkeypatch.setenv("HAWKEYE_SUPABASE_PERFORMER_PERSON_ID", "person-1")

    captured = []

    class FakeResponse:
        def __init__(self, returned_id):
            self.status_code = 201
            self.text = ""
            self._returned_id = returned_id

        def json(self):
            return [{"id": self._returned_id}]

    def fake_post(url, headers, json, timeout):
        captured.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        returned_id = "session-1" if url.endswith("/activity_sessions") else "obs-1"
        return FakeResponse(returned_id)

    monkeypatch.setattr(supabase_observations.requests, "post", fake_post)

    sample_result = _sample_result(score=2)
    sample_result["physio_context"] = _sample_physio_context()

    result = supabase_observations.save_analysis_observation(sample_result)

    assert result.enabled is True
    assert result.saved is True
    assert result.observation_id == "obs-1"
    assert result.activity_session_id == "session-1"
    assert captured[0]["url"] == "https://example.supabase.co/rest/v1/activity_sessions"
    assert captured[1]["url"] == "https://example.supabase.co/rest/v1/observations"
    assert captured[1]["headers"]["Authorization"] == "Bearer server-secret"
    assert captured[1]["json"]["source_type"] == "ai"
    assert captured[1]["json"]["subject_person_id"] == "person-1"
    assert captured[1]["json"]["organization_id"] == "org-1"
    assert captured[1]["json"]["performer_person_id"] == "person-1"
    assert captured[1]["json"]["activity_session_id"] == "session-1"
