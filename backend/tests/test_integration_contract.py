from services.supabase_observations import (
    SupabaseObservationConfig,
    build_activity_session_row,
    build_observation_row,
)
from services.timeline_service import TimelineService


def _result():
    return {
        "id": "analysis-123",
        "patient_id": "person-123",
        "assessment_session_id": "assessment-123",
        "video_type": "finger_tapping",
        "updrs_score": {"total_score": 2, "confidence": 0.8},
        "metrics": {"tapping_speed": 3.1},
    }


def _config():
    return SupabaseObservationConfig(
        url="https://example.supabase.co",
        key="secret",
        subject_person_id="person-123",
        organization_id="org-123",
        created_by="person-123",
        performer_person_id="person-123",
        activity_session_id="activity-123",
    )


def test_assessment_session_id_is_persisted_in_hawk_i_rows():
    activity = build_activity_session_row(_result(), _config())
    observation = build_observation_row(_result(), _config())

    assert activity["metrics"]["assessment_session_id"] == "assessment-123"
    assert observation["measurement_context"]["assessment_session_id"] == "assessment-123"


def test_medication_timeline_is_empty_until_patient_data_is_connected():
    timeline = TimelineService().get_patient_timeline("person-123")

    assert timeline == {
        "patient_id": "person-123",
        "available": False,
        "source": "not_connected",
        "timeline": [],
        "pattern": None,
        "recommendations": None,
    }
