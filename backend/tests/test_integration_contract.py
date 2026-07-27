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
        "medication_context": {
            "available": True,
            "source": "patient_reported_local",
            "medication": "레보도파",
            "dose_mg": 100.0,
            "taken_at": "2026-07-27T00:00:00Z",
            "assessment_at": "2026-07-27T01:30:00Z",
            "hours_before_assessment": 1.5,
        },
        "medication_timing": {
            "available": True,
            "relationship": "after_patient_reported_dose",
            "hours_after_reported_dose": 1.5,
            "timing_window": "within_2_hours",
            "evidence_level": "single_observation",
            "can_infer_medication_effect": False,
        },
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
    assert activity["metrics"]["medication_context"]["medication"] == "레보도파"
    assert activity["metrics"]["medication_timing"]["can_infer_medication_effect"] is False
    assert observation["measurement_context"]["medication_context"]["dose_mg"] == 100.0
    assert observation["measurement_context"]["medication_timing"]["timing_window"] == "within_2_hours"


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
