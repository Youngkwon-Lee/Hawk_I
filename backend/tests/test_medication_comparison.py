from services.medication_comparison import build_medication_comparison


def result(date, score, hours, speed, dose=100, patient="synthetic-person"):
    return {
        "video_id": f"result-{date}",
        "date": date,
        "task_type": "finger_tapping",
        "score": score,
        "metrics": {"tapping_speed": speed, "amplitude_mean": 0.4, "fatigue_rate": 8},
        "patient_id": patient,
        "medication_context": {
            "available": True,
            "source": "patient_reported_local",
            "medication": "비식별 테스트 약물",
            "dose_mg": dose,
            "assessment_at": date,
            "hours_before_assessment": hours,
        },
        "medication_timing": {
            "available": True,
            "hours_after_reported_dose": hours,
            "can_infer_medication_effect": False,
        },
    }


def test_repeated_comparison_keeps_patients_and_doses_separate():
    comparison = build_medication_comparison([
        result("2026-07-27T01:00:00Z", 2, 1, 3.0),
        result("2026-07-28T01:00:00Z", 1, 2, 3.5, dose=150),
        result("2026-07-29T01:00:00Z", 1, 2, 3.6, patient="other-person"),
    ])

    assert comparison["available"] is False
    assert comparison["observation_count"] == 1
    assert comparison["can_infer_medication_effect"] is False


def test_repeated_comparison_reports_observed_deltas_without_efficacy_claim():
    comparison = build_medication_comparison([
        result("2026-07-27T01:00:00Z", 2, 0.5, 3.0),
        result("2026-07-28T01:00:00Z", 1, 1.5, 3.5),
    ])

    assert comparison["available"] is True
    assert comparison["observation_count"] == 2
    assert comparison["observed_change"] == {
        "score": -1.0,
        "tapping_speed": 0.5,
        "amplitude_mean": 0.0,
        "fatigue_rate": 0.0,
    }
    assert comparison["can_infer_medication_effect"] is False
    assert comparison["requires_clinician_review"] is True
