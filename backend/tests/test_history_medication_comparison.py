import shutil
from pathlib import Path


def test_history_stats_exposes_safe_repeated_medication_comparison(tmp_path):
    from app import app

    fixtures = Path(__file__).parent / "fixtures"
    shutil.copyfile(
        fixtures / "medication_result.json",
        tmp_path / "medication-contract-preview_result.json",
    )
    shutil.copyfile(
        fixtures / "medication_result_followup.json",
        tmp_path / "medication-contract-followup_result.json",
    )
    app.config["UPLOAD_FOLDER"] = str(tmp_path)

    response = app.test_client().get(
        "/api/history/stats",
        query_string={
            "patient_id": "synthetic-medication-contract",
            "task_type": "finger_tapping",
        },
    )

    assert response.status_code == 200
    comparison = response.get_json()["data"]["medication_comparison"]
    assert comparison["available"] is True
    assert comparison["observation_count"] == 2
    assert comparison["observed_change"]["score"] == -1.0
    assert comparison["observed_change"]["tapping_speed"] == 0.5
    assert comparison["can_infer_medication_effect"] is False
    assert comparison["requires_clinician_review"] is True
