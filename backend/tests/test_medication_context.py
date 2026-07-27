import json
from pathlib import Path

import pytest

from services.medication_context import (
    describe_medication_timing,
    parse_medication_context,
)


def test_parse_medication_context_whitelists_and_recomputes_elapsed_time():
    context = parse_medication_context(json.dumps({
        "available": True,
        "source": "untrusted",
        "medication": " 레보도파 ",
        "dose_mg": 100,
        "taken_at": "2026-07-27T00:00:00Z",
        "assessment_at": "2026-07-27T01:30:00Z",
        "hours_before_assessment": 999,
        "secret": "discard-me",
    }))

    assert context == {
        "available": True,
        "source": "patient_reported_local",
        "assessment_at": "2026-07-27T01:30:00Z",
        "taken_at": "2026-07-27T00:00:00Z",
        "medication": "레보도파",
        "dose_mg": 100.0,
        "hours_before_assessment": 1.5,
    }


def test_parse_medication_context_rejects_invalid_or_future_timestamps():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_medication_context("{")
    with pytest.raises(ValueError, match="must not be after"):
        parse_medication_context(json.dumps({
            "available": True,
            "taken_at": "2026-07-27T02:00:00Z",
            "assessment_at": "2026-07-27T01:00:00Z",
        }))


def test_medication_timing_is_descriptive_not_an_efficacy_claim():
    timing = describe_medication_timing({
        "available": True,
        "hours_before_assessment": 1.5,
    })

    assert timing == {
        "available": True,
        "relationship": "after_patient_reported_dose",
        "hours_after_reported_dose": 1.5,
        "timing_window": "within_2_hours",
        "evidence_level": "single_observation",
        "can_infer_medication_effect": False,
    }


def test_deidentified_frontend_contract_fixture_keeps_the_safety_boundary():
    fixture_path = Path(__file__).parent / "fixtures" / "medication_result.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert payload["patient_id"].startswith("synthetic-")
    assert payload["medication_context"]["source"] == "patient_reported_local"
    assert payload["medication_timing"]["evidence_level"] == "single_observation"
    assert payload["medication_timing"]["can_infer_medication_effect"] is False

    followup_path = Path(__file__).parent / "fixtures" / "medication_result_followup.json"
    followup = json.loads(followup_path.read_text(encoding="utf-8"))
    assert followup["patient_id"] == payload["patient_id"]
    assert followup["medication_context"]["dose_mg"] == payload["medication_context"]["dose_mg"]
    assert followup["medication_timing"]["can_infer_medication_effect"] is False
