"""Synthetic-only display-gate contract. No real subjects or model calls."""
import json
import unittest
from copy import deepcopy

from services.finger_result_projection import project_finger_review


def candidate(score=0):
    return {
        "video_type": "finger_tapping",
        "patient_id": "synthetic-do-not-echo",
        "updrs_score": {"total_score": score, "severity": "prototype"},
        "ai_interpretation": {"summary": "synthetic score leaked here"},
        "vlm_results": [{"score": score, "reasoning": "synthetic score leaked here"}],
        "metrics": {"amplitude": 0.2},
    }


def sampling(status="review"):
    return {"task": "finger_tapping", "precheck": {"status": status}}


def trace(status="pass"):
    return {"task": "finger_tapping", "quality_gate": {"status": status}}


class FingerResultProjectionTests(unittest.TestCase):
    def test_failed_sampling_holds_even_zero_candidate(self):
        result = project_finger_review(candidate(0),
                                       sampling_precheck=sampling("hold"),
                                       trace_qc=trace(),
                                       validation_result={"is_valid": True})
        self.assertEqual(result["state"], "hold")
        self.assertIn("sampling_precheck_not_satisfied", result["hold_reasons"])
        self.assertIsNone(result["clinical_score"])
        self.assertIsNone(result["score_anchor"]["updrs_3_4"])

    def test_passed_structural_checks_still_require_human_review(self):
        result = project_finger_review(candidate(4),
                                       sampling_precheck=sampling(),
                                       trace_qc=trace(),
                                       validation_result={"is_valid": True})
        self.assertEqual(result["state"], "human_review_required")
        self.assertEqual(result["hold_reasons"], [])
        self.assertFalse(result["research_prediction_exposed"])
        self.assertIsNone(result["clinical_score"])

    def test_missing_evidence_defaults_hold(self):
        result = project_finger_review(candidate(),
                                       sampling_precheck=None,
                                       trace_qc=None,
                                       validation_result=None)
        self.assertEqual(result["state"], "hold")
        self.assertEqual(len(result["hold_reasons"]), 5)

    def test_trace_hold_and_invalid_validation_do_not_become_score_zero(self):
        result = project_finger_review(candidate(3),
                                       sampling_precheck=sampling(),
                                       trace_qc=trace("hold"),
                                       validation_result={"is_valid": False})
        self.assertEqual(result["state"], "hold")
        self.assertIn("analysis_validation_not_passed", result["hold_reasons"])
        self.assertIn("trace_quality_not_passed", result["hold_reasons"])
        self.assertIsNone(result["clinical_report"])

    def test_untrusted_validation_flag_cannot_approve(self):
        result = project_finger_review(candidate(2),
                                       sampling_precheck=sampling(),
                                       trace_qc=trace(),
                                       validation_result={"is_valid": "true"})
        self.assertEqual(result["state"], "hold")

    def test_bad_statuses_and_cross_task_gates_hold(self):
        result = project_finger_review(candidate(1),
                                       sampling_precheck={"task": "gait", "precheck": {"status": "structural_precheck_pass"}},
                                       trace_qc={"task": "gait", "quality_gate": {"status": "pass"}},
                                       validation_result={"is_valid": True})
        self.assertEqual(result["state"], "hold")
        self.assertIn("trace_qc_missing_or_task_mismatch", result["hold_reasons"])

    def test_unknown_tasks_are_not_silently_shown_as_finger(self):
        bad = candidate()
        bad["video_type"] = "gait"
        with self.assertRaisesRegex(ValueError, "Only finger_tapping"):
            project_finger_review(bad, sampling_precheck=sampling(),
                                  trace_qc=trace(), validation_result={"is_valid": True})

    def test_no_raw_identifiers_or_nested_scores_leak_and_input_unchanged(self):
        raw = candidate(0)
        original = deepcopy(raw)
        result = project_finger_review(raw, sampling_precheck=sampling(),
                                       trace_qc=trace(), validation_result={"is_valid": True})
        wire = json.dumps(result)
        for forbidden in ("synthetic-do-not-echo", "updrs_score", "vlm_results",
                          "ai_interpretation", "metrics", "prototype", "synthetic score leaked"):
            self.assertNotIn(forbidden, wire)
        self.assertEqual(raw, original)


if __name__ == "__main__":
    unittest.main()
