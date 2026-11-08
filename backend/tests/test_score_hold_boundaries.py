"""Synthetic-only regression for held report and pre-write score validation."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class _FakeCtx:
    def __init__(self, score, validation):
        self.status = "clinical_done"
        self.task_type = "finger_tapping"
        self.clinical_scores = score
        self.validation_result = validation
        self.kinematic_metrics = {"tapping_speed": 2.5}
        self.clinical_charts = None
        self.report = None
        self.logs = []

    def log(self, *args, **kwargs):
        self.logs.append((args, kwargs))


class _ForbiddenInterpreter:
    def interpret_finger_tapping(self, *args, **kwargs):
        raise AssertionError("no report may be generated on quality hold")


class ScoreHoldTests(unittest.TestCase):
    def _report(self, score, validation):
        from agents.report_agent import ReportAgent
        agent = object.__new__(ReportAgent)
        agent.interpreter = _ForbiddenInterpreter()
        ctx = _FakeCtx(score, validation)
        agent.process(ctx)
        return ctx

    def test_missing_score_never_falls_back_to_zero(self):
        ctx = self._report({}, {"is_valid": True})
        self.assertIsNone(ctx.clinical_scores)
        self.assertIsNone(ctx.report)
        self.assertEqual(ctx.status, "report_done")

    def test_invalid_validation_holds_even_when_score_is_zero(self):
        ctx = self._report({"total_score": 0}, {"is_valid": False})
        self.assertIsNone(ctx.clinical_scores)
        self.assertIsNone(ctx.report)
        self.assertEqual(ctx.logs[-1][1]["meta"]["reason"], "validation_missing_or_failed")

    def test_missing_validation_holds_nonzero_score(self):
        ctx = self._report({"total_score": 2}, None)
        self.assertIsNone(ctx.clinical_scores)
        self.assertIsNone(ctx.report)

    def test_non_finite_and_boolean_scores_hold(self):
        for value in (float("nan"), float("inf"), True, -1, 5, "0"):
            with self.subTest(value=value):
                ctx = self._report({"total_score": value}, {"is_valid": True})
                self.assertIsNone(ctx.report)
                self.assertIsNone(ctx.clinical_scores)

    def test_valid_numeric_zero_is_not_a_missing_score(self):
        from agents.report_agent import ReportAgent
        agent = object.__new__(ReportAgent)
        invoked = []

        class Stub:
            def interpret_finger_tapping(self, score, *args):
                invoked.append(score)
                return SimpleNamespace(summary="synthetic", explanation="research",
                                       recommendations=[])

        agent.interpreter = Stub()
        ctx = _FakeCtx({"total_score": 0, "severity": "research"}, {"is_valid": True})
        result = agent.process(ctx)
        self.assertEqual(invoked, [0])
        self.assertIsNotNone(result.report)
        self.assertEqual(result.clinical_scores["total_score"], 0)

    def test_no_score_skips_all_supabase_writes_before_session(self):
        from services import supabase_observations as so
        config = so.SupabaseObservationConfig(
            url="https://example.invalid", key="synthetic",
            subject_person_id="fake-subject", organization_id="fake-org",
            created_by="fake", performer_person_id="fake",
        )
        for bad in (None, float("nan"), -1, True, 5):
            with self.subTest(bad=bad):
                with patch.object(so, "get_supabase_observation_config", return_value=config), \
                     patch.object(so, "_post_row", side_effect=AssertionError("write attempted")):
                    result = so.save_analysis_observation({
                        "video_type": "finger_tapping",
                        "id": "fake",
                        "updrs_score": {"total_score": bad},
                        "physio_context": {"subject_person_id": "fake-subject", "organization_id": "fake-org"}
                    })
                    self.assertFalse(result.saved)
                    self.assertIn("writes held", result.reason)

    def test_numeric_zero_stays_available_for_separate_authorized_path(self):
        from services import supabase_observations as so
        self.assertEqual(so._score_value({"updrs_score": {"total_score": 0}}), 0)
        self.assertIsNone(so._score_value({"updrs_score": {"total_score": None}}))


if __name__ == "__main__":
    unittest.main()
