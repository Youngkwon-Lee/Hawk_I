"""Synthetic-only tests; no raw videos, user identifiers or clinical grades."""
import unittest

from services.finger_sampling_precheck import inspect_finger_sampling


def frames(total=400, fps=80, stride=1):
    return [
        {"frame_number": idx, "timestamp": idx / fps,
         "landmarks": [{"id": 4, "x": .5, "y": .5, "z": 0.0}]}
        for idx in range(0, total, stride)
    ]


def run_precheck(**kwargs):
    opts = dict(
        task_type="finger_tapping",
        source_fps=80.0,
        source_frame_count=400,
        configured_stride=1,
        retained_frames=frames(),
    )
    opts.update(kwargs)
    return inspect_finger_sampling(**opts)


class FingerSamplingPrecheckTests(unittest.TestCase):
    def test_full_80fps_400frames_are_available_as_research_candidate(self):
        out = run_precheck()
        self.assertEqual(out["policy"]["recommended_stride"], 1)
        self.assertEqual(out["sampling_evidence"]["expected_processed_frames"], 400)
        self.assertEqual(out["precheck"]["status"], "review")
        self.assertIn("source_presentation_timestamps_unverified", out["precheck"]["warnings"])
        self.assertIsNone(out["score_anchor"]["updrs_3_4"])

    def test_current_skip_three_retains_134_frames_and_warns(self):
        out = run_precheck(
            configured_stride=3, retained_frames=frames(stride=3)
        )
        self.assertEqual(out["sampling_evidence"]["expected_processed_frames"], 134)
        self.assertAlmostEqual(out["sampling_evidence"]["nominal_retained_fps"], 80 / 3)
        self.assertEqual(out["precheck"]["status"], "review")
        self.assertIn("temporal_decimation_risk", out["precheck"]["warnings"])
        self.assertEqual(out["policy"]["recommended_stride"], 1)
        self.assertFalse(out["policy"]["applied_to_runtime"])

    def test_tracking_dropout_produces_hold_not_zero(self):
        observed = frames(stride=3)
        observed.pop(50)
        out = run_precheck(configured_stride=3, retained_frames=observed)
        self.assertEqual(out["precheck"]["status"], "hold")
        self.assertIn("missing_or_unexpected_retained_frames", out["precheck"]["reasons"])
        self.assertIsNone(out["score_anchor"]["updrs_3_4"])

    def test_repeated_timestamp_holds(self):
        observed = frames(stride=3)
        observed[4]["timestamp"] = observed[3]["timestamp"]
        out = run_precheck(configured_stride=3, retained_frames=observed)
        self.assertEqual(out["precheck"]["status"], "hold")
        self.assertIn("nonmonotonic_frame_timebase", out["precheck"]["reasons"])

    def test_out_of_order_and_nonfinite_hold(self):
        observed = frames(stride=3)
        observed[4]["frame_number"] = observed[3]["frame_number"]
        out = run_precheck(configured_stride=3, retained_frames=observed)
        self.assertEqual(out["precheck"]["status"], "hold")
        observed = frames(stride=3)
        observed[4]["timestamp"] = float("nan")
        out = run_precheck(configured_stride=3, retained_frames=observed)
        self.assertEqual(out["precheck"]["status"], "hold")
        self.assertIn("invalid_frame_index_or_timestamp", out["precheck"]["reasons"])

    def test_short_and_missing_landmarks_hold(self):
        out = run_precheck(source_frame_count=1, retained_frames=frames(total=1))
        self.assertEqual(out["precheck"]["status"], "hold")
        self.assertIn("insufficient_temporal_samples", out["precheck"]["reasons"])
        observed = frames(stride=3)
        observed[4]["landmarks"] = []
        out = run_precheck(configured_stride=3, retained_frames=observed)
        self.assertIn("unavailable_pose_landmarks", out["precheck"]["reasons"])

    def test_other_task_no_policy_and_no_score(self):
        out = run_precheck(task_type="gait")
        self.assertEqual(out["precheck"]["status"], "hold")
        self.assertIsNone(out["policy"]["recommended_stride"])
        self.assertIsNone(out["score_anchor"]["updrs_3_4"])

    def test_invalid_video_metadata_holds(self):
        for bad in (0, -80, float("inf"), float("nan"), True):
            out = run_precheck(source_fps=bad)
            self.assertEqual(out["precheck"]["status"], "hold")
            self.assertIsNone(out["score_anchor"]["updrs_3_4"])

    def test_pts_evidence_flag_never_approves_clinical_score(self):
        out = run_precheck(time_source="pts_observed")
        self.assertIn("pts_provenance_requires_independent_verification",
                      out["precheck"]["warnings"])
        self.assertIsNone(out["score_anchor"]["updrs_3_4"])


if __name__ == "__main__":
    unittest.main()
