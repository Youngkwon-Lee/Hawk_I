"""Synthetic-only unit tests for the group segment metadata adapter."""
import unittest
from copy import deepcopy

from services.group_segment_adapter import (
    adapt_segments, prepare_research_handoff, candidates_from_observation_windows,
    windows_from_tracker_observations, tracker_frames_to_observations,
    build_tracker_review_queue,
)


def candidate(**changes):
    value = {
        "start_ms": 1000, "end_ms": 3000, "task": "finger_tapping",
        "side": "right", "track_id": "synthetic_A",
        "identity_status": "confirmed", "performed": True,
        "occluded": False, "quality_status": "pass",
    }
    value.update(changes)
    return value


class GroupSegmentAdapterTests(unittest.TestCase):
    def test_observable_candidate_is_pending_and_unscored(self):
        row = adapt_segments("synthetic_session", [candidate()])[0]
        self.assertEqual(row["clip_id"], "synthetic_session_seg_0001")
        self.assertTrue(row["model_input_eligible"])
        self.assertEqual(row["review_status"], "pending_review")
        self.assertIsNone(row["score_anchor"])
        self.assertIsNone(row["part_iii_total"])
        self.assertIsNone(row["hy_stage"])

    def test_occlusion_and_unknown_identity_hold(self):
        row = adapt_segments("synthetic_session", [candidate(track_id=None, occluded=True)])[0]
        self.assertEqual(row["quality_gate"]["reasons"], ["id_unknown", "occluded"])
        self.assertFalse(row["model_input_eligible"])
        self.assertIsNone(row["track_id"])
        self.assertIsNone(row["score_anchor"])

    def test_not_performed_is_not_zero(self):
        row = adapt_segments("synthetic_session", [candidate(performed=False)])[0]
        self.assertIn("not_performed", row["quality_gate"]["reasons"])
        self.assertIsNone(row["score_anchor"])

    def test_invalid_candidates_rejected(self):
        for changes in (
            {"start_ms": 3000, "end_ms": 1000},
            {"start_ms": True},
            {"patient_id": "secret"},
            {"unexpected_payload": "anything"},
            {"task": "gait", "side": "left"},
            {"quality_status": "hold", "subject_ref": "secret"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    adapt_segments("synthetic_session", [candidate(**changes)])

    def test_unsupported_task_is_held(self):
        row = adapt_segments("synthetic_session", [candidate(task="rigidity")])[0]
        self.assertEqual(row["task"], "unknown")
        self.assertIn("unsupported_task", row["quality_gate"]["reasons"])
        self.assertFalse(row["model_input_eligible"])

    def test_handoff_requires_review_and_separate_permission(self):
        rows = adapt_segments("synthetic_session", [candidate(), candidate(occluded=True)])
        first, held = (row["clip_id"] for row in rows)
        self.assertEqual(rows[0]["handoff_status"], "awaiting_review_and_permission")
        self.assertEqual(rows[1]["handoff_status"], "blocked_quality")
        self.assertEqual(prepare_research_handoff(rows, reviewed_clip_ids={first}, approved_clip_ids=set()), [])
        self.assertEqual(prepare_research_handoff(rows, reviewed_clip_ids=set(), approved_clip_ids={first}), [])
        ready = prepare_research_handoff(rows, reviewed_clip_ids={first, held}, approved_clip_ids={first, held})
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0]["clip_id"], first)
        self.assertEqual(ready[0]["handoff_status"], "reviewed_permission_asserted_unverified")
        self.assertNotIn("score_anchor", ready[0])
        self.assertNotIn("track_id", ready[0])

    def test_forged_or_malformed_handoff_metadata_is_rejected(self):
        original = adapt_segments("synthetic_session", [candidate()])[0]
        for changes in (
            {"quality_gate": {"status": "pass", "reasons": ["occluded"]}},
            {"review_status": "published"},
            {"handoff_status": "metadata_ready"},
            {"score_anchor": {"updrs_3_4": 0}},
            {"part_iii_total": 0},
            {"hy_stage": 1},
            {"start_ms": True},
            {"end_ms": 0},
            {"side": "na"},
        ):
            with self.subTest(changes=changes):
                forged = deepcopy(original)
                forged.update(changes)
                self.assertEqual(prepare_research_handoff(
                    [forged], reviewed_clip_ids={original["clip_id"]},
                    approved_clip_ids={original["clip_id"]},
                ), [])

    def test_preannotated_windows_merge_only_contiguous_same_state(self):
        first = candidate(start_ms=0, end_ms=1000)
        second = candidate(start_ms=1000, end_ms=2000)
        third = candidate(start_ms=2000, end_ms=3000, occluded=True)
        fourth = candidate(start_ms=4000, end_ms=5000)
        grouped = candidates_from_observation_windows([fourth, second, third, first])
        self.assertEqual([(x["start_ms"], x["end_ms"]) for x in grouped], [
            (0, 2000), (2000, 3000), (4000, 5000),
        ])
        adapted = adapt_segments("synthetic_session", grouped)
        self.assertEqual([x["quality_gate"]["status"] for x in adapted], ["pass", "hold", "pass"])

    def test_overlapping_or_invalid_windows_rejected(self):
        for windows in (
            [candidate(start_ms=0, end_ms=2000), candidate(start_ms=1000, end_ms=3000)],
            [candidate(start_ms=True)],
            [candidate(media_uri="sensitive")],
        ):
            with self.subTest(windows=windows):
                with self.assertRaises(ValueError):
                    candidates_from_observation_windows(windows)

    def test_simultaneous_distinct_tracks_are_retained(self):
        windows = [
            candidate(start_ms=0, end_ms=1000, track_id="synthetic_A"),
            candidate(start_ms=0, end_ms=1000, track_id="synthetic_B"),
            candidate(start_ms=1000, end_ms=2000, track_id="synthetic_A"),
        ]
        grouped = candidates_from_observation_windows(windows)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["end_ms"], 2000)
        self.assertEqual(grouped[1]["track_id"], "synthetic_B")

    def test_unknown_tracks_are_never_merged(self):
        windows = [
            candidate(start_ms=0, end_ms=1000, track_id=None, identity_status="unknown"),
            candidate(start_ms=1000, end_ms=2000, track_id=None, identity_status="unknown"),
        ]
        self.assertEqual(len(candidates_from_observation_windows(windows)), 2)

    def test_external_tracker_claims_remain_held(self):
        observations = [
            {"start_ms": 0, "end_ms": 1000, "track_id": "A", "track_state": "stable", "task": "gait", "side": "na"},
            {"start_ms": 1000, "end_ms": 2000, "track_id": "A", "track_state": "switched", "task": "gait", "side": "na"},
        ]
        windows = windows_from_tracker_observations(observations)
        rows = adapt_segments("synthetic", candidates_from_observation_windows(windows))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["quality_gate"]["status"] == "hold" for row in rows))
        self.assertTrue(all(row["track_id"] is None for row in rows))
        self.assertEqual(prepare_research_handoff(rows, reviewed_clip_ids={r["clip_id"] for r in rows}, approved_clip_ids={r["clip_id"] for r in rows}), [])

    def test_tracker_input_rejects_extra_fields_and_bad_state(self):
        base = {"start_ms": 0, "end_ms": 1000, "track_id": "A", "track_state": "stable"}
        for extra in ({"patient_id": "secret"}, {"track_state": "confirmed"}, {"start_ms": True}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    windows_from_tracker_observations([dict(base, **extra)])

    def test_tracker_frame_adapter_keeps_parallel_tracks_held(self):
        frames = [
            {"start_ms": 0, "end_ms": 40, "tracks": [
                {"track_id": "A", "state": "stable"}, {"track_id": "B", "state": "stable"},
            ]},
            {"start_ms": 40, "end_ms": 80, "tracks": [
                {"track_id": "A", "state": "switched"},
            ]},
        ]
        obs = tracker_frames_to_observations(frames)
        self.assertEqual(len(obs), 3)
        windows = windows_from_tracker_observations(obs)
        rows = adapt_segments("synthetic", candidates_from_observation_windows(windows))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["quality_gate"]["status"] == "hold" for row in rows))

    def test_tracker_frame_adapter_rejects_overlap_and_duplicate_tracks(self):
        for frames in (
            [{"start_ms": 0, "end_ms": 40, "tracks": []}, {"start_ms": 20, "end_ms": 60, "tracks": []}],
            [{"start_ms": 0, "end_ms": 40, "tracks": [{"track_id": "A", "state": "stable"}, {"track_id": "A", "state": "stable"}]}],
            [{"start_ms": 0, "end_ms": 40, "tracks": [{"track_id": "A", "state": "stable", "name": "secret"}]}],
        ):
            with self.subTest(frames=frames):
                with self.assertRaises(ValueError):
                    tracker_frames_to_observations(frames)

    def test_tracker_review_queue_is_held_end_to_end(self):
        frames = [
            {"start_ms": 0, "end_ms": 40, "tracks": [
                {"track_id": "synthetic_A", "state": "stable"},
                {"track_id": "synthetic_B", "state": "stable"},
            ]},
            {"start_ms": 40, "end_ms": 80, "tracks": [
                {"track_id": "synthetic_A", "state": "switched"},
                {"track_id": "synthetic_B", "state": "stable"},
            ]},
        ]
        rows = build_tracker_review_queue("synthetic_session", frames)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["quality_gate"]["status"] == "hold" for row in rows))
        self.assertTrue(all(row["model_input_eligible"] is False for row in rows))
        self.assertTrue(all(row["score_anchor"] is None for row in rows))

    def test_gait_is_not_side_scoped(self):
        row = adapt_segments("synthetic_session", [candidate(task="gait", side="na")])[0]
        self.assertEqual(row["quality_gate"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
