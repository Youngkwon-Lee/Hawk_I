"""Synthetic bounding-box association and review-boundary checks."""
import unittest

from services.synthetic_box_association import associate_synthetic_boxes
from services.group_segment_adapter import (
    tracker_frames_to_observations, windows_from_tracker_observations,
    candidates_from_observation_windows, adapt_segments,
)


class SyntheticAssociationTests(unittest.TestCase):
    def test_two_tracks_preserved_without_identity_verification(self):
        frames = [
            {"start_ms": 0, "end_ms": 40, "boxes": [(0.1, 0.1, 0.3, 0.5), (0.6, 0.1, 0.8, 0.5)]},
            {"start_ms": 40, "end_ms": 80, "boxes": [(0.11, 0.1, 0.31, 0.5), (0.59, 0.1, 0.79, 0.5)]},
        ]
        tracked = associate_synthetic_boxes(frames)
        self.assertEqual(tracked[0]["tracks"], tracked[1]["tracks"])
        rows = adapt_segments("synthetic", candidates_from_observation_windows(
            windows_from_tracker_observations(tracker_frames_to_observations(tracked))
        ))
        self.assertTrue(rows)
        self.assertTrue(all(row["quality_gate"]["status"] == "hold" for row in rows))

    def test_ambiguous_association_is_not_stable(self):
        frames = [
            {"start_ms": 0, "end_ms": 40, "boxes": [(0.1, 0.1, 0.5, 0.5)]},
            {"start_ms": 40, "end_ms": 80, "boxes": [(0.1, 0.1, 0.3, 0.5), (0.3, 0.1, 0.5, 0.5)]},
        ]
        tracked = associate_synthetic_boxes(frames)
        self.assertEqual([x["state"] for x in tracked[1]["tracks"]], ["ambiguous", "ambiguous"])

    def test_bad_boxes_and_overlapping_frames_rejected(self):
        cases = [
            [{"start_ms": 0, "end_ms": 40, "boxes": [(0.5, 0, 0.2, 1)]}],
            [{"start_ms": 0, "end_ms": 40, "boxes": [(0, 0, 1, 1)]},
             {"start_ms": 20, "end_ms": 60, "boxes": []}],
            [{"start_ms": 0, "end_ms": 40, "boxes": [(0, 0, True, 1)]}],
            [{"start_ms": 0, "end_ms": 40, "boxes": [], "patient_id": "secret"}],
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                associate_synthetic_boxes(case)


if __name__ == "__main__":
    unittest.main()
