import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_hubu_pd4t_comparison_eda.py"
)
SPEC = importlib.util.spec_from_file_location("hubu_pd4t_comparison_eda", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HubuPd4tComparisonEdaTests(unittest.TestCase):
    def test_normalize_windows_path_for_wsl(self):
        self.assertEqual(
            MODULE.normalize_windows_path(
                r"D:\clip\finger_tapping\subject\clip.mp4"
            ),
            Path("/mnt/d/clip/finger_tapping/subject/clip.mp4"),
        )

    def test_parse_fraction_handles_common_ffprobe_values(self):
        self.assertAlmostEqual(MODULE.parse_fraction("30000/1001"), 29.97003)
        self.assertIsNone(MODULE.parse_fraction("0/0"))
        self.assertIsNone(MODULE.parse_fraction(None))

    def test_numeric_summary_uses_interpolated_quartiles(self):
        self.assertEqual(
            MODULE.numeric_summary([1, 2, 3, 4]),
            {
                "count": 4,
                "min": 1.0,
                "p25": 1.75,
                "median": 2.5,
                "p75": 3.25,
                "max": 4.0,
            },
        )

    def test_hubu_identifier_pattern_extracts_cohort_subject_and_side(self):
        match = MODULE.HUBU_ID_PATTERN.match("CONTROL17_DCHA")
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), ("CONTROL", "17", "DCHA"))


if __name__ == "__main__":
    unittest.main()
