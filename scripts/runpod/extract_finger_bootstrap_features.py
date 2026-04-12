"""
Extract finger tapping kinematic features for a bootstrap candidate CSV.

This is a narrow wrapper around the existing MediaPipe + MetricsCalculator path
so we can compute feature-fusion inputs for a small set of finger-tapping
videos without running a full train/test split job.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from services.mediapipe_processor import MediaPipeProcessor  # noqa: E402
from services.metrics_calculator import MetricsCalculator  # noqa: E402


FEATURE_FIELDS = [
    "tapping_speed",
    "total_taps",
    "amplitude_mean",
    "amplitude_std",
    "amplitude_decrement",
    "first_half_amplitude",
    "second_half_amplitude",
    "opening_velocity_mean",
    "closing_velocity_mean",
    "peak_velocity_mean",
    "velocity_decrement",
    "rhythm_variability",
    "halt_count",
    "hesitation_count",
    "freeze_episodes",
    "fatigue_rate",
    "duration",
    "velocity_first_third",
    "velocity_mid_third",
    "velocity_last_third",
    "amplitude_first_third",
    "amplitude_mid_third",
    "amplitude_last_third",
    "velocity_slope",
    "amplitude_slope",
    "rhythm_slope",
    "variability_first_half",
    "variability_second_half",
    "variability_change",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def extract_features_from_video(video_path: Path, processor: MediaPipeProcessor, calculator: MetricsCalculator) -> dict:
    landmark_frames = processor.process_video(
        str(video_path),
        skip_video_generation=True,
        resize_width=256,
        use_mediapipe_optimal=True,
    )
    frames_dict = [asdict(frame) for frame in landmark_frames]
    metrics = calculator.calculate_finger_tapping_metrics(frames_dict)
    return {field: getattr(metrics, field) for field in FEATURE_FIELDS}


def main() -> int:
    args = parse_args()
    processor = MediaPipeProcessor(mode="hand")
    calculator = MetricsCalculator(fps=30.0)

    rows = []
    with args.candidates.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    results = []
    for index, row in enumerate(rows, start=1):
        video_path = args.repo_root / row["video_path"]
        started = time.time()
        features = extract_features_from_video(video_path, processor, calculator)
        elapsed = time.time() - started
        results.append(
            {
                "candidate_id": row["candidate_id"],
                "task": row["task"],
                "video_id": row["video_id"],
                "subject_id": row["subject_id"],
                "score": int(row["score"]),
                "video_path": row["video_path"],
                "elapsed_sec": round(elapsed, 3),
                **features,
            }
        )
        print(f"[{index}/{len(rows)}] {row['candidate_id']} ok ({elapsed:.1f}s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print({"output": str(args.output), "rows": len(results)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
