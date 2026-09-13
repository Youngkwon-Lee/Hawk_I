"""Build the privacy-safe HawkEye filming demo from a licensed public clip.

The input must be a non-patient video that is approved for reuse. The script
creates a browser-ready skeleton overlay and chart data derived from the same
frames so the comparison screen never substitutes unrelated mock values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.mediapipe_processor import MediaPipeProcessor  # noqa: E402
from services.visualization_data_generator import VisualizationDataGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.overlay.parent.mkdir(parents=True, exist_ok=True)
    args.data.parent.mkdir(parents=True, exist_ok=True)

    processor = MediaPipeProcessor(mode="pose")
    frames = processor.process_video(
        str(args.input),
        output_video_path=str(args.overlay),
        frame_skip=1,
        max_duration=5.0,
        resize_width=406,
        use_mediapipe_optimal=False,
    )
    if len(frames) < 10:
        raise RuntimeError("Pose extraction returned too few frames")

    serialized_frames = [
        {
            "frame": frame.frame_number,
            "timestamp": round(frame.timestamp, 3),
            "keypoints": frame.landmarks,
        }
        for frame in frames
    ]
    visualization = VisualizationDataGenerator(fps=24.0).generate(
        serialized_frames,
        task_type="gait",
    )
    payload = {
        "source": "Pexels video 9616883; public filming demo only",
        "duration": 5.0,
        "fps": 24,
        "detection_rate": round(len(frames) / 120, 3),
        "visualization_data": visualization,
    }
    args.data.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "frames": len(frames),
                "joint_angle_points": len(visualization["joint_angles"]),
                "speed_points": len(visualization["speed_profile"]),
                "overlay": str(args.overlay),
                "data": str(args.data),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
