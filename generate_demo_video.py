"""
Generate Demo Video with Skeleton Overlay and Real-time Metrics Display
Outputs video with:
- MediaPipe pose skeleton overlay
- Real-time metrics display (speed, stride, arm swing, step height)
- UPDRS score prediction
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import sys

sys.path.insert(0, 'backend')

from services.mediapipe_processor import MediaPipeProcessor
from services.metrics_calculator import MetricsCalculator, GaitMetrics
from services.updrs_scorer import UPDRSScorer


def draw_metrics_overlay(frame, metrics: GaitMetrics, updrs_score: float, frame_num: int, fps: float):
    """Draw metrics overlay on frame"""
    height, width = frame.shape[:2]

    # Semi-transparent background for metrics
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Colors
    white = (255, 255, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    cyan = (255, 255, 0)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22

    y = 35

    # Title
    cv2.putText(frame, "Gait Analysis - World Landmarks", (20, y), font, 0.6, cyan, 2)
    y += line_height + 5

    # Frame info
    time_sec = frame_num / fps
    cv2.putText(frame, f"Time: {time_sec:.1f}s | Frame: {frame_num}", (20, y), font, font_scale, white, thickness)
    y += line_height

    # UPDRS Score (larger)
    score_color = green if updrs_score <= 1 else (yellow if updrs_score <= 2 else red)
    cv2.putText(frame, f"UPDRS Score: {updrs_score:.1f}", (20, y), font, 0.7, score_color, 2)
    y += line_height + 5

    # Separator
    cv2.line(frame, (20, y), (380, y), white, 1)
    y += 10

    # Primary Metrics
    cv2.putText(frame, f"Walking Speed: {metrics.walking_speed:.2f} m/s", (20, y), font, font_scale, white, thickness)
    y += line_height
    cv2.putText(frame, f"Stride Length: {metrics.stride_length:.2f} m", (20, y), font, font_scale, white, thickness)
    y += line_height
    cv2.putText(frame, f"Cadence: {metrics.cadence:.1f} steps/min", (20, y), font, font_scale, white, thickness)
    y += line_height + 5

    # Arm Swing (World Landmarks)
    cv2.putText(frame, "Arm Swing (3D):", (20, y), font, font_scale, yellow, thickness)
    y += line_height
    cv2.putText(frame, f"  L: {metrics.arm_swing_amplitude_left*100:.1f}cm  R: {metrics.arm_swing_amplitude_right*100:.1f}cm",
                (20, y), font, font_scale, white, thickness)
    y += line_height
    asym_color = green if metrics.arm_swing_asymmetry < 15 else (yellow if metrics.arm_swing_asymmetry < 30 else red)
    cv2.putText(frame, f"  Asymmetry: {metrics.arm_swing_asymmetry:.1f}%", (20, y), font, font_scale, asym_color, thickness)
    y += line_height + 5

    # Step Height (World Landmarks)
    cv2.putText(frame, "Step Height (3D):", (20, y), font, font_scale, yellow, thickness)
    y += line_height
    cv2.putText(frame, f"  L: {metrics.step_height_left*100:.1f}cm  R: {metrics.step_height_right*100:.1f}cm",
                (20, y), font, font_scale, white, thickness)
    y += line_height
    height_color = green if metrics.step_height_mean > 0.05 else (yellow if metrics.step_height_mean > 0.03 else red)
    cv2.putText(frame, f"  Mean: {metrics.step_height_mean*100:.1f}cm", (20, y), font, font_scale, height_color, thickness)

    return frame


def process_video_with_overlay(video_path: str, output_path: str, max_frames: int = 300):
    """Process video and create overlay with metrics"""

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Processing first {max_frames} frames...")

    # Collect landmarks first for metrics calculation
    landmark_frames = []
    frames_data = []

    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Extract normalized landmarks
            landmarks = []
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                landmarks.append({
                    'id': idx,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })

            # Extract world landmarks
            world_landmarks = None
            if results.pose_world_landmarks:
                world_landmarks = []
                for idx, lm in enumerate(results.pose_world_landmarks.landmark):
                    world_landmarks.append({
                        'id': idx,
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })

            landmark_frames.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'keypoints': landmarks,
                'world_keypoints': world_landmarks
            })

        frames_data.append((frame.copy(), results))
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"  Collected {frame_count} frames...")

    cap.release()

    print(f"  Total frames with landmarks: {len(landmark_frames)}")

    # Calculate metrics
    if len(landmark_frames) < 30:
        print("  ERROR: Not enough frames for analysis")
        return None

    calculator = MetricsCalculator(fps=fps)
    scorer = UPDRSScorer()

    try:
        metrics = calculator.calculate_gait_metrics(landmark_frames)
        result = scorer.score_gait(metrics)
        updrs_score = result.total_score
        print(f"\n  UPDRS Score: {updrs_score:.1f}")
        print(f"  Walking Speed: {metrics.walking_speed:.2f} m/s")
        print(f"  Stride Length: {metrics.stride_length:.2f} m")
        print(f"  Arm Swing Asymmetry: {metrics.arm_swing_asymmetry:.1f}%")
        print(f"  Step Height Mean: {metrics.step_height_mean*100:.1f} cm")
    except Exception as e:
        print(f"  ERROR calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n  Creating overlay video...")

    for i, (frame, results) in enumerate(frames_data):
        # Draw skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Draw metrics overlay
        frame = draw_metrics_overlay(frame, metrics, updrs_score, i, fps)

        out.write(frame)

        if i % 30 == 0:
            print(f"    Rendered {i}/{len(frames_data)} frames...")

    out.release()
    pose.close()

    # Convert to H.264 for browser compatibility
    try:
        import subprocess
        temp_path = output_path.replace('.mp4', '_temp.mp4')
        cmd = [
            'ffmpeg', '-i', output_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-y',
            temp_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(temp_path):
            os.replace(temp_path, output_path)
            print(f"  Converted to H.264")
    except Exception as e:
        print(f"  Note: ffmpeg conversion skipped ({e})")

    print(f"\n  Output saved: {output_path}")
    return output_path


def main():
    video_base = 'C:/Users/YK/tulip/PD4T/PD4T/PD4T/Videos/Gait'
    output_dir = 'C:/Users/YK/tulip/Hawkeye/demo_videos'

    os.makedirs(output_dir, exist_ok=True)

    # Test videos for each UPDRS score
    test_videos = {
        1: '14-005690',
        2: '15-003012',
        3: '13-007586',
    }

    def find_video(video_name):
        for subdir in os.listdir(video_base):
            subpath = os.path.join(video_base, subdir)
            if os.path.isdir(subpath):
                for f in os.listdir(subpath):
                    if video_name in f and f.endswith('.mp4'):
                        return os.path.join(subpath, f)
        return None

    print("\n" + "="*60)
    print("Demo Video Generator - Skeleton Overlay + Metrics")
    print("="*60)

    for expected_score, video_name in test_videos.items():
        video_path = find_video(video_name)
        if not video_path:
            print(f"\nScore {expected_score}: Video not found for {video_name}")
            continue

        output_path = os.path.join(output_dir, f"score_{expected_score}_{video_name}_demo.mp4")
        process_video_with_overlay(video_path, output_path, max_frames=300)

    print("\n" + "="*60)
    print(f"Demo videos saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
