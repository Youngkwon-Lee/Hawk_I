"""
Create skeleton visualization videos from PD4T dataset
Generates short clips with MediaPipe skeleton overlays for presentation
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def visualize_skeleton(input_video_path, output_video_path, max_frames=120):
    """
    Extract and visualize skeleton from video
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output skeleton video
        max_frames: Limit frames for faster processing
    """
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Limit frames
    frames_to_process = min(max_frames, total_frames)

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process pose
            results = pose.process(rgb_frame)

            # Draw skeleton on frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Add text annotation
            cv2.putText(frame, f'Frame: {frame_count + 1}/{frames_to_process}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write frame
            out.write(frame)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{frames_to_process} frames")

    cap.release()
    out.release()
    print(f"✓ Saved to {output_video_path}")

def main():
    """Create skeleton visualization for sample videos"""

    pd4t_root = "data/raw/PD4T/PD4T/PD4T/Videos"
    output_root = "results/skeleton_videos"
    os.makedirs(output_root, exist_ok=True)

    # Sample videos from each task
    samples = {
        "Gait": "data/raw/PD4T/PD4T/PD4T/Videos/Gait/001/15-001760.mp4",
        "Finger_Tapping": "data/raw/PD4T/PD4T/PD4T/Videos/Finger tapping/001/12-104705_r.mp4",
        "Hand_Movement": "data/raw/PD4T/PD4T/PD4T/Videos/Hand movement/001/13-007887_r.mp4",
        "Leg_Agility": "data/raw/PD4T/PD4T/PD4T/Videos/Leg agility/001/15-004054_r.mp4",
    }

    for task_name, video_path in samples.items():
        if not os.path.exists(video_path):
            print(f"⚠ Video not found: {video_path}")
            # Try to find alternative
            task_dir = os.path.dirname(video_path)
            videos = list(Path(task_dir).glob("*.mp4"))
            if videos:
                video_path = str(videos[0])
                print(f"  Using alternative: {video_path}")
            else:
                print(f"✗ No videos found for {task_name}")
                continue

        output_path = os.path.join(output_root, f"{task_name}_skeleton.mp4")

        print(f"\nProcessing {task_name}...")
        print(f"  Input: {video_path}")

        try:
            visualize_skeleton(video_path, output_path, max_frames=120)
        except Exception as e:
            print(f"✗ Error processing {task_name}: {e}")

if __name__ == "__main__":
    print("="*60)
    print("Skeleton Visualization Video Generator")
    print("="*60)
    main()
    print("\n" + "="*60)
    print("Complete! Check results/skeleton_videos/")
    print("="*60)
