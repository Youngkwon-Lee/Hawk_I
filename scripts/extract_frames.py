"""
Extract frames from PD4T videos for visualization
Creates skeleton annotations on extracted frames
"""

import cv2
import mediapipe as mp
import os
from pathlib import Path

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_frame_with_skeleton(video_path, output_dir, frame_indices=[10, 30, 50]):
    """Extract specific frames from video with skeleton overlay"""

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust frame indices if video is shorter
    frame_indices = [f for f in frame_indices if f < total_frames]

    frames_saved = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:

        for frame_idx, target_frame in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()

            if not ret:
                continue

            # Resize if too large
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

            # Process pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Draw skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Add frame number
            cv2.putText(frame, f'Frame: {target_frame}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save frame
            output_path = os.path.join(output_dir, f"frame_{frame_idx:02d}.jpg")
            cv2.imwrite(output_path, frame)
            frames_saved.append(output_path)

    cap.release()
    return frames_saved

def main():
    output_root = "results/sample_frames"
    os.makedirs(output_root, exist_ok=True)

    # Sample videos
    samples = {
        "Gait": "data/raw/PD4T/PD4T/PD4T/Videos/Gait/001/15-001760.mp4",
        "Finger_Tapping": "data/raw/PD4T/PD4T/PD4T/Videos/Finger tapping/001/12-104705_r.mp4",
        "Hand_Movement": "data/raw/PD4T/PD4T/PD4T/Videos/Hand movement/001/13-007887_r.mp4",
        "Leg_Agility": "data/raw/PD4T/PD4T/PD4T/Videos/Leg agility/001/15-004054_r.mp4",
    }

    for task_name, video_path in samples.items():
        # Find video if doesn't exist
        if not os.path.exists(video_path):
            base_task = task_name.replace("_", " ")
            task_dir = f"data/raw/PD4T/PD4T/PD4T/Videos/{base_task}"

            videos = list(Path(task_dir).glob("*.mp4"))
            if videos:
                video_path = str(videos[0])
                print(f"Using: {video_path}")
            else:
                print(f"Skipping {task_name} - no video found")
                continue

        print(f"\nProcessing {task_name}...")
        task_output = os.path.join(output_root, task_name)
        os.makedirs(task_output, exist_ok=True)

        try:
            frames = extract_frame_with_skeleton(video_path, task_output, frame_indices=[5, 15, 25])
            print(f"  Saved {len(frames)} frames")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
