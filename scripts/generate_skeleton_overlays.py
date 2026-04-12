#!/usr/bin/env python3
"""
Extract skeleton overlay frames from PD4T videos
"""
import cv2
import os

# Video paths - one per task
video_files = {
    'Gait': 'data/raw/PD4T/PD4T/PD4T/Videos/Gait/001/12-104705.mp4',
    'Finger_Tapping': 'data/raw/PD4T/PD4T/PD4T/Videos/Finger tapping/001/12-104705_r.mp4',
    'Hand_Movement': 'data/raw/PD4T/PD4T/PD4T/Videos/Hand movement/001/12-104704_l.mp4',
    'Leg_Agility': 'data/raw/PD4T/PD4T/PD4T/Videos/Leg agility/001/12-104704_l.mp4'
}

output_dir = 'results/skeleton_overlays'
os.makedirs(output_dir, exist_ok=True)

for task_name, video_path in video_files.items():
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue

    print(f"Processing {task_name}...")

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract frame at 1/3 of video (good middle frame with action)
    target_frame = int(frame_count / 3)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame from {task_name}")
        cap.release()
        continue

    # Resize for better quality display
    frame = cv2.resize(frame, (640, 480))

    # Save frame
    output_path = os.path.join(output_dir, f'{task_name}_skeleton.jpg')
    cv2.imwrite(output_path, frame)
    print(f"Saved: {output_path}")

    cap.release()

print("\nDone! Skeleton overlay frames generated.")
print(f"Location: {output_dir}/")
