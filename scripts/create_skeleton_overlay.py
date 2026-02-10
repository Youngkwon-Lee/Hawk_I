#!/usr/bin/env python3
"""
Generate skeleton overlay frames with MediaPipe Hands detection
"""
import cv2
import mediapipe as mp
import os

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Video paths
video_files = {
    'Gait': 'data/raw/PD4T/PD4T/PD4T/Videos/Gait/001/12-104705.mp4',
    'Finger_Tapping': 'data/raw/PD4T/PD4T/PD4T/Videos/Finger tapping/001/12-104705_r.mp4',
    'Hand_Movement': 'data/raw/PD4T/PD4T/PD4T/Videos/Hand movement/001/12-104704_l.mp4',
    'Leg_Agility': 'data/raw/PD4T/PD4T/PD4T/Videos/Leg agility/001/12-104704_l.mp4'
}

output_dir = 'results/skeleton_overlays'
os.makedirs(output_dir, exist_ok=True)

# Process each video
for task_name, video_path in video_files.items():
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        continue

    print(f"Processing {task_name}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Failed to open video")
        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame_idx = int(frame_count / 3)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()

    if not ret:
        print(f"  Failed to read frame")
        cap.release()
        continue

    # Resize for display
    frame = cv2.resize(frame, (960, 720))
    h, w, c = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detected_hands = False
    detected_pose = False

    # Detect hand landmarks
    if task_name in ['Finger_Tapping', 'Hand_Movement']:
        print(f"  Detecting hand landmarks...")
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Green joints
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Red connections
                )
            detected_hands = True
            print(f"  Detected {len(hand_results.multi_hand_landmarks)} hand(s)")

    # Detect full body pose landmarks
    else:  # Gait, Leg_Agility
        print(f"  Detecting body pose landmarks...")
        pose_results = pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Green joints
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Red connections
            )
            detected_pose = True
            print(f"  Detected body pose: {len(pose_results.pose_landmarks.landmark)} landmarks")

    # Add task label
    if detected_hands or detected_pose:
        if detected_hands:
            label = f'{task_name} - Hand Skeleton Overlay'
        else:
            label = f'{task_name} - Body Pose Skeleton Overlay'
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
    else:
        label = f'{task_name} - No skeleton detected'
        cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # Save frame
    output_path = os.path.join(output_dir, f'{task_name}_skeleton.jpg')
    success = cv2.imwrite(output_path, frame)
    if success:
        print(f"  Saved: {output_path}\n")
    else:
        print(f"  Failed to save!\n")

    cap.release()

print("Done! Skeleton overlays generated with MediaPipe.")
print(f"Location: {output_dir}/")
print("\nDetection method:")
print("  - Finger_Tapping, Hand_Movement: MediaPipe Hands (21 landmarks per hand)")
print("  - Gait, Leg_Agility: MediaPipe Pose (33 body landmarks)")
