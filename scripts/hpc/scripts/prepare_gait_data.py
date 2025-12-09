"""
Prepare PD4T Gait data for HPC training
Run this locally to create pickle files, then transfer to HPC

Usage:
    python scripts/prepare_gait_data.py
"""
import os
import sys
import cv2
import numpy as np
import pickle
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
PD4T_ROOT = "C:/Users/YK/tulip/PD4T/PD4T/PD4T"
ANNOTATION_DIR = f"{PD4T_ROOT}/Annotations_split/Gait"
VIDEO_DIR = f"{PD4T_ROOT}/Videos/Gait"
OUTPUT_DIR = "./data"

SEQUENCE_LENGTH = 300  # Gait videos are longer
NUM_LANDMARKS = 33  # MediaPipe Pose
LANDMARK_DIM = 3


def parse_annotation_txt(txt_path):
    """Parse Annotations_split txt file for Gait"""
    annotations = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                video_id_full = parts[0]  # e.g., "15-001760_009"
                frames = int(parts[1])
                score = int(parts[2])

                # Gait format: {base_id}_{subject}
                parts_split = video_id_full.rsplit('_', 1)
                base_id = parts_split[0]
                subject = parts_split[1]

                video_path = f"{VIDEO_DIR}/{subject}/{base_id}.mp4"

                annotations.append({
                    'video_id': video_id_full,
                    'base_id': base_id,
                    'subject': subject,
                    'frames': frames,
                    'score': score,
                    'video_path': video_path
                })
    return annotations


def extract_landmarks_mediapipe_pose(video_path, max_frames=600):
    """Extract pose landmarks using MediaPipe Pose"""
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            frame_landmarks = []
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(frame_landmarks)
        else:
            # No detection - use zeros
            landmarks_list.append([0.0] * (NUM_LANDMARKS * LANDMARK_DIM))

        frame_count += 1

    cap.release()
    pose.close()

    return np.array(landmarks_list)


def extract_gait_clinical_features(landmarks):
    """
    Extract clinically relevant gait features from pose landmarks

    Key landmarks for gait:
    - 11, 12: Shoulders (left, right)
    - 13, 14: Elbows
    - 15, 16: Wrists
    - 23, 24: Hips (left, right)
    - 25, 26: Knees (left, right)
    - 27, 28: Ankles (left, right)
    - 29, 30: Heels
    - 31, 32: Foot index (toes)
    """
    if len(landmarks) < 2:
        return landmarks

    # Reshape: (frames, 33*3) -> for easier indexing
    n_frames = len(landmarks)

    # Extract key joint positions (each joint has x, y, z)
    def get_joint(frame_data, joint_id):
        idx = joint_id * 3
        return np.array([frame_data[idx], frame_data[idx+1], frame_data[idx+2]])

    features_list = []

    for i in range(n_frames):
        frame = landmarks[i]

        # Joint positions
        left_shoulder = get_joint(frame, 11)
        right_shoulder = get_joint(frame, 12)
        left_hip = get_joint(frame, 23)
        right_hip = get_joint(frame, 24)
        left_knee = get_joint(frame, 25)
        right_knee = get_joint(frame, 26)
        left_ankle = get_joint(frame, 27)
        right_ankle = get_joint(frame, 28)
        left_heel = get_joint(frame, 29)
        right_heel = get_joint(frame, 30)
        left_toe = get_joint(frame, 31)
        right_toe = get_joint(frame, 32)
        left_wrist = get_joint(frame, 15)
        right_wrist = get_joint(frame, 16)

        # Calculate features
        # 1. Step width (distance between ankles in x)
        step_width = abs(left_ankle[0] - right_ankle[0])

        # 2. Hip center height (vertical position)
        hip_center = (left_hip + right_hip) / 2
        hip_height = hip_center[1]

        # 3. Trunk angle (shoulder to hip angle)
        shoulder_center = (left_shoulder + right_shoulder) / 2
        trunk_vector = hip_center - shoulder_center
        trunk_angle = np.arctan2(trunk_vector[0], trunk_vector[1])

        # 4. Left knee angle
        left_thigh = left_knee - left_hip
        left_shin = left_ankle - left_knee
        left_knee_angle = np.arccos(np.clip(
            np.dot(left_thigh, left_shin) / (np.linalg.norm(left_thigh) * np.linalg.norm(left_shin) + 1e-6),
            -1, 1
        ))

        # 5. Right knee angle
        right_thigh = right_knee - right_hip
        right_shin = right_ankle - right_knee
        right_knee_angle = np.arccos(np.clip(
            np.dot(right_thigh, right_shin) / (np.linalg.norm(right_thigh) * np.linalg.norm(right_shin) + 1e-6),
            -1, 1
        ))

        # 6. Left ankle height (for step detection)
        left_ankle_height = left_ankle[1]

        # 7. Right ankle height
        right_ankle_height = right_ankle[1]

        # 8. Arm swing - left (wrist to shoulder distance)
        left_arm_swing = np.linalg.norm(left_wrist - left_shoulder)

        # 9. Arm swing - right
        right_arm_swing = np.linalg.norm(right_wrist - right_shoulder)

        # 10. Body sway (hip center x position)
        body_sway = hip_center[0]

        # 11. Stride length proxy (ankle x difference)
        stride_proxy = abs(left_ankle[0] - right_ankle[0])

        # 12. Hip asymmetry
        hip_asymmetry = abs(left_hip[1] - right_hip[1])

        # 13. Shoulder asymmetry
        shoulder_asymmetry = abs(left_shoulder[1] - right_shoulder[1])

        # 14. Left hip angle
        left_hip_angle = np.arccos(np.clip(
            np.dot(-trunk_vector, left_thigh) / (np.linalg.norm(trunk_vector) * np.linalg.norm(left_thigh) + 1e-6),
            -1, 1
        ))

        # 15. Right hip angle
        right_hip_angle = np.arccos(np.clip(
            np.dot(-trunk_vector, right_thigh) / (np.linalg.norm(trunk_vector) * np.linalg.norm(right_thigh) + 1e-6),
            -1, 1
        ))

        frame_features = [
            step_width,
            hip_height,
            trunk_angle,
            left_knee_angle,
            right_knee_angle,
            left_ankle_height,
            right_ankle_height,
            left_arm_swing,
            right_arm_swing,
            body_sway,
            stride_proxy,
            hip_asymmetry,
            shoulder_asymmetry,
            left_hip_angle,
            right_hip_angle,
        ]

        features_list.append(frame_features)

    features = np.array(features_list)

    # Add velocity features
    velocity = np.diff(features, axis=0)
    velocity = np.vstack([np.zeros((1, features.shape[1])), velocity])

    # Combine position and velocity features
    combined = np.hstack([features, velocity])

    return combined


def pad_sequence(seq, target_len):
    """Pad or truncate sequence to target length"""
    if len(seq) >= target_len:
        return seq[:target_len]
    else:
        padded = np.zeros((target_len, seq.shape[1]))
        padded[:len(seq)] = seq
        return padded


def process_split(annotations, split_name):
    """Process all videos in a split"""
    print(f"\nProcessing {split_name}: {len(annotations)} videos")

    all_features = []
    all_scores = []
    all_ids = []

    for ann in tqdm(annotations, desc=split_name):
        video_path = ann['video_path']

        if not os.path.exists(video_path):
            print(f"  Skipped (not found): {video_path}")
            continue

        try:
            # Extract landmarks
            landmarks = extract_landmarks_mediapipe_pose(video_path)
            if len(landmarks) < 30:  # Need minimum frames
                continue

            # Extract clinical features
            clinical = extract_gait_clinical_features(landmarks)

            # Pad/truncate
            padded = pad_sequence(clinical, SEQUENCE_LENGTH)

            all_features.append(padded)
            all_scores.append(ann['score'])
            all_ids.append(ann['video_id'])

        except Exception as e:
            print(f"Error {ann['video_id']}: {e}")
            continue

    X = np.array(all_features)
    y = np.array(all_scores)

    print(f"  Processed: {len(X)} samples, shape: {X.shape}")

    return X, y, all_ids


def main():
    print("="*60)
    print("PD4T Gait Data Preparation for HPC")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load annotations
    print("\nLoading annotations from Annotations_split/Gait...")
    train_ann = parse_annotation_txt(f"{ANNOTATION_DIR}/train.txt")
    valid_ann = parse_annotation_txt(f"{ANNOTATION_DIR}/valid.txt")
    test_ann = parse_annotation_txt(f"{ANNOTATION_DIR}/test.txt")

    print(f"Train: {len(train_ann)}, Valid: {len(valid_ann)}, Test: {len(test_ann)}")

    # Process each split
    X_train, y_train, ids_train = process_split(train_ann, "train")
    X_valid, y_valid, ids_valid = process_split(valid_ann, "valid")
    X_test, y_test, ids_test = process_split(test_ann, "test")

    # Combine train + valid
    X_trainval = np.vstack([X_train, X_valid])
    y_trainval = np.concatenate([y_train, y_valid])

    print(f"\nCombined Train+Valid: {X_trainval.shape}")

    # Save as pickle
    print("\nSaving data...")

    train_data = {
        'X': X_trainval,
        'y': y_trainval,
        'ids': ids_train + ids_valid,
        'task': 'gait',
        'features': [
            'step_width', 'hip_height', 'trunk_angle',
            'left_knee_angle', 'right_knee_angle',
            'left_ankle_height', 'right_ankle_height',
            'left_arm_swing', 'right_arm_swing',
            'body_sway', 'stride_proxy',
            'hip_asymmetry', 'shoulder_asymmetry',
            'left_hip_angle', 'right_hip_angle',
            # Velocity features (same names with _vel suffix)
        ]
    }
    with open(f"{OUTPUT_DIR}/gait_train_data.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    print(f"  Saved: {OUTPUT_DIR}/gait_train_data.pkl")

    test_data = {
        'X': X_test,
        'y': y_test,
        'ids': ids_test,
        'task': 'gait'
    }
    with open(f"{OUTPUT_DIR}/gait_test_data.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    print(f"  Saved: {OUTPUT_DIR}/gait_test_data.pkl")

    # Summary
    print(f"\n{'='*60}")
    print("Gait Data Preparation Complete!")
    print(f"{'='*60}")
    print(f"Train+Valid: {X_trainval.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Feature shape: {X_trainval.shape[1:]} (seq_len, features)")
    print(f"Features per frame: 15 position + 15 velocity = 30")
    print(f"\nLabel distribution (Train+Valid):")
    for i in range(5):
        count = (y_trainval == i).sum()
        if count > 0:
            print(f"  UPDRS {i}: {count} ({count/len(y_trainval)*100:.1f}%)")

    print(f"\nFiles ready for HPC transfer:")
    print(f"  - {OUTPUT_DIR}/gait_train_data.pkl")
    print(f"  - {OUTPUT_DIR}/gait_test_data.pkl")


if __name__ == "__main__":
    main()
