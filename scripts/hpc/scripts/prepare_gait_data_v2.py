"""
Prepare PD4T Gait data with Raw 3D + Clinical features
Run locally to create pickle files, then transfer to HPC

Features: 129 per frame (99 raw 3D + 30 clinical)
- Raw 3D: 33 landmarks × 3 coordinates = 99
- Clinical: 15 position + 15 velocity features

Usage:
    python scripts/prepare_gait_data_v2.py
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
NUM_LANDMARKS = 33     # MediaPipe Pose
LANDMARK_DIM = 3       # x, y, z
NUM_CLINICAL = 30      # 15 position + 15 velocity
TOTAL_FEATURES = NUM_LANDMARKS * LANDMARK_DIM + NUM_CLINICAL  # 99 + 30 = 129


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
    - 15, 16: Wrists
    - 23, 24: Hips (left, right)
    - 25, 26: Knees (left, right)
    - 27, 28: Ankles (left, right)
    """
    n_frames = len(landmarks)
    if n_frames < 2:
        return np.zeros((n_frames, NUM_CLINICAL))

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
        left_wrist = get_joint(frame, 15)
        right_wrist = get_joint(frame, 16)

        # 1. Step width (distance between ankles in x)
        step_width = abs(left_ankle[0] - right_ankle[0])

        # 2. Hip center height
        hip_center = (left_hip + right_hip) / 2
        hip_height = hip_center[1]

        # 3. Trunk angle
        shoulder_center = (left_shoulder + right_shoulder) / 2
        trunk_vector = hip_center - shoulder_center
        trunk_angle = np.arctan2(trunk_vector[0], trunk_vector[1])

        # 4-5. Knee angles
        left_thigh = left_knee - left_hip
        left_shin = left_ankle - left_knee
        left_knee_angle = np.arccos(np.clip(
            np.dot(left_thigh, left_shin) / (np.linalg.norm(left_thigh) * np.linalg.norm(left_shin) + 1e-6),
            -1, 1
        ))

        right_thigh = right_knee - right_hip
        right_shin = right_ankle - right_knee
        right_knee_angle = np.arccos(np.clip(
            np.dot(right_thigh, right_shin) / (np.linalg.norm(right_thigh) * np.linalg.norm(right_shin) + 1e-6),
            -1, 1
        ))

        # 6-7. Ankle heights
        left_ankle_height = left_ankle[1]
        right_ankle_height = right_ankle[1]

        # 8-9. Arm swing
        left_arm_swing = np.linalg.norm(left_wrist - left_shoulder)
        right_arm_swing = np.linalg.norm(right_wrist - right_shoulder)

        # 10. Body sway
        body_sway = hip_center[0]

        # 11. Stride proxy
        stride_proxy = abs(left_ankle[0] - right_ankle[0])

        # 12-13. Asymmetry
        hip_asymmetry = abs(left_hip[1] - right_hip[1])
        shoulder_asymmetry = abs(left_shoulder[1] - right_shoulder[1])

        # 14-15. Hip angles
        left_hip_angle = np.arccos(np.clip(
            np.dot(-trunk_vector, left_thigh) / (np.linalg.norm(trunk_vector) * np.linalg.norm(left_thigh) + 1e-6),
            -1, 1
        ))
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
    clinical = np.hstack([features, velocity])  # 15 + 15 = 30

    return clinical


def combine_raw_and_clinical(landmarks, clinical):
    """Combine raw 3D landmarks with clinical features"""
    # landmarks: (frames, 99) - raw 3D coordinates
    # clinical: (frames, 30) - clinical features
    return np.hstack([landmarks, clinical])  # (frames, 129)


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
    skipped = 0

    for ann in tqdm(annotations, desc=split_name):
        video_path = ann['video_path']

        if not os.path.exists(video_path):
            skipped += 1
            continue

        try:
            # Extract raw 3D landmarks
            landmarks = extract_landmarks_mediapipe_pose(video_path)
            if len(landmarks) < 30:
                skipped += 1
                continue

            # Extract clinical features
            clinical = extract_gait_clinical_features(landmarks)

            # Combine raw + clinical
            combined = combine_raw_and_clinical(landmarks, clinical)

            # Pad/truncate to fixed length
            padded = pad_sequence(combined, SEQUENCE_LENGTH)

            all_features.append(padded)
            all_scores.append(ann['score'])
            all_ids.append(ann['video_id'])

        except Exception as e:
            print(f"Error {ann['video_id']}: {e}")
            skipped += 1
            continue

    X = np.array(all_features)
    y = np.array(all_scores)

    print(f"  Processed: {len(X)} samples, Skipped: {skipped}")
    print(f"  Shape: {X.shape} (samples, frames, features)")

    return X, y, all_ids


def main():
    print("=" * 60)
    print("PD4T Gait - Raw 3D + Clinical Features")
    print(f"Features per frame: {TOTAL_FEATURES} (99 raw + 30 clinical)")
    print("=" * 60)

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

    # Save each split separately
    print("\nSaving data...")

    # Generate feature names
    pose_landmarks = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    raw_feature_names = []
    for lm in pose_landmarks:
        raw_feature_names.extend([f'{lm}_x', f'{lm}_y', f'{lm}_z'])

    clinical_position = [
        'step_width', 'hip_height', 'trunk_angle',
        'left_knee_angle', 'right_knee_angle',
        'left_ankle_height', 'right_ankle_height',
        'left_arm_swing', 'right_arm_swing',
        'body_sway', 'stride_proxy',
        'hip_asymmetry', 'shoulder_asymmetry',
        'left_hip_angle', 'right_hip_angle'
    ]
    clinical_velocity = [f'{f}_vel' for f in clinical_position]

    feature_names = raw_feature_names + clinical_position + clinical_velocity

    for split_name, X, y, ids in [
        ('train', X_train, y_train, ids_train),
        ('valid', X_valid, y_valid, ids_valid),
        ('test', X_test, y_test, ids_test)
    ]:
        data = {
            'X': X,
            'y': y,
            'ids': ids,
            'task': 'gait',
            'version': 'v2_raw3d_clinical',
            'features': feature_names,
            'feature_groups': {
                'raw_3d': list(range(0, 99)),
                'clinical_position': list(range(99, 114)),
                'clinical_velocity': list(range(114, 129))
            }
        }
        filename = f"{OUTPUT_DIR}/gait_{split_name}_v2.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved: {filename}")

    # Summary
    print(f"\n{'='*60}")
    print("Extraction Complete!")
    print(f"{'='*60}")
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Valid: {X_valid.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Feature shape: {X_train.shape[1:]} (seq_len, features)")
    print(f"  - Raw 3D: 99 (33 landmarks × 3)")
    print(f"  - Clinical Position: 15")
    print(f"  - Clinical Velocity: 15")
    print(f"\nLabel distribution (Train):")
    for i in range(5):
        count = (y_train == i).sum()
        if count > 0:
            print(f"  UPDRS {i}: {count} ({count/len(y_train)*100:.1f}%)")

    print(f"\nFiles ready for HPC transfer:")
    print(f"  - {OUTPUT_DIR}/gait_train_v2.pkl")
    print(f"  - {OUTPUT_DIR}/gait_valid_v2.pkl")
    print(f"  - {OUTPUT_DIR}/gait_test_v2.pkl")


if __name__ == "__main__":
    main()
