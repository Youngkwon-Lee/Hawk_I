"""
Prepare PD4T data for HPC training
Run this locally to create pickle files, then transfer to HPC

Usage:
    python scripts/prepare_data.py
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
# Local paths (adjust as needed)
PD4T_ROOT = "C:/Users/YK/tulip/PD4T/PD4T/PD4T"
ANNOTATION_DIR = f"{PD4T_ROOT}/Annotations_split/Finger tapping"
VIDEO_DIR = f"{PD4T_ROOT}/Videos/Finger tapping"
OUTPUT_DIR = "./data"

SEQUENCE_LENGTH = 150  # Padded sequence length
NUM_LANDMARKS = 21
LANDMARK_DIM = 3


def parse_annotation_txt(txt_path):
    """Parse Annotations_split txt file"""
    annotations = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                video_id_full = parts[0]
                frames = int(parts[1])
                score = int(parts[2])

                parts_split = video_id_full.rsplit('_', 2)
                base_id = parts_split[0]
                hand = parts_split[1]
                subject = parts_split[2]

                video_path = f"{VIDEO_DIR}/{subject}/{base_id}_{hand}.mp4"

                annotations.append({
                    'video_id': video_id_full,
                    'base_id': base_id,
                    'hand': hand,
                    'subject': subject,
                    'frames': frames,
                    'score': score,
                    'video_path': video_path
                })
    return annotations


def extract_landmarks_mediapipe(video_path, max_frames=300):
    """Extract hand landmarks using MediaPipe"""
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
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
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            frame_landmarks = []
            for lm in hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(frame_landmarks)
        else:
            landmarks_list.append([0.0] * (NUM_LANDMARKS * LANDMARK_DIM))

        frame_count += 1

    cap.release()
    hands.close()

    return np.array(landmarks_list)


def add_velocity_features(landmarks):
    """Add velocity features to landmarks"""
    if len(landmarks) < 2:
        return landmarks

    velocity = np.diff(landmarks, axis=0)
    velocity = np.vstack([np.zeros((1, landmarks.shape[1])), velocity])

    thumb_vel = velocity[:, 12:15]
    index_vel = velocity[:, 24:27]
    thumb_speed = np.linalg.norm(thumb_vel, axis=1, keepdims=True)
    index_speed = np.linalg.norm(index_vel, axis=1, keepdims=True)

    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]
    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1, keepdims=True)

    features = np.hstack([landmarks, velocity, thumb_speed, index_speed, finger_distance])
    return features


def extract_clinical_features(landmarks):
    """Extract clinically relevant time-series features"""
    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]
    wrist_pos = landmarks[:, 0:3]

    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1)
    dist_velocity = np.gradient(finger_distance)
    dist_accel = np.gradient(dist_velocity)

    thumb_vel = landmarks[:, 63+12:63+15]
    thumb_speed = np.linalg.norm(thumb_vel, axis=1)

    index_vel = landmarks[:, 63+24:63+27]
    index_speed = np.linalg.norm(index_vel, axis=1)

    combined_speed = thumb_speed + index_speed

    thumb_from_wrist = np.linalg.norm(thumb_pos - wrist_pos, axis=1)
    index_from_wrist = np.linalg.norm(index_pos - wrist_pos, axis=1)

    hand_size = np.maximum(thumb_from_wrist, index_from_wrist) + 0.001
    normalized_distance = finger_distance / hand_size

    features = np.stack([
        finger_distance,
        dist_velocity,
        dist_accel,
        thumb_speed,
        index_speed,
        combined_speed,
        thumb_from_wrist,
        index_from_wrist,
        normalized_distance,
        hand_size,
    ], axis=1)

    return features


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
            continue

        try:
            # Extract landmarks
            landmarks = extract_landmarks_mediapipe(video_path)
            if len(landmarks) < 10:
                continue

            # Add velocity
            features = add_velocity_features(landmarks)

            # Extract clinical features
            clinical = extract_clinical_features(features)

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
    print("PD4T Data Preparation for HPC")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load annotations
    print("\nLoading annotations from Annotations_split...")
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
        'ids': ids_train + ids_valid
    }
    with open(f"{OUTPUT_DIR}/train_data.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    print(f"  Saved: {OUTPUT_DIR}/train_data.pkl")

    test_data = {
        'X': X_test,
        'y': y_test,
        'ids': ids_test
    }
    with open(f"{OUTPUT_DIR}/test_data.pkl", 'wb') as f:
        pickle.dump(test_data, f)
    print(f"  Saved: {OUTPUT_DIR}/test_data.pkl")

    # Summary
    print(f"\n{'='*60}")
    print("Data Preparation Complete!")
    print(f"{'='*60}")
    print(f"Train+Valid: {X_trainval.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Feature shape: {X_trainval.shape[1:]} (seq_len, features)")
    print(f"\nLabel distribution (Train+Valid):")
    for i in range(5):
        count = (y_trainval == i).sum()
        if count > 0:
            print(f"  UPDRS {i}: {count} ({count/len(y_trainval)*100:.1f}%)")

    print(f"\nFiles ready for HPC transfer:")
    print(f"  - {OUTPUT_DIR}/train_data.pkl")
    print(f"  - {OUTPUT_DIR}/test_data.pkl")


if __name__ == "__main__":
    main()
