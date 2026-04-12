"""
Prepare PD4T Finger Tapping data with Raw 3D + Clinical features
Run locally to create pickle files, then transfer to HPC

Features: 73 per frame (63 raw 3D + 10 clinical)
- Raw 3D: 21 landmarks × 3 coordinates = 63
- Clinical: finger_distance, dist_velocity, dist_accel, thumb_speed,
           index_speed, combined_speed, thumb_from_wrist, index_from_wrist,
           normalized_distance, hand_size

Usage:
    python scripts/prepare_data_v2.py
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
ANNOTATION_DIR = f"{PD4T_ROOT}/Annotations_split/Finger tapping"
VIDEO_DIR = f"{PD4T_ROOT}/Videos/Finger tapping"
OUTPUT_DIR = "./data"

SEQUENCE_LENGTH = 150  # Padded sequence length
NUM_LANDMARKS = 21     # MediaPipe Hands
LANDMARK_DIM = 3       # x, y, z
NUM_CLINICAL = 10      # Clinical features
TOTAL_FEATURES = NUM_LANDMARKS * LANDMARK_DIM + NUM_CLINICAL  # 63 + 10 = 73


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


def extract_clinical_features(landmarks):
    """
    Extract clinically relevant features from raw landmarks

    Key landmarks:
    - THUMB_TIP (4): indices 12-14
    - INDEX_TIP (8): indices 24-26
    - WRIST (0): indices 0-2
    """
    n_frames = len(landmarks)
    if n_frames < 2:
        return np.zeros((n_frames, NUM_CLINICAL))

    # Extract key positions
    thumb_pos = landmarks[:, 12:15]  # THUMB_TIP (landmark 4)
    index_pos = landmarks[:, 24:27]  # INDEX_TIP (landmark 8)
    wrist_pos = landmarks[:, 0:3]    # WRIST (landmark 0)

    # 1. Finger distance (thumb tip to index tip)
    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1)

    # 2. Distance velocity (rate of change)
    dist_velocity = np.gradient(finger_distance)

    # 3. Distance acceleration
    dist_accel = np.gradient(dist_velocity)

    # 4-5. Thumb and index speeds (using frame-to-frame displacement)
    thumb_diff = np.diff(thumb_pos, axis=0)
    thumb_speed = np.linalg.norm(thumb_diff, axis=1)
    thumb_speed = np.concatenate([[0], thumb_speed])

    index_diff = np.diff(index_pos, axis=0)
    index_speed = np.linalg.norm(index_diff, axis=1)
    index_speed = np.concatenate([[0], index_speed])

    # 6. Combined speed
    combined_speed = thumb_speed + index_speed

    # 7-8. Distance from wrist
    thumb_from_wrist = np.linalg.norm(thumb_pos - wrist_pos, axis=1)
    index_from_wrist = np.linalg.norm(index_pos - wrist_pos, axis=1)

    # 9-10. Normalized distance and hand size
    hand_size = np.maximum(thumb_from_wrist, index_from_wrist) + 1e-6
    normalized_distance = finger_distance / hand_size

    # Stack all clinical features
    clinical = np.stack([
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

    return clinical


def combine_raw_and_clinical(landmarks, clinical):
    """Combine raw 3D landmarks with clinical features"""
    # landmarks: (frames, 63) - raw 3D coordinates
    # clinical: (frames, 10) - clinical features
    return np.hstack([landmarks, clinical])  # (frames, 73)


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
            landmarks = extract_landmarks_mediapipe(video_path)
            if len(landmarks) < 10:
                skipped += 1
                continue

            # Extract clinical features
            clinical = extract_clinical_features(landmarks)

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
    print("PD4T Finger Tapping - Raw 3D + Clinical Features")
    print(f"Features per frame: {TOTAL_FEATURES} (63 raw + 10 clinical)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load annotations
    print("\nLoading annotations from Annotations_split/Finger tapping...")
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

    feature_names = [
        # Raw 3D (63)
        'wrist_x', 'wrist_y', 'wrist_z',
        'thumb_cmc_x', 'thumb_cmc_y', 'thumb_cmc_z',
        'thumb_mcp_x', 'thumb_mcp_y', 'thumb_mcp_z',
        'thumb_ip_x', 'thumb_ip_y', 'thumb_ip_z',
        'thumb_tip_x', 'thumb_tip_y', 'thumb_tip_z',
        'index_mcp_x', 'index_mcp_y', 'index_mcp_z',
        'index_pip_x', 'index_pip_y', 'index_pip_z',
        'index_dip_x', 'index_dip_y', 'index_dip_z',
        'index_tip_x', 'index_tip_y', 'index_tip_z',
        'middle_mcp_x', 'middle_mcp_y', 'middle_mcp_z',
        'middle_pip_x', 'middle_pip_y', 'middle_pip_z',
        'middle_dip_x', 'middle_dip_y', 'middle_dip_z',
        'middle_tip_x', 'middle_tip_y', 'middle_tip_z',
        'ring_mcp_x', 'ring_mcp_y', 'ring_mcp_z',
        'ring_pip_x', 'ring_pip_y', 'ring_pip_z',
        'ring_dip_x', 'ring_dip_y', 'ring_dip_z',
        'ring_tip_x', 'ring_tip_y', 'ring_tip_z',
        'pinky_mcp_x', 'pinky_mcp_y', 'pinky_mcp_z',
        'pinky_pip_x', 'pinky_pip_y', 'pinky_pip_z',
        'pinky_dip_x', 'pinky_dip_y', 'pinky_dip_z',
        'pinky_tip_x', 'pinky_tip_y', 'pinky_tip_z',
        # Clinical (10)
        'finger_distance', 'dist_velocity', 'dist_accel',
        'thumb_speed', 'index_speed', 'combined_speed',
        'thumb_from_wrist', 'index_from_wrist',
        'normalized_distance', 'hand_size',
    ]

    for split_name, X, y, ids in [
        ('train', X_train, y_train, ids_train),
        ('valid', X_valid, y_valid, ids_valid),
        ('test', X_test, y_test, ids_test)
    ]:
        data = {
            'X': X,
            'y': y,
            'ids': ids,
            'task': 'finger_tapping',
            'version': 'v2_raw3d_clinical',
            'features': feature_names,
            'feature_groups': {
                'raw_3d': list(range(0, 63)),
                'clinical': list(range(63, 73))
            }
        }
        filename = f"{OUTPUT_DIR}/finger_{split_name}_v2.pkl"
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
    print(f"  - Raw 3D: 63 (21 landmarks × 3)")
    print(f"  - Clinical: 10")
    print(f"\nLabel distribution (Train):")
    for i in range(5):
        count = (y_train == i).sum()
        if count > 0:
            print(f"  UPDRS {i}: {count} ({count/len(y_train)*100:.1f}%)")

    print(f"\nFiles ready for HPC transfer:")
    print(f"  - {OUTPUT_DIR}/finger_train_v2.pkl")
    print(f"  - {OUTPUT_DIR}/finger_valid_v2.pkl")
    print(f"  - {OUTPUT_DIR}/finger_test_v2.pkl")


if __name__ == "__main__":
    main()
