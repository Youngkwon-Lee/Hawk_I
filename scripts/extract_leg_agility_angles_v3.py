"""
Extract Leg Agility Features with Joint Angles (v3)
Based on literature review: AI WALKUP, VisionMD, SA-GCN

Features:
1. Hip-Knee angle (flexion angle)
2. Knee-Ankle angle (extension angle)
3. Leg-to-vertical angle (elevation height)
4. Left/Right separation (symmetry analysis)
5. Decrementing amplitude (fatigue pattern)
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc  # For garbage collection

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.mediapipe_processor import MediaPipeProcessor

# Paths
PD4T_ROOT = "C:/Users/YK/tulip/Hawkeye/data/raw/PD4T/PD4T/PD4T"
VIDEO_ROOT = f"{PD4T_ROOT}/Videos/Leg agility"
ANNOTATION_ROOT = f"{PD4T_ROOT}/Annotations/Leg agility/stratified"
OUTPUT_DIR = "C:/Users/YK/tulip/Hawkeye/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_annotation(csv_path):
    """Parse annotation CSV"""
    annotations = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue

            video_id_subject = parts[0]
            score = int(parts[2])

            parts_split = video_id_subject.split('_')
            if len(parts_split) == 3:
                video_id = f"{parts_split[0]}_{parts_split[1]}"
                subject = parts_split[2]
            elif len(parts_split) == 2:
                video_id = parts_split[0]
                subject = parts_split[1]
            else:
                continue

            side = 'left' if video_id.endswith('_l') else 'right'

            annotations.append({
                'video_id': video_id,
                'subject': subject,
                'score': score,
                'side': side
            })

    return annotations


def calculate_angle(p1, p2, p3):
    """
    Calculate angle between three points (in degrees)
    p1, p2, p3: (x, y, z) coordinates
    Returns angle at p2
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def calculate_angle_to_vertical(p1, p2):
    """
    Calculate angle between vector (p1->p2) and vertical axis
    """
    vec = p2 - p1
    vertical = np.array([0, -1, 0])  # Vertical downward in image coordinates

    cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def extract_angle_features(landmarks):
    """
    Extract joint angle features from MediaPipe landmarks

    Landmarks indices:
    23: Left Hip, 24: Right Hip
    25: Left Knee, 26: Right Knee
    27: Left Ankle, 28: Right Ankle

    Args:
        landmarks: dict of {id: {'x', 'y', 'z', ...}} from MediaPipe

    Returns: dict of features
    """
    if landmarks is None or len(landmarks) < 29:
        return None

    # Check if all required landmarks exist
    required_ids = [23, 24, 25, 26, 27, 28]
    if not all(idx in landmarks for idx in required_ids):
        return None

    # Extract landmark coordinates
    left_hip = np.array([landmarks[23]['x'], landmarks[23]['y'], landmarks[23]['z']])
    right_hip = np.array([landmarks[24]['x'], landmarks[24]['y'], landmarks[24]['z']])
    left_knee = np.array([landmarks[25]['x'], landmarks[25]['y'], landmarks[25]['z']])
    right_knee = np.array([landmarks[26]['x'], landmarks[26]['y'], landmarks[26]['z']])
    left_ankle = np.array([landmarks[27]['x'], landmarks[27]['y'], landmarks[27]['z']])
    right_ankle = np.array([landmarks[28]['x'], landmarks[28]['y'], landmarks[28]['z']])

    features = {}

    # 1. Hip-Knee angles (flexion)
    features['left_hip_knee_angle'] = calculate_angle(left_hip, left_knee, left_ankle)
    features['right_hip_knee_angle'] = calculate_angle(right_hip, right_knee, right_ankle)

    # 2. Knee-Ankle angles (extension)
    # Use hip as reference for vertical
    features['left_knee_ankle_angle'] = calculate_angle(left_knee, left_ankle, left_hip)
    features['right_knee_ankle_angle'] = calculate_angle(right_knee, right_ankle, right_hip)

    # 3. Leg-to-vertical angles (elevation)
    features['left_leg_vertical_angle'] = calculate_angle_to_vertical(left_hip, left_ankle)
    features['right_leg_vertical_angle'] = calculate_angle_to_vertical(right_hip, right_ankle)

    # 4. Vertical displacement (y-axis movement)
    features['left_ankle_y'] = left_ankle[1]
    features['right_ankle_y'] = right_ankle[1]

    return features


def extract_temporal_features(angle_sequence):
    """
    Extract temporal features from angle sequence

    Returns: dict of aggregated features
    """
    if len(angle_sequence) == 0:
        return None

    # Convert list of dicts to dict of arrays
    keys = angle_sequence[0].keys()
    temporal_data = {k: np.array([frame[k] for frame in angle_sequence]) for k in keys}

    features = {}

    # For each angle feature, calculate temporal statistics
    for key in ['left_hip_knee_angle', 'right_hip_knee_angle',
                'left_knee_ankle_angle', 'right_knee_ankle_angle',
                'left_leg_vertical_angle', 'right_leg_vertical_angle']:

        data = temporal_data[key]

        # Basic statistics
        features[f'{key}_mean'] = np.mean(data)
        features[f'{key}_std'] = np.std(data)
        features[f'{key}_min'] = np.min(data)
        features[f'{key}_max'] = np.max(data)
        features[f'{key}_range'] = np.ptp(data)  # peak-to-peak

        # Velocity (angular velocity)
        velocity = np.abs(np.diff(data))
        features[f'{key}_velocity_mean'] = np.mean(velocity) if len(velocity) > 0 else 0
        features[f'{key}_velocity_max'] = np.max(velocity) if len(velocity) > 0 else 0

    # 5. Decrementing amplitude (divide into 5 segments)
    for key in ['left_ankle_y', 'right_ankle_y']:
        data = temporal_data[key]
        T = len(data)
        segment_size = T // 5

        if segment_size > 0:
            amplitudes = []
            for i in range(5):
                start = i * segment_size
                end = (i + 1) * segment_size if i < 4 else T
                segment = data[start:end]
                amp = np.ptp(segment)  # amplitude in this segment
                amplitudes.append(amp)

            # Calculate amplitude decay
            amplitudes = np.array(amplitudes)
            features[f'{key}_decrement'] = (amplitudes[0] - amplitudes[-1]) / (amplitudes[0] + 1e-8)
            features[f'{key}_decrement_linear_slope'] = np.polyfit(range(5), amplitudes, 1)[0]

    # 6. Left/Right symmetry
    for base_key in ['hip_knee_angle', 'knee_ankle_angle', 'leg_vertical_angle']:
        left_mean = features[f'left_{base_key}_mean']
        right_mean = features[f'right_{base_key}_mean']
        features[f'{base_key}_symmetry'] = abs(left_mean - right_mean)
        features[f'{base_key}_asymmetry_ratio'] = abs(left_mean - right_mean) / (left_mean + right_mean + 1e-8)

    return features


def process_video(video_path, processor):
    """Process single video and extract angle features"""
    if not os.path.exists(video_path):
        return None

    try:
        # Extract skeleton using MediaPipe
        landmark_frames = processor.process_video(
            str(video_path),
            skip_video_generation=True,
            resize_width=256,
            use_mediapipe_optimal=True
        )

        if not landmark_frames or len(landmark_frames) == 0:
            return None

        # Extract angle features frame by frame
        angle_sequence = []
        for lf in landmark_frames:
            if not lf.landmarks:
                continue

            # Convert landmark list to dict format for extract_angle_features
            lm_dict = {lm['id']: lm for lm in lf.landmarks}

            angles = extract_angle_features(lm_dict)
            if angles is not None:
                angle_sequence.append(angles)

        if len(angle_sequence) < 10:  # Minimum 10 frames
            return None

        # Extract temporal features
        features = extract_temporal_features(angle_sequence)

        return features

    except Exception as e:
        return None


def extract_features(split='train'):
    """Extract angle-based features for train or test set"""
    print(f"\n{'='*60}")
    print(f"Extracting Angle Features - {split.upper()} SET")
    print(f"{'='*60}\n")

    # Parse annotations
    csv_path = f"{ANNOTATION_ROOT}/{split}.csv"
    annotations = parse_annotation(csv_path)
    print(f"Total annotations: {len(annotations)}")

    # Process videos
    processor = MediaPipeProcessor(mode='pose')  # Specify pose mode for leg tracking
    features_list = []
    scores = []
    video_ids = []
    sides = []
    failed = []

    for i, annotation in enumerate(tqdm(annotations, desc=f"{split} extraction")):
        video_id = annotation['video_id']
        subject = annotation['subject']
        video_filename = f"{video_id}.mp4"
        video_path = f"{VIDEO_ROOT}/{subject}/{video_filename}"

        try:
            features = process_video(video_path, processor)

            if features is not None:
                features_list.append(list(features.values()))
                scores.append(annotation['score'])
                video_ids.append(video_id)
                sides.append(annotation['side'])
            else:
                failed.append(video_id)
        except Exception as e:
            print(f"\n  ERROR processing {video_id}: {e}")
            failed.append(video_id)

        # Periodic garbage collection every 50 videos
        if (i + 1) % 50 == 0:
            gc.collect()

    # Convert to numpy arrays with error handling
    try:
        X = np.array(features_list, dtype=np.float32)  # Use float32 to save memory
        y = np.array(scores, dtype=np.int32)
    except Exception as e:
        print(f"\n  ERROR converting to numpy arrays: {e}")
        print(f"  Features list length: {len(features_list)}")
        if len(features_list) > 0:
            print(f"  First feature length: {len(features_list[0])}")
        raise

    print(f"\n{split.upper()} SET:")
    print(f"  Successfully extracted: {len(X)}/{len(annotations)} videos")
    print(f"  Failed: {len(failed)} videos")
    if len(X) > 0:
        print(f"  Feature shape: {X.shape}")
        print(f"  Scores: {np.bincount(y.astype(int))}")
        print(f"  Side distribution: Left={sides.count('left')}, Right={sides.count('right')}")
    else:
        print("  WARNING: No features extracted!")

    # Save features with error handling
    output_path = f"{OUTPUT_DIR}/leg_agility_{split}_angles_v3.pkl"
    try:
        print(f"\n  Saving to: {output_path}")
        data_to_save = {
            'X': X,
            'y': y,
            'video_id': video_ids,
            'side': sides,
            'feature_names': list(features.keys()) if features else []
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  [OK] Successfully saved to: {output_path}")

        # Verify saved file
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"\n  ERROR saving pickle file: {e}")
        print(f"  Attempting to save to backup location...")

        # Try backup location
        backup_path = f"{OUTPUT_DIR}/leg_agility_{split}_angles_v3_backup.pkl"
        try:
            with open(backup_path, 'wb') as f:
                pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [OK] Saved to backup: {backup_path}")
        except Exception as e2:
            print(f"  ERROR saving backup: {e2}")
            raise

    # Save failed list
    if failed:
        failed_path = f"{OUTPUT_DIR}/leg_agility_{split}_failed_v3.txt"
        with open(failed_path, 'w') as f:
            for vid in failed:
                f.write(f"{vid}\n")
        print(f"  Failed videos: {failed_path}")

    return X, y, video_ids, sides


def main():
    print("="*60)
    print("Leg Agility Angle-Based Feature Extraction (v3)")
    print("="*60)
    print("\nFeatures:")
    print("  1. Hip-Knee angles (flexion)")
    print("  2. Knee-Ankle angles (extension)")
    print("  3. Leg-to-vertical angles (elevation)")
    print("  4. Left/Right symmetry")
    print("  5. Decrementing amplitude (fatigue)")

    # Extract train
    extract_features('train')

    # Extract test
    extract_features('test')

    print("\n" + "="*60)
    print("Extraction Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
