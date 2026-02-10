"""
Extract Leg Agility Features with Tapping Count & Height Analysis (v4)
Frontal plane: Only use knee, ankle Y coordinates
No angle analysis (meaningless in frontal view)

Features:
1. Tapping count (ankle y peak detection)
2. Maximum height (amplitude)
3. Average height
4. Decrementing amplitude (first 5 vs last 5 taps)
5. Tapping frequency
6. Rhythm regularity
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from scipy.signal import find_peaks

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.mediapipe_processor import MediaPipeProcessor

# Paths
ANNOTATION_ROOT = "data/raw/PD4T/PD4T/PD4T/Annotations/Leg agility"
VIDEO_ROOT = "data/raw/PD4T/PD4T/PD4T/Videos/Leg agility"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Leg landmarks (only knee and ankle)
LEG_LANDMARKS = {
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle'
}

def parse_annotation(csv_path):
    """Parse annotation CSV"""
    annotations = []
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                video_id_subject = parts[0]
                score = int(parts[2])

                # Extract subject and video_id
                parts_split = video_id_subject.split('_')
                subject = parts_split[-1]
                video_id = '_'.join(parts_split[:-1])

                # Determine side
                side = 'left' if video_id.endswith('_l') else 'right'

                annotations.append({
                    'video_id': video_id,
                    'subject': subject,
                    'score': score,
                    'side': side
                })

    return annotations

def extract_tapping_features(ankle_y_trajectory, fps=30):
    """
    Extract tapping count and height features from ankle y trajectory

    Args:
        ankle_y_trajectory: Array of ankle y coordinates (normalized 0-1)
        fps: Frames per second (for frequency calculation)

    Returns:
        Dictionary of features
    """
    if len(ankle_y_trajectory) < 10:
        return None

    # Invert y (MediaPipe y is top=0, bottom=1, we want bottom=0, top=1)
    ankle_height = 1.0 - ankle_y_trajectory

    # 1. Detect taps (peaks in height)
    # Use prominence to filter small movements
    peaks, properties = find_peaks(
        ankle_height,
        prominence=0.05,  # Minimum height change
        distance=5  # Minimum 5 frames between taps
    )

    n_taps = len(peaks)

    if n_taps == 0:
        return None

    # Get peak heights
    peak_heights = ankle_height[peaks]

    # 2. Basic statistics
    max_height = np.max(ankle_height)
    min_height = np.min(ankle_height)
    amplitude = max_height - min_height
    mean_height = np.mean(peak_heights)
    std_height = np.std(peak_heights)

    # 3. Tapping frequency
    duration_sec = len(ankle_height) / fps
    frequency = n_taps / duration_sec if duration_sec > 0 else 0

    # 4. Rhythm regularity (inter-tap interval variability)
    if n_taps > 1:
        inter_tap_intervals = np.diff(peaks)
        rhythm_std = np.std(inter_tap_intervals)
        rhythm_cv = rhythm_std / np.mean(inter_tap_intervals) if np.mean(inter_tap_intervals) > 0 else 0
    else:
        rhythm_std = 0
        rhythm_cv = 0

    # 5. Decrementing amplitude (fatigue)
    if n_taps >= 10:
        first_5_heights = peak_heights[:5]
        last_5_heights = peak_heights[-5:]

        first_mean = np.mean(first_5_heights)
        last_mean = np.mean(last_5_heights)

        decrement = first_mean - last_mean
        decrement_ratio = decrement / first_mean if first_mean > 0 else 0
    else:
        # Use first half vs second half
        n_half = n_taps // 2
        if n_half > 0:
            first_mean = np.mean(peak_heights[:n_half])
            last_mean = np.mean(peak_heights[n_half:])
            decrement = first_mean - last_mean
            decrement_ratio = decrement / first_mean if first_mean > 0 else 0
        else:
            decrement = 0
            decrement_ratio = 0

    # 6. Speed features (ankle velocity)
    ankle_velocity = np.abs(np.diff(ankle_height))
    max_velocity = np.max(ankle_velocity)
    mean_velocity = np.mean(ankle_velocity)

    features = {
        # Count features
        'n_taps': n_taps,
        'frequency_hz': frequency,

        # Height features
        'max_height': max_height,
        'amplitude': amplitude,
        'mean_peak_height': mean_height,
        'std_peak_height': std_height,

        # Rhythm features
        'rhythm_std': rhythm_std,
        'rhythm_cv': rhythm_cv,

        # Fatigue features
        'decrement_absolute': decrement,
        'decrement_ratio': decrement_ratio,

        # Velocity features
        'max_velocity': max_velocity,
        'mean_velocity': mean_velocity
    }

    return features

def process_video(video_path, processor):
    """Process video and extract tapping features"""
    try:
        # Extract landmarks
        landmark_frames = processor.process_video(
            str(video_path),
            skip_video_generation=True,
            resize_width=256,
            use_mediapipe_optimal=True
        )

        if not landmark_frames:
            return None

        # Extract ankle y trajectory (use the side being tested)
        ankle_y_left = []
        ankle_y_right = []

        for lf in landmark_frames:
            if not lf.landmarks:
                continue

            # Convert to dict
            lm_dict = {lm['id']: lm for lm in lf.landmarks}

            # Check if ankle landmarks exist
            if 27 in lm_dict:  # left ankle
                ankle_y_left.append(lm_dict[27]['y'])

            if 28 in lm_dict:  # right ankle
                ankle_y_right.append(lm_dict[28]['y'])

        # Use the side with more detections
        if len(ankle_y_left) > len(ankle_y_right):
            ankle_y = np.array(ankle_y_left)
        else:
            ankle_y = np.array(ankle_y_right)

        if len(ankle_y) < 10:
            return None

        # Extract features
        features = extract_tapping_features(ankle_y)

        return features

    except Exception as e:
        return None

def extract_features(split='train'):
    """Extract tapping features for train or test set"""
    print(f"\n{'='*60}")
    print(f"Extracting Tapping Features - {split.upper()} SET")
    print(f"{'='*60}\n")

    # Parse annotations
    csv_path = f"{ANNOTATION_ROOT}/{split}.csv"
    annotations = parse_annotation(csv_path)
    print(f"Total annotations: {len(annotations)}")

    # Process videos
    processor = MediaPipeProcessor(mode='pose')
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
            failed.append(video_id)

        # Periodic garbage collection every 50 videos
        if (i + 1) % 50 == 0:
            gc.collect()

    # Convert to numpy arrays with error handling
    try:
        X = np.array(features_list, dtype=np.float32)
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
    output_path = f"{OUTPUT_DIR}/leg_agility_{split}_tapping_v4.pkl"
    try:
        print(f"\n  Saving to: {output_path}")

        # Get feature names
        if features is not None:
            feature_names = list(features.keys())
        else:
            feature_names = ['n_taps', 'frequency_hz', 'max_height', 'amplitude',
                           'mean_peak_height', 'std_peak_height', 'rhythm_std',
                           'rhythm_cv', 'decrement_absolute', 'decrement_ratio',
                           'max_velocity', 'mean_velocity']

        data_to_save = {
            'X': X,
            'y': y,
            'video_id': video_ids,
            'side': sides,
            'feature_names': feature_names
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  [OK] Successfully saved to: {output_path}")

        # Verify saved file
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"\n  ERROR saving pickle file: {e}")
        raise

    # Save failed list
    if failed:
        failed_path = f"{OUTPUT_DIR}/leg_agility_{split}_failed_v4.txt"
        with open(failed_path, 'w') as f:
            for vid in failed:
                f.write(f"{vid}\n")
        print(f"  Failed videos: {failed_path}")

    return X, y, video_ids, sides

def main():
    print("="*60)
    print("Leg Agility Tapping Count & Height Analysis (v4)")
    print("="*60)
    print("\nFeatures:")
    print("  1. Tapping count (peak detection)")
    print("  2. Maximum height & amplitude")
    print("  3. Rhythm regularity")
    print("  4. Decrementing amplitude (fatigue)")
    print("  5. Velocity features")
    print("\nOnly using: Knee, Ankle Y coordinates (no angles)")

    # Check if train already exists
    train_path = f"{OUTPUT_DIR}/leg_agility_train_tapping_v4.pkl"
    if os.path.exists(train_path):
        print(f"\n[SKIP] Train set already exists: {train_path}")
    else:
        extract_features('train')

    # Extract test set
    extract_features('test')

    print("\n" + "="*60)
    print("Extraction Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
