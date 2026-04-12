"""
Extract Hand Movement Skeletons V2 - Industry Standard Preprocessing

Improvements over v1:
1. Savitzky-Golay smoothing (noise reduction)
2. Cubic spline interpolation (smoother resampling)
3. Wrist-based normalization (translation invariance)
4. Scale normalization (size invariance)

Based on:
- ParkTest (2024): Hand movement analysis for PD
- Leg Agility v2 success: Pearson 0.221 -> 0.307 (+38.9%)

Usage:
    python scripts/extract_hand_movement_skeletons_v2.py
"""
import os
import sys
import gc
import pickle
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.mediapipe_processor import MediaPipeProcessor

# Paths
PD4T_ROOT = "C:/Users/YK/tulip/Hawkeye/data/raw/PD4T/PD4T/PD4T"
VIDEO_ROOT = f"{PD4T_ROOT}/Videos/Hand movement"
ANNOTATION_ROOT = f"{PD4T_ROOT}/Annotations/Hand movement/stratified"
OUTPUT_DIR = "C:/Users/YK/tulip/Hawkeye/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe Hand Landmarks (21 points)
# 0: WRIST
# 1-4: THUMB (CMC, MCP, IP, TIP)
# 5-8: INDEX (MCP, PIP, DIP, TIP)
# 9-12: MIDDLE (MCP, PIP, DIP, TIP)
# 13-16: RING (MCP, PIP, DIP, TIP)
# 17-20: PINKY (MCP, PIP, DIP, TIP)

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9  # For scale reference


def parse_annotation(csv_path):
    """Parse annotation CSV: video_id_subject,frame_count,score"""
    annotations = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 3:
                continue
            video_id_subject = parts[0]
            score = int(parts[2])

            # Parse: 13-005933_l_003 -> video_id=13-005933_l, subject=003
            parts_split = video_id_subject.rsplit('_', 1)
            video_id = parts_split[0]
            subject = parts_split[1]

            annotations.append({
                'video_id': video_id,
                'subject': subject,
                'score': score
            })
    return annotations


def apply_savgol_smoothing(sequence, window_length=5, polyorder=2):
    """
    Apply Savitzky-Golay filter for noise reduction

    Args:
        sequence: (T, F) array
        window_length: Must be odd, typically 5-11
        polyorder: Polynomial order, typically 2-3

    Returns:
        Smoothed sequence
    """
    if len(sequence) < window_length:
        return sequence

    smoothed = np.zeros_like(sequence)
    for i in range(sequence.shape[1]):
        smoothed[:, i] = savgol_filter(sequence[:, i], window_length, polyorder)

    return smoothed


def cubic_spline_resample(sequence, target_frames):
    """
    Resample using cubic spline interpolation (smoother than linear)

    Args:
        sequence: (T, F) array
        target_frames: Target number of frames

    Returns:
        Resampled sequence of shape (target_frames, F)
    """
    if len(sequence) == target_frames:
        return sequence

    old_indices = np.linspace(0, 1, len(sequence))
    new_indices = np.linspace(0, 1, target_frames)

    resampled = np.zeros((target_frames, sequence.shape[1]))
    for i in range(sequence.shape[1]):
        cs = CubicSpline(old_indices, sequence[:, i])
        resampled[:, i] = cs(new_indices)

    return resampled


def normalize_hand_skeleton(sequence, n_landmarks=21):
    """
    Normalize hand skeleton:
    1. Translation: Center at wrist (landmark 0)
    2. Scale: Normalize by wrist-to-middle-MCP distance

    Args:
        sequence: (T, F) where F = n_landmarks * 3
        n_landmarks: Number of landmarks (21 for MediaPipe Hand)

    Returns:
        Normalized sequence
    """
    T, F = sequence.shape
    coords_per_landmark = 3  # x, y, z

    if F != n_landmarks * coords_per_landmark:
        print(f"  Warning: Expected {n_landmarks * coords_per_landmark} features, got {F}")
        return sequence

    # Reshape to (T, n_landmarks, 3)
    reshaped = sequence.reshape(T, n_landmarks, coords_per_landmark)

    normalized = np.zeros_like(reshaped)

    for t in range(T):
        frame = reshaped[t]

        # Get wrist position (reference point)
        wrist = frame[WRIST_IDX]

        # Translation: Center at wrist
        centered = frame - wrist

        # Scale: Use wrist-to-middle-MCP distance
        middle_mcp = centered[MIDDLE_MCP_IDX]
        scale = np.linalg.norm(middle_mcp)

        if scale > 1e-6:
            normalized[t] = centered / scale
        else:
            normalized[t] = centered

    # Reshape back to (T, F)
    return normalized.reshape(T, F)


def extract_skeleton_sequence_v2(video_path, mode="hand", target_frames=150):
    """
    Extract skeleton sequence with industry-standard preprocessing

    Pipeline:
    1. MediaPipe extraction
    2. Savitzky-Golay smoothing
    3. Wrist-based normalization
    4. Cubic spline resampling

    Args:
        video_path: Path to video
        mode: "hand" for MediaPipe Hands
        target_frames: Target number of frames

    Returns:
        numpy array of shape (target_frames, n_features) or None
    """
    processor = MediaPipeProcessor(mode=mode)

    try:
        landmark_frames = processor.process_video(
            video_path,
            skip_video_generation=True,
            resize_width=256,
            use_mediapipe_optimal=True
        )

        if not landmark_frames or len(landmark_frames) == 0:
            return None

        # Extract raw coordinates
        sequences = []
        for lf in landmark_frames:
            if not lf.landmarks:
                continue

            coords = []
            for lm in lf.landmarks:
                coords.extend([lm['x'], lm['y'], lm['z']])
            sequences.append(coords)

        if len(sequences) < 10:  # Too few frames
            return None

        sequences = np.array(sequences, dtype=np.float32)

        # Step 1: Savitzky-Golay smoothing
        window = min(5, len(sequences) if len(sequences) % 2 == 1 else len(sequences) - 1)
        if window >= 3:
            sequences = apply_savgol_smoothing(sequences, window_length=window, polyorder=2)

        # Step 2: Wrist-based normalization
        sequences = normalize_hand_skeleton(sequences, n_landmarks=21)

        # Step 3: Cubic spline resampling
        sequences = cubic_spline_resample(sequences, target_frames)

        return sequences.astype(np.float32)

    except Exception as e:
        print(f"  Error processing {video_path}: {e}")
        return None


def main():
    print("="*60)
    print("HAND MOVEMENT SKELETON EXTRACTION V2")
    print("="*60)
    print("Industry Standard Preprocessing:")
    print("  1. Savitzky-Golay smoothing (noise reduction)")
    print("  2. Wrist-based normalization (translation invariance)")
    print("  3. Scale normalization (size invariance)")
    print("  4. Cubic spline interpolation (smooth resampling)")
    print("="*60)

    for split in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")

        # Load annotations
        annotation_path = f"{ANNOTATION_ROOT}/{split}.csv"
        annotations = parse_annotation(annotation_path)
        print(f"Loaded {len(annotations)} annotations")

        X_list = []
        y_list = []
        video_ids = []
        failed = 0

        for i, ann in enumerate(tqdm(annotations, desc=f"{split} extraction")):
            video_id = ann['video_id']
            subject = ann['subject']
            score = ann['score']

            # Find video file
            video_path = None
            for ext in ['.mp4', '.avi', '.MOV']:
                test_path = f"{VIDEO_ROOT}/{subject}/{video_id}{ext}"
                if os.path.exists(test_path):
                    video_path = test_path
                    break

            if video_path is None:
                failed += 1
                continue

            # Extract skeleton with v2 preprocessing
            skeleton = extract_skeleton_sequence_v2(
                video_path,
                mode="hand",
                target_frames=150
            )

            if skeleton is None:
                failed += 1
                continue

            X_list.append(skeleton)
            y_list.append(score)
            video_ids.append(f"{video_id}_{subject}")

            # Memory management
            if (i + 1) % 50 == 0:
                gc.collect()

        # Convert to arrays
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        print(f"\n{split.upper()} Results:")
        print(f"  Extracted: {len(X)} videos")
        print(f"  Failed: {failed} videos")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Score distribution: {np.bincount(y)}")

        # Save with metadata
        output_path = f"{OUTPUT_DIR}/hand_movement_{split}_v2.pkl"
        save_data = {
            'X': X,
            'y': y,
            'video_ids': video_ids,
            'preprocessing': {
                'smoothing': 'savgol_filter(window=5, polyorder=2)',
                'normalization': 'wrist_centered + scale_normalized',
                'resampling': 'cubic_spline(target=150)',
                'features': '21_landmarks * 3_coords = 63'
            },
            'version': 'v2'
        }

        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"  Saved: {output_path}")

        gc.collect()

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE (V2)")
    print(f"{'='*60}")
    print("Files created:")
    print(f"  - {OUTPUT_DIR}/hand_movement_train_v2.pkl")
    print(f"  - {OUTPUT_DIR}/hand_movement_test_v2.pkl")


if __name__ == "__main__":
    main()
