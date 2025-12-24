"""
Extract Leg Agility 3D Skeletons from PD4T Videos (PARALLEL VERSION)

Improvements over v2:
1. Multiprocessing for parallel video processing
2. 4-8x faster on multi-core CPUs
3. Same quality as v2 (Savitzky-Golay + Cubic Spline)

Usage:
    python scripts/extract_leg_agility_skeletons_v3_parallel.py --workers 4
"""
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter
import multiprocessing as mp
from functools import partial
import argparse

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.mediapipe_processor import MediaPipeProcessor

# Paths
PD4T_ROOT = "C:/Users/YK/tulip/Hawkeye/data/raw/PD4T/PD4T/PD4T"
VIDEO_ROOT = f"{PD4T_ROOT}/Videos/Leg agility"
ANNOTATION_ROOT = f"{PD4T_ROOT}/Annotations/Leg agility/stratified"
OUTPUT_DIR = "C:/Users/YK/tulip/Hawkeye/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


class KalmanFilter1D:
    """Simple 1D Kalman filter for smoothing trajectories"""
    def __init__(self, process_variance=1e-5, measurement_variance=1e-3):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate


def smooth_trajectory(trajectory, window_length=5, polyorder=2, use_kalman=False):
    """
    Smooth trajectory using Savitzky-Golay filter

    Args:
        trajectory: (T, F) array of features over time
        window_length: Window size for Savitzky-Golay filter
        polyorder: Polynomial order
        use_kalman: Whether to apply Kalman filter (default: False, too aggressive)
    """
    T, F = trajectory.shape

    # Ensure window_length is odd and <= T
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, T)
    if window_length < polyorder + 2:
        return trajectory  # Can't smooth, return original

    smoothed = np.zeros_like(trajectory)

    # Apply Savitzky-Golay filter to each feature
    for f in range(F):
        try:
            smoothed[:, f] = savgol_filter(trajectory[:, f], window_length, polyorder)
        except:
            smoothed[:, f] = trajectory[:, f]

    # Optional: Apply Kalman filter (disabled by default as it's too aggressive)
    if use_kalman:
        for f in range(F):
            kf = KalmanFilter1D()
            for t in range(T):
                smoothed[t, f] = kf.update(smoothed[t, f])

    return smoothed


def parse_annotation(csv_path):
    """Parse annotation CSV and extract video_id, subject, score, side"""
    annotations = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue

            video_id_subject = parts[0]
            score = int(parts[2])

            # Parse video_id and subject
            # Format: 15-005087_l_042 or 15-001760_009
            parts_split = video_id_subject.split('_')

            if len(parts_split) == 3:
                # Format: 15-005087_l_042
                video_id = f"{parts_split[0]}_{parts_split[1]}"  # 15-005087_l
                subject = parts_split[2]  # 042
            elif len(parts_split) == 2:
                # Format: 15-001760_009
                video_id = parts_split[0]  # 15-001760
                subject = parts_split[1]  # 009
            else:
                print(f"Warning: Unexpected format: {video_id_subject}")
                continue

            # Detect side from filename ending
            side = 'left' if video_id.endswith('_l') else 'right'

            annotations.append({
                'video_id': video_id,
                'subject': subject,
                'score': score,
                'side': side
            })

    return annotations


def resample_sequence(sequences, target_frames=150, interpolation='cubic'):
    """
    Resample sequence to target_frames using interpolation

    Args:
        sequences: (T, F) array
        target_frames: Target number of frames
        interpolation: 'linear' or 'cubic'
    """
    T, F = sequences.shape

    if T == target_frames:
        return sequences

    old_indices = np.linspace(0, 1, T)
    new_indices = np.linspace(0, 1, target_frames)

    resampled = []
    for feature_idx in range(F):
        if interpolation == 'cubic' and len(sequences) >= 4:
            try:
                # Use cubic spline for smooth interpolation
                cs = CubicSpline(old_indices, sequences[:, feature_idx])
                resampled.append(cs(new_indices))
            except:
                # Fallback to linear if cubic fails
                interp_func = interp1d(old_indices, sequences[:, feature_idx], kind='linear')
                resampled.append(interp_func(new_indices))
        else:
            # Linear interpolation
            interp_func = interp1d(old_indices, sequences[:, feature_idx], kind='linear')
            resampled.append(interp_func(new_indices))

    return np.array(resampled).T  # (target_frames, F)


def process_single_video(item):
    """
    Process a single video (for multiprocessing)

    Args:
        item: Tuple of (annotation_dict, video_path)

    Returns:
        dict with 'sequence', 'score', 'video_id', 'side', or None if failed
    """
    # Import at function start for multiprocessing
    import os
    import sys

    annotation, video_path = item

    if not os.path.exists(video_path):
        return None

    try:
        # Add backend to path for MediaPipe
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from services.mediapipe_processor import MediaPipeProcessor

        # Extract skeleton using MediaPipe (create fresh instance per process)
        processor = MediaPipeProcessor()
        landmarks = processor.process_video(
            str(video_path),
            task='pose',
            visualization=False,  # No visualization for speed
            verbose=False
        )

        # Extract leg keypoints (6 landmarks: hip, knee, ankle x2)
        leg_indices = [23, 24, 25, 26, 27, 28]
        sequences = []

        for frame_landmarks in landmarks:
            if frame_landmarks is None:
                continue

            coords = []
            for idx in leg_indices:
                if idx < len(frame_landmarks):
                    lm = frame_landmarks[idx]
                    # Extract 3D coordinates (x, y, z)
                    coords.extend([lm['x'], lm['y'], lm['z']])
                else:
                    coords.extend([0.0, 0.0, 0.0])

            sequences.append(coords)

        if len(sequences) < 10:
            return None

        sequences = np.array(sequences)  # (T, 18) - 6 landmarks x 3 coords

        # Apply smoothing (Savitzky-Golay filter)
        sequences = smooth_trajectory(sequences, window_length=5, polyorder=2, use_kalman=False)

        # Resample to fixed length (150 frames) using cubic spline
        sequences = resample_sequence(sequences, target_frames=150, interpolation='cubic')

        return {
            'sequence': sequences,
            'score': annotation['score'],
            'video_id': annotation['video_id'],
            'side': annotation['side']
        }

    except Exception as e:
        return None


def extract_parallel(split='train', num_workers=4):
    """
    Extract skeletons in parallel

    Args:
        split: 'train' or 'test'
        num_workers: Number of parallel workers (default: 4)
    """
    print(f"\n{'='*60}")
    print(f"Extracting {split} set with {num_workers} parallel workers")
    print(f"{'='*60}\n")

    # Parse annotations
    csv_path = f"{ANNOTATION_ROOT}/{split}.csv"
    annotations = parse_annotation(csv_path)

    # Prepare video paths
    items = []
    for annotation in annotations:
        video_id = annotation['video_id']
        subject = annotation['subject']
        video_filename = f"{video_id}.mp4"
        video_path = f"{VIDEO_ROOT}/{subject}/{video_filename}"
        items.append((annotation, video_path))

    print(f"Total videos: {len(items)}")
    print(f"Processing with {num_workers} workers...\n")

    # Process in parallel with progress bar
    X = []
    y = []
    video_ids = []
    sides = []
    failed_videos = []

    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, items),
            total=len(items),
            desc=f"{split} extraction"
        ))

    # Collect results
    for i, result in enumerate(results):
        if result is not None:
            X.append(result['sequence'])
            y.append(result['score'])
            video_ids.append(result['video_id'])
            sides.append(result['side'])
        else:
            failed_videos.append(items[i][0]['video_id'])

    X = np.array(X)
    y = np.array(y)

    print(f"\n{split} set:")
    print(f"  Successfully extracted: {len(X)}/{len(annotations)} videos")
    print(f"  Failed: {len(failed_videos)} videos")

    if len(X) > 0:
        print(f"  Shape: {X.shape}")
        print(f"  Scores: {np.bincount(y.astype(int))}")
        print(f"  Side distribution: Left={sides.count('left')}, Right={sides.count('right')}")
    else:
        print(f"  ERROR: All videos failed to extract!")
        return None, None, None, None

    # Save to pickle
    output_path = f"{OUTPUT_DIR}/leg_agility_{split}_v3_parallel.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump({
            'X': X,
            'y': y,
            'video_id': video_ids,
            'side': sides
        }, f)

    print(f"  Saved to: {output_path}")

    # Save failed videos list
    if failed_videos:
        failed_path = f"{OUTPUT_DIR}/leg_agility_{split}_failed_v3.txt"
        with open(failed_path, 'w') as f:
            for vid in failed_videos:
                f.write(f"{vid}\n")
        print(f"  Failed videos saved to: {failed_path}")

    return X, y, video_ids, sides


def main():
    parser = argparse.ArgumentParser(description='Extract leg agility skeletons in parallel')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--split', type=str, default='both', choices=['train', 'test', 'both'],
                        help='Which split to process (default: both)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Leg Agility Skeleton Extraction (Parallel Version)")
    print(f"{'='*60}")
    print(f"Workers: {args.workers}")
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Split: {args.split}")

    if args.split in ['train', 'both']:
        extract_parallel('train', num_workers=args.workers)

    if args.split in ['test', 'both']:
        extract_parallel('test', num_workers=args.workers)

    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
