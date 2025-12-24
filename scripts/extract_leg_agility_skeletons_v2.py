"""
Extract Leg Agility 3D Skeletons from PD4T Videos (IMPROVED VERSION)

Improvements:
1. Save video_id and side (left/right) information
2. Apply Savitzky-Golay smoothing for noise reduction
3. Use cubic spline interpolation for better resampling
4. Optional Kalman filter for trajectory smoothing
5. Better error handling and logging

Leg Landmarks (6 keypoints):
- 23: Left Hip
- 24: Right Hip
- 25: Left Knee
- 26: Right Knee
- 27: Left Ankle
- 28: Right Ankle

Usage:
    python scripts/extract_leg_agility_skeletons_v2.py
"""
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter

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
        # Prediction
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate


def apply_kalman_filter(signal):
    """Apply Kalman filter to 1D signal"""
    kf = KalmanFilter1D()
    filtered = []
    for val in signal:
        filtered.append(kf.update(val))
    return np.array(filtered)


def smooth_trajectory(trajectory, window_length=5, polyorder=2, use_kalman=False):
    """
    Smooth trajectory using Savitzky-Golay filter and optionally Kalman filter

    Args:
        trajectory: (T, F) array - T frames, F features
        window_length: Savitzky-Golay window length (must be odd)
        polyorder: Polynomial order for Savitzky-Golay
        use_kalman: Whether to apply Kalman filter after Savitzky-Golay

    Returns:
        Smoothed trajectory (T, F)
    """
    T, F = trajectory.shape

    # Ensure window_length is odd and <= T
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, T)
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length > T:
            # Can't apply Savitzky-Golay, return original
            return trajectory

    smoothed = np.zeros_like(trajectory)

    # Apply Savitzky-Golay filter to each feature
    for f in range(F):
        try:
            smoothed[:, f] = savgol_filter(trajectory[:, f], window_length, polyorder)
        except:
            # If fails, use original
            smoothed[:, f] = trajectory[:, f]

    # Optional Kalman filter
    if use_kalman:
        kalman_smoothed = np.zeros_like(smoothed)
        for f in range(F):
            kalman_smoothed[:, f] = apply_kalman_filter(smoothed[:, f])
        smoothed = kalman_smoothed

    return smoothed


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

            # Detect side
            side = 'left' if video_id.endswith('_l') else 'right'

            annotations.append({
                'video_id': video_id,
                'subject': subject,
                'score': score,
                'side': side
            })
    return annotations


def extract_skeleton_sequence(video_path, mode="pose", target_frames=150,
                              use_smoothing=True, use_kalman=False,
                              interpolation='cubic'):
    """
    Extract skeleton sequence from video (LEG LANDMARKS ONLY)

    Args:
        video_path: Path to video
        mode: "pose" (uses MediaPipe Pose)
        target_frames: Target number of frames (resample to this)
        use_smoothing: Apply Savitzky-Golay smoothing
        use_kalman: Apply Kalman filter
        interpolation: 'linear' or 'cubic' for resampling

    Returns:
        numpy array of shape (target_frames, 18) or None
        18 features = 6 leg landmarks × 3 coordinates (x, y, z)
    """
    processor = MediaPipeProcessor(mode=mode)

    # Leg landmarks only (Hip, Knee, Ankle)
    LEG_LANDMARKS = [23, 24, 25, 26, 27, 28]

    try:
        landmark_frames = processor.process_video(
            video_path,
            skip_video_generation=True,
            resize_width=256,
            use_mediapipe_optimal=True
        )

        if not landmark_frames or len(landmark_frames) == 0:
            return None

        # Extract coordinates (LEG ONLY)
        sequences = []
        for lf in landmark_frames:
            if not lf.landmarks:
                continue

            coords = []
            # Filter only leg landmarks (23-28)
            for lm in lf.landmarks:
                if lm['id'] in LEG_LANDMARKS:
                    coords.extend([lm['x'], lm['y'], lm['z']])

            # Only add if we have all 6 leg landmarks (18 coordinates)
            if len(coords) == 18:
                sequences.append(coords)

        if len(sequences) == 0:
            return None

        sequences = np.array(sequences)  # (T, 18)

        # Apply smoothing BEFORE resampling
        if use_smoothing and len(sequences) >= 5:
            sequences = smooth_trajectory(sequences,
                                         window_length=5,
                                         polyorder=2,
                                         use_kalman=use_kalman)

        # Resample to target_frames
        if len(sequences) != target_frames:
            old_indices = np.linspace(0, len(sequences)-1, len(sequences))
            new_indices = np.linspace(0, len(sequences)-1, target_frames)

            resampled = []
            for feature_idx in range(sequences.shape[1]):
                if interpolation == 'cubic' and len(sequences) >= 4:
                    # Cubic spline interpolation
                    try:
                        cs = CubicSpline(old_indices, sequences[:, feature_idx])
                        resampled.append(cs(new_indices))
                    except:
                        # Fallback to linear
                        interp_func = interp1d(old_indices, sequences[:, feature_idx], kind='linear')
                        resampled.append(interp_func(new_indices))
                else:
                    # Linear interpolation
                    interp_func = interp1d(old_indices, sequences[:, feature_idx], kind='linear')
                    resampled.append(interp_func(new_indices))

            sequences = np.array(resampled).T

        return sequences

    except Exception as e:
        print(f"  Error processing {video_path}: {e}")
        return None


def main():
    print("="*60)
    print("LEG AGILITY SKELETON EXTRACTION V2 (IMPROVED)")
    print("Features:")
    print("  - 3D coordinates (x, y, z)")
    print("  - Savitzky-Golay smoothing")
    print("  - Cubic spline interpolation")
    print("  - Saves video_id and side information")
    print("="*60)

    for split in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")

        # Load annotations
        annotation_path = f"{ANNOTATION_ROOT}/{split}.csv"
        annotations = parse_annotation(annotation_path)
        print(f"Loaded {len(annotations)} annotations")

        # Count sides
        left_count = sum(1 for a in annotations if a['side'] == 'left')
        right_count = sum(1 for a in annotations if a['side'] == 'right')
        print(f"  Left: {left_count} ({left_count/len(annotations)*100:.1f}%)")
        print(f"  Right: {right_count} ({right_count/len(annotations)*100:.1f}%)")

        X_list = []
        y_list = []
        video_ids = []
        sides = []
        failed = 0
        failed_videos = []

        for ann in tqdm(annotations, desc=f"{split} extraction"):
            video_id = ann['video_id']
            subject = ann['subject']
            score = ann['score']
            side = ann['side']

            # Find video file
            video_path = f"{VIDEO_ROOT}/{subject}/{video_id}.mp4"
            if not os.path.exists(video_path):
                video_path = f"{VIDEO_ROOT}/{subject}/{video_id}.avi"
            if not os.path.exists(video_path):
                video_path = f"{VIDEO_ROOT}/{subject}/{video_id}.MOV"

            if not os.path.exists(video_path):
                print(f"  Video not found: {video_path}")
                failed += 1
                failed_videos.append(video_id)
                continue

            # Extract skeleton (with smoothing and cubic interpolation)
            skeleton = extract_skeleton_sequence(
                video_path,
                mode="pose",
                target_frames=150,
                use_smoothing=True,
                use_kalman=False,  # Kalman can be too aggressive
                interpolation='cubic'
            )

            if skeleton is None:
                failed += 1
                failed_videos.append(video_id)
                continue

            X_list.append(skeleton)
            y_list.append(score)
            video_ids.append(video_id)
            sides.append(side)

        # Convert to arrays
        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\n{split.upper()} Results:")
        print(f"  Extracted: {len(X)} videos")
        print(f"  Failed: {failed} videos ({failed/(len(annotations))*100:.1f}%)")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Score distribution: {np.bincount(y)}")

        # Side distribution
        left_extracted = sides.count('left')
        right_extracted = sides.count('right')
        print(f"  Extracted sides:")
        print(f"    Left: {left_extracted} ({left_extracted/len(sides)*100:.1f}%)")
        print(f"    Right: {right_extracted} ({right_extracted/len(sides)*100:.1f}%)")

        # Save with video_id and side
        output_path = f"{OUTPUT_DIR}/leg_agility_{split}_v2.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'X': X,
                'y': y,
                'video_id': video_ids,
                'side': sides
            }, f)
        print(f"  Saved: {output_path}")

        # Save failed videos list
        if failed_videos:
            failed_path = f"{OUTPUT_DIR}/leg_agility_{split}_failed.txt"
            with open(failed_path, 'w') as f:
                for vid in failed_videos:
                    f.write(f"{vid}\n")
            print(f"  Failed videos saved: {failed_path}")

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
