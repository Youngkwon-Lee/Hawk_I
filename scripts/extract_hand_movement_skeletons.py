"""
Extract Hand Movement 3D Skeletons from PD4T Videos
Saves as pickle files for CORAL training

Usage:
    python scripts/extract_hand_movement_skeletons.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.mediapipe_processor import MediaPipeProcessor

# Paths
PD4T_ROOT = "C:/Users/YK/tulip/Hawkeye/data/raw/PD4T/PD4T/PD4T"
VIDEO_ROOT = f"{PD4T_ROOT}/Videos/Hand movement"
ANNOTATION_ROOT = f"{PD4T_ROOT}/Annotations/Hand movement/stratified"
OUTPUT_DIR = "C:/Users/YK/tulip/Hawkeye/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def extract_skeleton_sequence(video_path, mode="hand", target_frames=150):
    """
    Extract skeleton sequence from video

    Args:
        video_path: Path to video
        mode: "hand" or "pose"
        target_frames: Target number of frames (resample to this)

    Returns:
        numpy array of shape (target_frames, n_keypoints * 3) or None
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

        # Extract coordinates
        sequences = []
        for lf in landmark_frames:
            if not lf.landmarks:
                continue

            coords = []
            for lm in lf.landmarks:
                coords.extend([lm['x'], lm['y'], lm['z']])
            sequences.append(coords)

        if len(sequences) == 0:
            return None

        sequences = np.array(sequences)

        # Resample to target_frames using linear interpolation
        if len(sequences) != target_frames:
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, len(sequences)-1, len(sequences))
            new_indices = np.linspace(0, len(sequences)-1, target_frames)

            resampled = []
            for feature_idx in range(sequences.shape[1]):
                interp_func = interp1d(old_indices, sequences[:, feature_idx], kind='linear')
                resampled.append(interp_func(new_indices))

            sequences = np.array(resampled).T

        return sequences

    except Exception as e:
        print(f"  Error processing {video_path}: {e}")
        return None


def main():
    print("="*60)
    print("HAND MOVEMENT SKELETON EXTRACTION")
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
        failed = 0

        for ann in tqdm(annotations, desc=f"{split} extraction"):
            video_id = ann['video_id']
            subject = ann['subject']
            score = ann['score']

            # Find video file
            video_path = f"{VIDEO_ROOT}/{subject}/{video_id}.mp4"
            if not os.path.exists(video_path):
                video_path = f"{VIDEO_ROOT}/{subject}/{video_id}.avi"
            if not os.path.exists(video_path):
                video_path = f"{VIDEO_ROOT}/{subject}/{video_id}.MOV"

            if not os.path.exists(video_path):
                print(f"  Video not found: {video_path}")
                failed += 1
                continue

            # Extract skeleton
            skeleton = extract_skeleton_sequence(
                video_path,
                mode="hand",
                target_frames=150
            )

            if skeleton is None:
                failed += 1
                continue

            X_list.append(skeleton)
            y_list.append(score)

        # Convert to arrays
        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\n{split.upper()} Results:")
        print(f"  Extracted: {len(X)} videos")
        print(f"  Failed: {failed} videos")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Score distribution: {np.bincount(y)}")

        # Save
        output_path = f"{OUTPUT_DIR}/hand_movement_{split}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({'X': X, 'y': y}, f)
        print(f"  Saved: {output_path}")

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
