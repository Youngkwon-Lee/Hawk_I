"""
Visualize Leg Agility skeleton overlay from v3 extraction
"""
import os
import sys
import cv2
import pickle
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from services.mediapipe_processor import MediaPipeProcessor

# Paths
ANNOTATION_ROOT = "data/raw/PD4T/PD4T/PD4T/Annotations/Leg agility"
VIDEO_ROOT = "data/raw/PD4T/PD4T/PD4T/Videos/Leg agility"
OUTPUT_DIR = "scripts/analysis/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Leg landmarks (MediaPipe Pose)
LEG_LANDMARKS = {
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle'
}

# Leg connections
LEG_CONNECTIONS = [
    (23, 25),  # left hip -> left knee
    (25, 27),  # left knee -> left ankle
    (24, 26),  # right hip -> right knee
    (26, 28),  # right knee -> right ankle
]

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

def draw_skeleton_overlay(frame, landmarks, color=(0, 255, 0), thickness=2):
    """Draw skeleton overlay on frame"""
    h, w = frame.shape[:2]

    # Draw connections
    for connection in LEG_CONNECTIONS:
        start_idx, end_idx = connection

        if start_idx not in landmarks or end_idx not in landmarks:
            continue

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        # Convert normalized coords to pixel coords
        start_point = (int(start['x'] * w), int(start['y'] * h))
        end_point = (int(end['x'] * w), int(end['y'] * h))

        cv2.line(frame, start_point, end_point, color, thickness)

    # Draw joints
    for idx, landmark in landmarks.items():
        if idx in LEG_LANDMARKS:
            x = int(landmark['x'] * w)
            y = int(landmark['y'] * h)
            cv2.circle(frame, (x, y), 5, color, -1)

            # Label
            label = LEG_LANDMARKS[idx].replace('_', ' ').title()
            cv2.putText(frame, label, (x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

def visualize_video(video_path, video_id, score, side, output_prefix):
    """Visualize skeleton overlay for a video"""
    print(f"\nProcessing: {video_id} (Score: {score}, Side: {side})")

    # Process video
    processor = MediaPipeProcessor(mode='pose')
    landmark_frames = processor.process_video(
        str(video_path),
        skip_video_generation=True,
        resize_width=256,
        use_mediapipe_optimal=True
    )

    if not landmark_frames:
        print("  No landmarks detected")
        return False

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    # Select frames to visualize (every 30 frames, max 6 frames)
    total_frames = len(landmark_frames)
    frame_indices = list(range(0, total_frames, max(1, total_frames // 6)))[:6]

    print(f"  Total frames: {total_frames}, Selected: {len(frame_indices)} frames")

    visualizations = []

    for i, frame_idx in enumerate(frame_indices):
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Get landmarks
        lf = landmark_frames[frame_idx]

        if not lf.landmarks:
            continue

        # Convert to dict
        lm_dict = {lm['id']: lm for lm in lf.landmarks}

        # Check if leg landmarks exist
        leg_lm = {k: v for k, v in lm_dict.items() if k in LEG_LANDMARKS}

        if len(leg_lm) < 4:  # Need at least 4 leg landmarks
            continue

        # Draw skeleton overlay
        frame_overlay = draw_skeleton_overlay(frame.copy(), lm_dict)

        # Add info text
        info_text = f"{video_id} | Score: {score} | Side: {side} | Frame: {frame_idx}/{total_frames}"
        cv2.putText(frame_overlay, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        visualizations.append(frame_overlay)

    cap.release()

    if not visualizations:
        print("  No valid frames found")
        return False

    # Create grid layout
    n_frames = len(visualizations)

    if n_frames == 1:
        grid = visualizations[0]
    elif n_frames <= 3:
        # 1 row
        grid = cv2.hconcat(visualizations)
    else:
        # 2 rows
        n_cols = (n_frames + 1) // 2
        row1 = cv2.hconcat(visualizations[:n_cols])
        row2 = cv2.hconcat(visualizations[n_cols:n_cols*2])

        # Pad if needed
        if row1.shape[1] > row2.shape[1]:
            padding = np.zeros((row2.shape[0], row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
            row2 = cv2.hconcat([row2, padding])

        grid = cv2.vconcat([row1, row2])

    # Save
    output_path = f"{OUTPUT_DIR}/{output_prefix}_skeleton_overlay.png"
    cv2.imwrite(output_path, grid)

    print(f"  [OK] Saved: {output_path}")
    print(f"  Image size: {grid.shape[1]}x{grid.shape[0]}")

    return True

def main():
    """Visualize skeleton overlays for sample videos"""
    print("="*60)
    print("Leg Agility Skeleton Overlay Visualization (v3)")
    print("="*60)

    # Parse train annotations
    csv_path = f"{ANNOTATION_ROOT}/train.csv"
    annotations = parse_annotation(csv_path)

    # Select representative samples (different scores)
    samples = []

    # Score 0 (normal)
    score_0 = [a for a in annotations if a['score'] == 0]
    if score_0:
        samples.append(score_0[0])

    # Score 1 (slight)
    score_1 = [a for a in annotations if a['score'] == 1]
    if score_1:
        samples.append(score_1[0])

    # Score 2 (mild)
    score_2 = [a for a in annotations if a['score'] == 2]
    if score_2:
        samples.append(score_2[0])

    # Score 3 (moderate)
    score_3 = [a for a in annotations if a['score'] == 3]
    if score_3:
        samples.append(score_3[0])

    print(f"\nSelected {len(samples)} sample videos (different scores)")

    # Visualize each sample
    success_count = 0

    for annotation in samples:
        video_id = annotation['video_id']
        subject = annotation['subject']
        score = annotation['score']
        side = annotation['side']

        video_filename = f"{video_id}.mp4"
        video_path = f"{VIDEO_ROOT}/{subject}/{video_filename}"

        if not os.path.exists(video_path):
            print(f"  Video not found: {video_path}")
            continue

        output_prefix = f"leg_agility_score{score}_{side}"

        if visualize_video(video_path, video_id, score, side, output_prefix):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Visualization Complete: {success_count}/{len(samples)} videos")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
