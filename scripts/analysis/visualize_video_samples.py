"""
Hand Movement & Leg Agility 비디오 샘플 시각화
- 실제 비디오에서 MediaPipe skeleton 오버레이
- ROI 확인 (손/다리가 프레임 안에 제대로 들어오는지)
- 카메라 각도, 거리 문제 파악
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import pandas as pd

class VideoSampleVisualizer:
    def __init__(self, task="hand_movement"):
        self.task = task
        self.base_dir = Path("C:/Users/YK/tulip/Hawkeye")
        self.data_dir = self.base_dir / "data"
        self.video_dir = self.data_dir / "raw/PD4T/PD4T/PD4T"
        self.annotation_dir = self.video_dir / "Annotations"
        self.output_dir = self.base_dir / "scripts/analysis/output/video_samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Task별 설정
        if task == "hand_movement":
            self.task_name = "Hand Movement"
            self.video_folder = "Hand movements"
            self.annotation_file = "train.csv"
        else:  # leg_agility
            self.task_name = "Leg Agility"
            self.video_folder = "Leg agility"
            self.annotation_file = "train.csv"

        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def load_annotations(self):
        """Load annotation CSV"""
        annotation_path = self.annotation_dir / self.task_name / self.annotation_file
        df = pd.read_csv(annotation_path, header=None, names=['video_subject', 'frame_count', 'score'])

        # Parse video_id and subject
        df['video_id'] = df['video_subject'].str.rsplit('_', n=1).str[0]
        df['subject'] = df['video_subject'].str.rsplit('_', n=1).str[1]

        print(f"\nLoaded {len(df)} annotations from {annotation_path}")
        print(f"Score distribution:")
        print(df['score'].value_counts().sort_index())

        return df

    def find_video_path(self, video_id, subject):
        """Find video file path"""
        video_folder = self.video_dir / "Videos" / self.task_name / subject

        # Try different extensions
        for ext in ['.mp4', '.avi', '.mov']:
            video_path = video_folder / f"{video_id}{ext}"
            if video_path.exists():
                return video_path

        print(f"  ⚠️ Video not found: {video_id} (subject {subject})")
        return None

    def visualize_sample(self, video_path, score, max_frames=30):
        """Visualize video sample with MediaPipe skeleton overlay"""
        print(f"\n  Processing: {video_path.name} (Score: {score})")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ❌ Failed to open video")
            return

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"    Size: {width}x{height}, FPS: {fps:.1f}, Frames: {frame_count}")

        # Initialize MediaPipe
        if self.task == "hand_movement":
            detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
        else:  # leg_agility
            detector = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5
            )

        # Sample frames evenly
        frame_indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)

        output_frames = []
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect landmarks
            if self.task == "hand_movement":
                results = detector.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
            else:  # leg_agility
                results = detector.process(frame_rgb)
                if results.pose_landmarks:
                    # Draw only leg landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )

            # Add frame info
            cv2.putText(frame, f"Frame {frame_idx}/{frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Score: {score}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            output_frames.append(frame)

        cap.release()
        detector.close()

        # Save montage
        if output_frames:
            montage = self._create_montage(output_frames, cols=6)
            output_path = self.output_dir / f"{self.task}_score{score}_{video_path.stem}.png"
            cv2.imwrite(str(output_path), montage)
            print(f"    ✅ Saved: {output_path}")
        else:
            print(f"    ⚠️ No frames extracted")

    def _create_montage(self, frames, cols=6):
        """Create image montage from frames"""
        n_frames = len(frames)
        rows = (n_frames + cols - 1) // cols

        # Resize frames
        target_height = 200
        resized_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            target_width = int(w * target_height / h)
            resized = cv2.resize(frame, (target_width, target_height))
            resized_frames.append(resized)

        # Create montage
        max_width = max(f.shape[1] for f in resized_frames)
        montage_height = target_height * rows
        montage_width = max_width * cols
        montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

        for i, frame in enumerate(resized_frames):
            row = i // cols
            col = i % cols
            h, w = frame.shape[:2]
            y = row * target_height
            x = col * max_width
            montage[y:y+h, x:x+w] = frame

        return montage

    def run_visualization(self, samples_per_score=2):
        """Run visualization for representative samples"""
        print(f"\n{'='*60}")
        print(f"🎬 {self.task_name} Video Sample Visualization")
        print(f"{'='*60}")

        df = self.load_annotations()

        # Select samples for each score
        for score in range(5):
            score_samples = df[df['score'] == score]
            if len(score_samples) == 0:
                print(f"\nScore {score}: No samples")
                continue

            print(f"\nScore {score}: {len(score_samples)} samples")

            # Randomly select samples
            selected = score_samples.sample(n=min(samples_per_score, len(score_samples)))

            for _, row in selected.iterrows():
                video_path = self.find_video_path(row['video_id'], row['subject'])
                if video_path:
                    self.visualize_sample(video_path, score)

        print(f"\n{'='*60}")
        print(f"✅ Visualization Complete!")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")


def main():
    # Hand Movement visualization
    print("\n" + "="*80)
    print("HAND MOVEMENT VIDEO SAMPLES (Pearson 0.593)")
    print("="*80)
    hand_viz = VideoSampleVisualizer(task="hand_movement")
    hand_viz.run_visualization(samples_per_score=2)

    # Leg Agility visualization
    print("\n" + "="*80)
    print("LEG AGILITY VIDEO SAMPLES (Pearson 0.221 - FAILED)")
    print("="*80)
    leg_viz = VideoSampleVisualizer(task="leg_agility")
    leg_viz.run_visualization(samples_per_score=2)


if __name__ == "__main__":
    main()
