"""
Leg Agility Full Body Feature Extraction
- MediaPipe Pose 33 landmarks (손 제외, 얼굴 제외)
- Upper body (torso, shoulders) - 보상 움직임 감지
- Lower body (hips, knees, ankles) - hip flexion 주 움직임
- ROI 정규화 없음 (raw skeleton)
"""

import os
import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

class LegFullBodyExtractor:
    def __init__(self):
        self.base_dir = Path("C:/Users/YK/tulip/Hawkeye")
        self.video_dir = self.base_dir / "data/raw/PD4T/PD4T/PD4T/Videos/Leg agility"
        self.annotation_dir = self.base_dir / "data/raw/PD4T/PD4T/PD4T/Annotations/Leg Agility"
        self.output_dir = self.base_dir / "data"

        # MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose

        # Landmark 선택: Face(0-10) 제외, Upper(11-22) + Lower(23-32) 사용
        # Upper body: 11-shoulders, 12-elbows, 13-wrists (torso stability)
        # Lower body: 23-hips, 25-knees, 27-ankles, 29-heels, 31-toes
        self.selected_landmarks = list(range(11, 33))  # 11-32: 22개 landmarks
        print(f"Using {len(self.selected_landmarks)} landmarks (upper + lower body, excluding face)")

    def load_annotations(self, split='train'):
        """Load train/test annotations"""
        annotation_file = self.annotation_dir / f"{split}.csv"
        df = pd.read_csv(annotation_file, header=None,
                        names=['video_subject', 'frame_count', 'score'])

        # Parse video_id and subject
        df['video_id'] = df['video_subject'].str.rsplit('_', n=1).str[0]
        df['subject'] = df['video_subject'].str.rsplit('_', n=1).str[1]

        print(f"\n{split.upper()}: {len(df)} samples")
        print(f"Score distribution:\n{df['score'].value_counts().sort_index()}")

        return df

    def find_video_path(self, video_id, subject):
        """Find video file"""
        video_folder = self.video_dir / subject

        for ext in ['.mp4', '.avi', '.mov']:
            video_path = video_folder / f"{video_id}{ext}"
            if video_path.exists():
                return video_path

        return None

    def extract_pose_sequence(self, video_path, max_frames=150):
        """Extract full body pose sequence (excluding face)"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 더 정확한 모델
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            # Sample frames evenly
            frame_indices = np.linspace(0, frame_count - 1,
                                       min(max_frames, frame_count), dtype=int)

            sequences = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    # Extract selected landmarks (11-32)
                    landmarks = []
                    for idx in self.selected_landmarks:
                        lm = results.pose_landmarks.landmark[idx]
                        landmarks.extend([lm.x, lm.y, lm.z])
                    sequences.append(landmarks)
                else:
                    # No detection: use zeros
                    sequences.append([0.0] * (len(self.selected_landmarks) * 3))

            cap.release()

            # Pad or truncate to max_frames
            while len(sequences) < max_frames:
                sequences.append([0.0] * (len(self.selected_landmarks) * 3))
            sequences = sequences[:max_frames]

            return np.array(sequences)  # (T, J*3)

    def extract_dataset(self, split='train'):
        """Extract full dataset"""
        print(f"\n{'='*60}")
        print(f"Extracting {split.upper()} dataset")
        print(f"{'='*60}")

        df = self.load_annotations(split)

        X_list = []
        y_list = []
        video_ids = []
        failed = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}"):
            video_path = self.find_video_path(row['video_id'], row['subject'])

            if video_path is None:
                failed.append(row['video_id'])
                continue

            sequence = self.extract_pose_sequence(video_path)

            if sequence is None:
                failed.append(row['video_id'])
                continue

            X_list.append(sequence)
            y_list.append(row['score'])
            video_ids.append(row['video_id'])

        if failed:
            print(f"\n[WARNING] Failed to process {len(failed)} videos")

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\nExtracted: X shape={X.shape}, y shape={y.shape}")

        return {
            'X': X,
            'y': y,
            'video_ids': video_ids
        }

    def save_dataset(self, data, split='train'):
        """Save to pickle"""
        output_path = self.output_dir / f"leg_agility_fullbody_{split}.pkl"

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\n[SAVED] {output_path}")
        print(f"  Shape: {data['X'].shape}")
        print(f"  Samples: {len(data['y'])}")

    def run_extraction(self):
        """Extract train and test datasets"""
        print(f"\n{'='*60}")
        print(f"Leg Agility Full Body Feature Extraction")
        print(f"{'='*60}")
        print(f"Landmarks: 22 (upper body + lower body, no face)")

        # Train
        train_data = self.extract_dataset(split='train')
        self.save_dataset(train_data, split='train')

        # Test
        test_data = self.extract_dataset(split='test')
        self.save_dataset(test_data, split='test')

        print(f"\n{'='*60}")
        print(f"[DONE] Extraction Complete!")
        print(f"{'='*60}")


def main():
    extractor = LegFullBodyExtractor()
    extractor.run_extraction()


if __name__ == "__main__":
    main()
