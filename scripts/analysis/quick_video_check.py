"""
Quick check: pkl 파일 구조 + skeleton 품질 확인
"""

import pickle
import numpy as np
from pathlib import Path

class QuickVideoChecker:
    def __init__(self):
        self.base_dir = Path("C:/Users/YK/tulip/Hawkeye")
        self.data_dir = self.base_dir / "data"

    def check_pkl_structure(self):
        """Check pkl file structure"""
        train_path = self.data_dir / "leg_agility_train.pkl"

        with open(train_path, 'rb') as f:
            data = pickle.load(f)

        print(f"\n{'='*60}")
        print(f"PKL File Structure")
        print(f"{'='*60}")
        print(f"Keys: {data.keys()}")

        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                print(f"\n{key}:")
                print(f"  Shape: {data[key].shape}")
                print(f"  Dtype: {data[key].dtype}")
                if key == 'y':
                    print(f"  Unique values: {np.unique(data[key])}")
                    print(f"  Distribution: {np.bincount(data[key].astype(int))}")
            elif isinstance(data[key], list):
                print(f"\n{key}:")
                print(f"  Length: {len(data[key])}")
                print(f"  First 5: {data[key][:5]}")

    def check_skeleton_quality(self):
        """Check skeleton quality"""
        train_path = self.data_dir / "leg_agility_train.pkl"

        with open(train_path, 'rb') as f:
            data = pickle.load(f)

        X = data['X']
        y = data['y']

        if len(X.shape) == 3:
            N, T, F = X.shape
            X = X.reshape(N, T, 6, 3)

        N, T, J, C = X.shape

        print(f"\n{'='*60}")
        print(f"Skeleton Quality Check")
        print(f"{'='*60}")
        print(f"Shape: {X.shape}")

        # Left vs Right
        print(f"\nLeft vs Right Landmarks (non-zero %)")
        left_hip = (X[:, :, 0, :] != 0).any(axis=2).sum() / (N*T) * 100
        right_hip = (X[:, :, 1, :] != 0).any(axis=2).sum() / (N*T) * 100
        left_knee = (X[:, :, 2, :] != 0).any(axis=2).sum() / (N*T) * 100
        right_knee = (X[:, :, 3, :] != 0).any(axis=2).sum() / (N*T) * 100
        left_ankle = (X[:, :, 4, :] != 0).any(axis=2).sum() / (N*T) * 100
        right_ankle = (X[:, :, 5, :] != 0).any(axis=2).sum() / (N*T) * 100

        print(f"  Left Hip: {left_hip:.1f}%, Right Hip: {right_hip:.1f}%")
        print(f"  Left Knee: {left_knee:.1f}%, Right Knee: {right_knee:.1f}%")
        print(f"  Left Ankle: {left_ankle:.1f}%, Right Ankle: {right_ankle:.1f}%")

        # Sample analysis
        print(f"\nSample 0 (Score {y[0]}):")
        sample = X[0]
        for j in range(6):
            names = ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
            coords = sample[:, j, :2]
            nonzero = (coords != 0).any(axis=1).sum()
            movement = np.linalg.norm(np.diff(coords, axis=0), axis=1).mean() if nonzero > 1 else 0
            print(f"  {names[j]}: {nonzero}/150 frames, movement={movement:.4f}")

        # Video ID
        if 'video_id' in data:
            print(f"\nVideo IDs (first 10):")
            for i in range(min(10, len(data['video_id']))):
                print(f"  {data['video_id'][i]} (score={y[i]})")

def main():
    checker = QuickVideoChecker()
    checker.check_pkl_structure()
    checker.check_skeleton_quality()

if __name__ == "__main__":
    main()
