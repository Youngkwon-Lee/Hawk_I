"""
Hand Movement & Leg Agility 데이터 품질 분석 스크립트
- ROI 확인 (비디오 프레임에서 손/다리 위치)
- MediaPipe skeleton 추출 품질
- Feature 분포 및 통계
- Annotation 품질 검증
"""

import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class DataQualityAnalyzer:
    def __init__(self, task="hand_movement"):
        self.task = task
        self.base_dir = Path("C:/Users/YK/tulip/Hawkeye")
        self.data_dir = self.base_dir / "data"
        self.video_dir = self.data_dir / "raw/PD4T/PD4T/PD4T/Videos"
        self.output_dir = self.base_dir / "scripts/analysis/output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Task별 설정
        if task == "hand_movement":
            self.task_name = "Hand Movement"
            self.video_folder = "Hand movements"
            self.train_pkl = "hand_movement_train.pkl"
            self.test_pkl = "hand_movement_test.pkl"
            self.num_joints = 21  # MediaPipe Hand landmarks
        else:  # leg_agility
            self.task_name = "Leg Agility"
            self.video_folder = "Leg agility"
            self.train_pkl = "leg_agility_train.pkl"
            self.test_pkl = "leg_agility_test.pkl"
            self.num_joints = 6   # Leg landmarks (hips, knees, ankles)

    def load_data(self):
        """Load train/test data"""
        print(f"\n{'='*60}")
        print(f"Loading {self.task_name} data...")
        print(f"{'='*60}")

        # Load pickled data
        train_path = self.data_dir / self.train_pkl
        test_path = self.data_dir / self.test_pkl

        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

        print(f"Train samples: {len(train_data['X'])}")
        print(f"Test samples: {len(test_data['X'])}")

        return train_data, test_data

    def analyze_skeleton_quality(self, train_data, test_data):
        """Analyze MediaPipe skeleton extraction quality"""
        print(f"\n{'='*60}")
        print(f"MediaPipe Skeleton Quality Analysis")
        print(f"{'='*60}")

        all_X = np.concatenate([train_data['X'], test_data['X']], axis=0)
        all_y = np.concatenate([train_data['y'], test_data['y']], axis=0)

        print(f"Total samples: {len(all_X)}")
        print(f"Shape: {all_X.shape}")  # (N, T, J, 3) or (N, T, J*3)

        # Reshape if needed
        if len(all_X.shape) == 3:
            # (N, T, Features) → (N, T, J, 3)
            N, T, F = all_X.shape
            J = self.num_joints
            all_X = all_X.reshape(N, T, J, 3)

        N, T, J, C = all_X.shape
        print(f"Reshaped: N={N}, T={T}, J={J}, C={C}")

        # 1. Check for missing/invalid values
        print(f"\n[1] Missing/Invalid Values Check")
        zero_frames = (all_X == 0).all(axis=(2, 3)).sum()
        print(f"  Zero frames: {zero_frames} / {N*T} ({zero_frames/(N*T)*100:.2f}%)")

        nan_count = np.isnan(all_X).sum()
        print(f"  NaN values: {nan_count}")

        # 2. Coordinate range analysis
        print(f"\n[2] Coordinate Range Analysis")
        x_coords = all_X[:, :, :, 0]
        y_coords = all_X[:, :, :, 1]
        z_coords = all_X[:, :, :, 2]

        print(f"  X: [{x_coords.min():.3f}, {x_coords.max():.3f}] (mean: {x_coords.mean():.3f})")
        print(f"  Y: [{y_coords.min():.3f}, {y_coords.max():.3f}] (mean: {y_coords.mean():.3f})")
        print(f"  Z: [{z_coords.min():.3f}, {z_coords.max():.3f}] (mean: {z_coords.mean():.3f})")

        # 3. Movement variance (동작 활발도)
        print(f"\n[3] Movement Variance (동작 활발도)")
        movement = np.diff(all_X, axis=1)  # (N, T-1, J, 3)
        movement_mag = np.linalg.norm(movement, axis=-1)  # (N, T-1, J)

        per_sample_movement = movement_mag.mean(axis=(1, 2))  # (N,)

        print(f"  Movement magnitude: [{per_sample_movement.min():.4f}, {per_sample_movement.max():.4f}]")
        print(f"  Mean: {per_sample_movement.mean():.4f}")
        print(f"  Std: {per_sample_movement.std():.4f}")

        # 4. Score별 움직임 차이
        print(f"\n[4] Movement by UPDRS Score")
        for score in range(5):
            mask = (all_y == score)
            if mask.sum() > 0:
                score_movement = per_sample_movement[mask].mean()
                print(f"  Score {score}: {score_movement:.4f} (n={mask.sum()})")

        # Visualization
        self._plot_skeleton_quality(all_X, all_y, per_sample_movement)

        return {
            'zero_frames': zero_frames,
            'nan_count': nan_count,
            'movement_stats': {
                'mean': per_sample_movement.mean(),
                'std': per_sample_movement.std(),
                'min': per_sample_movement.min(),
                'max': per_sample_movement.max()
            }
        }

    def _plot_skeleton_quality(self, X, y, movement):
        """Plot skeleton quality metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Movement distribution
        ax = axes[0, 0]
        ax.hist(movement, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(movement.mean(), color='red', linestyle='--', label=f'Mean: {movement.mean():.4f}')
        ax.set_xlabel('Movement Magnitude')
        ax.set_ylabel('Count')
        ax.set_title(f'{self.task_name}: Movement Distribution')
        ax.legend()

        # 2. Movement by score
        ax = axes[0, 1]
        movement_by_score = [movement[y == s] for s in range(5)]
        ax.boxplot(movement_by_score, labels=['0', '1', '2', '3', '4'])
        ax.set_xlabel('UPDRS Score')
        ax.set_ylabel('Movement Magnitude')
        ax.set_title(f'{self.task_name}: Movement by Score')

        # 3. X-Y coordinate scatter (첫 프레임, 모든 샘플)
        ax = axes[1, 0]
        x_coords = X[:, 0, :, 0].flatten()  # 첫 프레임의 모든 x 좌표
        y_coords = X[:, 0, :, 1].flatten()  # 첫 프레임의 모든 y 좌표
        ax.scatter(x_coords, y_coords, alpha=0.1, s=1)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'{self.task_name}: Landmark Positions (Frame 0)')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.invert_yaxis()  # MediaPipe Y는 위에서 아래

        # 4. Score distribution
        ax = axes[1, 1]
        score_counts = [(y == s).sum() for s in range(5)]
        ax.bar(range(5), score_counts, alpha=0.7, edgecolor='black')
        ax.set_xlabel('UPDRS Score')
        ax.set_ylabel('Count')
        ax.set_title(f'{self.task_name}: Score Distribution')
        for i, count in enumerate(score_counts):
            ax.text(i, count, str(count), ha='center', va='bottom')

        plt.tight_layout()
        output_path = self.output_dir / f"{self.task}_skeleton_quality.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✅ Saved: {output_path}")
        plt.close()

    def analyze_feature_distribution(self, train_data, test_data):
        """Analyze feature distribution"""
        print(f"\n{'='*60}")
        print(f"Feature Distribution Analysis")
        print(f"{'='*60}")

        all_X = np.concatenate([train_data['X'], test_data['X']], axis=0)
        all_y = np.concatenate([train_data['y'], test_data['y']], axis=0)

        # Flatten to (N, Features)
        if len(all_X.shape) == 4:
            N, T, J, C = all_X.shape
            all_X_flat = all_X.reshape(N, -1)
        else:
            all_X_flat = all_X

        print(f"Feature shape: {all_X_flat.shape}")

        # Feature statistics
        feature_means = all_X_flat.mean(axis=0)
        feature_stds = all_X_flat.std(axis=0)

        print(f"\nFeature Statistics:")
        print(f"  Mean range: [{feature_means.min():.4f}, {feature_means.max():.4f}]")
        print(f"  Std range: [{feature_stds.min():.4f}, {feature_stds.max():.4f}]")

        # Zero features
        zero_features = (feature_stds == 0).sum()
        print(f"  Zero-variance features: {zero_features} / {len(feature_stds)}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Feature mean distribution
        ax = axes[0]
        ax.hist(feature_means, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Feature Mean')
        ax.set_ylabel('Count')
        ax.set_title(f'{self.task_name}: Feature Mean Distribution')

        # Feature std distribution
        ax = axes[1]
        ax.hist(feature_stds, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Feature Std')
        ax.set_ylabel('Count')
        ax.set_title(f'{self.task_name}: Feature Std Distribution')

        plt.tight_layout()
        output_path = self.output_dir / f"{self.task}_feature_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  ✅ Saved: {output_path}")
        plt.close()

    def compare_with_gait(self):
        """Compare with Gait task (high performance baseline)"""
        print(f"\n{'='*60}")
        print(f"Comparison with Gait Task (Pearson 0.807)")
        print(f"{'='*60}")

        # TODO: Load Gait data and compare
        print("  [To be implemented]")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print(f"\n{'='*60}")
        print(f"🔍 {self.task_name} Data Quality Analysis")
        print(f"{'='*60}")

        # 1. Load data
        train_data, test_data = self.load_data()

        # 2. Skeleton quality
        skeleton_stats = self.analyze_skeleton_quality(train_data, test_data)

        # 3. Feature distribution
        self.analyze_feature_distribution(train_data, test_data)

        # 4. Compare with Gait
        self.compare_with_gait()

        print(f"\n{'='*60}")
        print(f"✅ Analysis Complete!")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")


def main():
    # Hand Movement analysis
    print("\n" + "="*80)
    print("HAND MOVEMENT ANALYSIS (Pearson 0.593)")
    print("="*80)
    hand_analyzer = DataQualityAnalyzer(task="hand_movement")
    hand_analyzer.run_full_analysis()

    # Leg Agility analysis
    print("\n" + "="*80)
    print("LEG AGILITY ANALYSIS (Pearson 0.221 - FAILED)")
    print("="*80)
    leg_analyzer = DataQualityAnalyzer(task="leg_agility")
    leg_analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
