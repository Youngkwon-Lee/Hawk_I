"""
PKL 파일에서 skeleton 직접 시각화 (비디오 불필요)
- Score별 skeleton 패턴 비교
- ROI 범위 확인
- Movement trajectory 시각화
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class SkeletonVisualizer:
    def __init__(self, task="hand_movement"):
        self.task = task
        self.base_dir = Path("C:/Users/YK/tulip/Hawkeye")
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "scripts/analysis/output/video_samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if task == "hand_movement":
            self.task_name = "Hand Movement"
            self.train_pkl = "hand_movement_train.pkl"
            self.test_pkl = "hand_movement_test.pkl"
            self.num_joints = 21
        else:
            self.task_name = "Leg Agility"
            self.train_pkl = "leg_agility_train.pkl"
            self.test_pkl = "leg_agility_test.pkl"
            self.num_joints = 6

    def load_data(self):
        """Load train/test data"""
        print(f"\n{'='*60}")
        print(f"[LOADING] {self.task_name}")
        print(f"{'='*60}")

        train_path = self.data_dir / self.train_pkl
        test_path = self.data_dir / self.test_pkl

        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

        return train_data, test_data

    def visualize_score_samples(self, train_data, test_data):
        """Visualize skeleton samples for each score"""
        all_X = np.concatenate([train_data['X'], test_data['X']], axis=0)
        all_y = np.concatenate([train_data['y'], test_data['y']], axis=0)

        # Reshape if needed
        if len(all_X.shape) == 3:
            N, T, F = all_X.shape
            all_X = all_X.reshape(N, T, self.num_joints, 3)

        N, T, J, C = all_X.shape
        print(f"Data shape: N={N}, T={T}, J={J}, C={C}")

        # Score 0, 2, 4만 시각화
        target_scores = [0, 2, 4]

        for score in target_scores:
            score_mask = (all_y == score)
            score_count = score_mask.sum()

            if score_count == 0:
                print(f"  Score {score}: No samples")
                continue

            print(f"  Score {score}: {score_count} samples")

            # 첫 번째 샘플 선택
            score_idx = np.where(score_mask)[0][0]
            sample = all_X[score_idx]  # (T, J, 3)

            # Visualize
            self._plot_skeleton_sequence(sample, score)

    def _plot_skeleton_sequence(self, sample, score):
        """Plot skeleton sequence (10 frames)"""
        T, J, C = sample.shape

        # Sample 10 frames evenly
        frame_indices = np.linspace(0, T-1, min(10, T), dtype=int)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i, frame_idx in enumerate(frame_indices):
            ax = axes[i]
            frame = sample[frame_idx]  # (J, 3)

            # Plot X-Y coordinates
            x_coords = frame[:, 0]
            y_coords = frame[:, 1]

            # Plot joints
            ax.scatter(x_coords, y_coords, c='red', s=50, alpha=0.7)

            # Connect joints (simple connections)
            if self.task == "hand_movement":
                # Hand connections (simplified)
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                ]
            else:  # leg_agility (6 joints: left hip, right hip, left knee, right knee, left ankle, right ankle)
                # Leg connections
                connections = [
                    (0, 2), (2, 4),  # Left: hip -> knee -> ankle
                    (1, 3), (3, 5),  # Right: hip -> knee -> ankle
                ]

            for conn in connections:
                if conn[0] < J and conn[1] < J:
                    ax.plot([x_coords[conn[0]], x_coords[conn[1]]],
                           [y_coords[conn[0]], y_coords[conn[1]]],
                           'b-', linewidth=1, alpha=0.5)

            # Set limits and labels
            ax.set_xlim([-0.2, 1.2])
            ax.set_ylim([-0.2, 1.2])
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_title(f'Frame {frame_idx}/{T}')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{self.task_name} - Score {score} Sample', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / f"{self.task}_score{score}_skeleton.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    [SAVED] {output_path}")
        plt.close()

    def visualize_roi_comparison(self, train_data, test_data):
        """Compare ROI across scores"""
        all_X = np.concatenate([train_data['X'], test_data['X']], axis=0)
        all_y = np.concatenate([train_data['y'], test_data['y']], axis=0)

        # Reshape if needed
        if len(all_X.shape) == 3:
            N, T, F = all_X.shape
            all_X = all_X.reshape(N, T, self.num_joints, 3)

        N, T, J, C = all_X.shape

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        for score in range(5):
            ax = axes[score]
            score_mask = (all_y == score)

            if score_mask.sum() == 0:
                ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
                ax.set_title(f'Score {score}')
                continue

            score_samples = all_X[score_mask]  # (N_score, T, J, 3)

            # 모든 프레임의 모든 joints X-Y 좌표
            x_coords = score_samples[:, :, :, 0].flatten()
            y_coords = score_samples[:, :, :, 1].flatten()

            # Scatter plot
            ax.scatter(x_coords, y_coords, alpha=0.1, s=1)
            ax.set_xlim([-0.5, 1.5])
            ax.set_ylim([-0.5, 1.5])
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_title(f'Score {score} (n={score_mask.sum()})')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{self.task_name} - ROI Comparison by Score', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / f"{self.task}_roi_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] {output_path}")
        plt.close()

    def run_visualization(self):
        """Run full visualization"""
        print(f"\n{'='*60}")
        print(f"[VISUALIZATION] {self.task_name}")
        print(f"{'='*60}")

        train_data, test_data = self.load_data()

        print(f"\n[1] Visualizing skeleton samples...")
        self.visualize_score_samples(train_data, test_data)

        print(f"\n[2] Visualizing ROI comparison...")
        self.visualize_roi_comparison(train_data, test_data)

        print(f"\n{'='*60}")
        print(f"[DONE] Visualization Complete!")
        print(f"{'='*60}")


def main():
    # Hand Movement
    print("\n" + "="*80)
    print("HAND MOVEMENT SKELETON VISUALIZATION (Pearson 0.593)")
    print("="*80)
    hand_viz = SkeletonVisualizer(task="hand_movement")
    hand_viz.run_visualization()

    # Leg Agility
    print("\n" + "="*80)
    print("LEG AGILITY SKELETON VISUALIZATION (Pearson 0.221 - FAILED)")
    print("="*80)
    leg_viz = SkeletonVisualizer(task="leg_agility")
    leg_viz.run_visualization()


if __name__ == "__main__":
    main()
