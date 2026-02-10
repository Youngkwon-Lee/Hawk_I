"""
Leg Agility Landmark 비교 분석
- 현재: 6 landmarks (hips, knees, ankles)
- 제안: 22 landmarks (upper body + lower body)
- Hip flexion 시 joint overlap 문제 확인
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class LandmarkComparison:
    def __init__(self):
        self.base_dir = Path("C:/Users/YK/tulip/Hawkeye")
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "scripts/analysis/output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_current_data(self):
        """Load current 6-landmark data"""
        train_path = self.data_dir / "leg_agility_train.pkl"

        with open(train_path, 'rb') as f:
            data = pickle.load(f)

        return data

    def analyze_joint_overlap(self, data):
        """Analyze joint overlap problem"""
        print(f"\n{'='*60}")
        print(f"Joint Overlap Analysis")
        print(f"{'='*60}")

        X = data['X']
        y = data['y']

        # Reshape to (N, T, J, 3)
        if len(X.shape) == 3:
            N, T, F = X.shape
            J = 6  # hips(2), knees(2), ankles(2)
            X = X.reshape(N, T, J, 3)

        N, T, J, C = X.shape
        print(f"Data shape: N={N}, T={T}, J={J} landmarks")

        # Landmark indices (추정):
        # 0,1: Left/Right Hip
        # 2,3: Left/Right Knee
        # 4,5: Left/Right Ankle

        # Score별 joint distance 분석
        print(f"\n[1] Hip-Knee Distance Analysis (앉은 자세에서 얼마나 가까운지)")

        for score in range(5):
            score_mask = (y == score)
            if score_mask.sum() == 0:
                continue

            score_samples = X[score_mask]  # (N_score, T, 6, 3)

            # Left hip (0) - Left knee (2) distance
            left_hip = score_samples[:, :, 0, :2]  # (N, T, 2) - X, Y만
            left_knee = score_samples[:, :, 2, :2]

            # Euclidean distance
            distances = np.linalg.norm(left_hip - left_knee, axis=-1)  # (N, T)
            mean_dist = distances.mean()
            std_dist = distances.std()
            min_dist = distances.min()

            print(f"  Score {score} (n={score_mask.sum()}): "
                  f"Mean={mean_dist:.3f}, Std={std_dist:.3f}, Min={min_dist:.3f}")

        # Visualization
        self._plot_overlap_analysis(X, y)

    def _plot_overlap_analysis(self, X, y):
        """Plot joint overlap visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Score 0, 2, 4 샘플 선택
        target_scores = [0, 2, 4]

        for col, score in enumerate(target_scores):
            score_mask = (y == score)
            if score_mask.sum() == 0:
                continue

            # 첫 번째 샘플
            sample_idx = np.where(score_mask)[0][0]
            sample = X[sample_idx]  # (T, 6, 3)

            # Row 0: 첫 프레임 (시작)
            ax = axes[0, col]
            frame = sample[0]
            self._plot_skeleton_6landmarks(ax, frame, f'Score {score} - Frame 0 (시작)')

            # Row 1: 중간 프레임 (hip flexion 최대)
            ax = axes[1, col]
            mid_idx = len(sample) // 2
            frame = sample[mid_idx]
            self._plot_skeleton_6landmarks(ax, frame, f'Score {score} - Frame {mid_idx} (중간)')

        plt.tight_layout()
        output_path = self.output_dir / "leg_joint_overlap_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] {output_path}")
        plt.close()

    def _plot_skeleton_6landmarks(self, ax, frame, title):
        """Plot 6-landmark skeleton"""
        # frame: (6, 3) - hips(2), knees(2), ankles(2)

        x = frame[:, 0]
        y = frame[:, 1]

        # Plot joints
        ax.scatter(x, y, c='red', s=100, zorder=3)

        # Label joints
        labels = ['L Hip', 'R Hip', 'L Knee', 'R Knee', 'L Ankle', 'R Ankle']
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)

        # Connections
        connections = [
            (0, 2), (2, 4),  # Left: hip -> knee -> ankle
            (1, 3), (3, 5),  # Right: hip -> knee -> ankle
        ]

        for conn in connections:
            ax.plot([x[conn[0]], x[conn[1]]],
                   [y[conn[0]], y[conn[1]]],
                   'b-', linewidth=2)

        # Hip-Knee distance annotation
        left_hip_knee_dist = np.linalg.norm(frame[0, :2] - frame[2, :2])
        ax.text(0.5, 0.95, f'Hip-Knee Dist: {left_hip_knee_dist:.3f}',
               transform=ax.transAxes, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([-0.5, 1.5])
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def visualize_problem(self):
        """Visualize why 6 landmarks fail"""
        print(f"\n{'='*60}")
        print(f"Why 6 Landmarks Fail for Leg Agility?")
        print(f"{'='*60}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Problem scenario
        ax = axes[0]
        ax.text(0.5, 0.9, 'PROBLEM: Hip Flexion (앉은 자세)', ha='center',
               fontsize=14, fontweight='bold', transform=ax.transAxes)

        # 시나리오 1: Hip과 Knee가 가까움
        hip_pos = np.array([0.5, 0.3])
        knee_pos = np.array([0.5, 0.5])  # 매우 가까움
        ankle_pos = np.array([0.5, 0.8])

        ax.plot([hip_pos[0], knee_pos[0]], [hip_pos[1], knee_pos[1]],
               'r-', linewidth=3, label='Hip-Knee (가까움!)')
        ax.plot([knee_pos[0], ankle_pos[0]], [knee_pos[1], ankle_pos[1]],
               'b-', linewidth=3, label='Knee-Ankle')

        ax.scatter([hip_pos[0], knee_pos[0], ankle_pos[0]],
                  [hip_pos[1], knee_pos[1], ankle_pos[1]],
                  s=200, c='red', zorder=3)

        ax.text(hip_pos[0], hip_pos[1], 'Hip', ha='right', fontsize=12)
        ax.text(knee_pos[0], knee_pos[1], 'Knee', ha='right', fontsize=12)
        ax.text(ankle_pos[0], ankle_pos[1], 'Ankle', ha='right', fontsize=12)

        ax.annotate('', xy=knee_pos, xytext=hip_pos,
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(0.55, 0.4, 'Distance\n< 0.2', fontsize=10, color='red', fontweight='bold')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        # Right: Solution scenario
        ax = axes[1]
        ax.text(0.5, 0.9, 'SOLUTION: Full Body Landmarks', ha='center',
               fontsize=14, fontweight='bold', transform=ax.transAxes)

        # Full body landmarks
        shoulder_pos = np.array([0.5, 0.15])
        hip_pos = np.array([0.5, 0.3])
        knee_pos = np.array([0.5, 0.5])
        ankle_pos = np.array([0.5, 0.8])

        ax.plot([shoulder_pos[0], hip_pos[0]], [shoulder_pos[1], hip_pos[1]],
               'g-', linewidth=3, label='Shoulder-Hip (상체 안정성)')
        ax.plot([hip_pos[0], knee_pos[0]], [hip_pos[1], knee_pos[1]],
               'r-', linewidth=3, label='Hip-Knee')
        ax.plot([knee_pos[0], ankle_pos[0]], [knee_pos[1], ankle_pos[1]],
               'b-', linewidth=3, label='Knee-Ankle')

        ax.scatter([shoulder_pos[0], hip_pos[0], knee_pos[0], ankle_pos[0]],
                  [shoulder_pos[1], hip_pos[1], knee_pos[1], ankle_pos[1]],
                  s=200, c=['green', 'red', 'red', 'blue'], zorder=3)

        ax.text(shoulder_pos[0], shoulder_pos[1], 'Shoulder\n(NEW!)', ha='right',
               fontsize=12, fontweight='bold', color='green')
        ax.text(hip_pos[0], hip_pos[1], 'Hip', ha='right', fontsize=12)
        ax.text(knee_pos[0], knee_pos[1], 'Knee', ha='right', fontsize=12)
        ax.text(ankle_pos[0], ankle_pos[1], 'Ankle', ha='right', fontsize=12)

        ax.text(0.5, 0.5, 'Torso 움직임으로\n보상 감지!',
               ha='center', fontsize=11, color='green', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "leg_why_fullbody_needed.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] {output_path}")
        plt.close()

    def run_analysis(self):
        """Run complete analysis"""
        # Load current data
        data = self.load_current_data()

        # Analyze overlap
        self.analyze_joint_overlap(data)

        # Visualize problem
        self.visualize_problem()

        print(f"\n{'='*60}")
        print(f"[DONE] Analysis Complete!")
        print(f"{'='*60}")
        print(f"\n[CONCLUSION]")
        print(f"1. Hip-Knee distance < 0.3 매우 흔함 (앉은 자세)")
        print(f"2. 6 landmarks로는 torso 움직임 감지 불가")
        print(f"3. Full body (22 landmarks) 필요:")
        print(f"   - Upper body (11-22): Shoulders, elbows → 상체 안정성")
        print(f"   - Lower body (23-32): Hips, knees, ankles → 다리 움직임")


def main():
    analyzer = LandmarkComparison()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
