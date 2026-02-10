"""
Leg Agility: Knee-Only Approach 테스트
- Hip 제거 (앉아있어서 거의 고정)
- Knee + Ankle만 사용 (4 landmarks)
- Knee trajectory features 추가
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class KneeOnlyAnalyzer:
    def __init__(self):
        self.base_dir = Path("C:/Users/YK/tulip/Hawkeye")
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "scripts/analysis/output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load current 6-landmark data"""
        train_path = self.data_dir / "leg_agility_train.pkl"
        test_path = self.data_dir / "leg_agility_test.pkl"

        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

        return train_data, test_data

    def extract_knee_only_features(self, X):
        """Extract knee + ankle only (4 landmarks)"""
        # X: (N, T, 18) or (N, T, 6, 3)

        if len(X.shape) == 3:
            N, T, F = X.shape
            X = X.reshape(N, T, 6, 3)

        N, T, J, C = X.shape

        # Landmark indices:
        # 0,1: Left/Right Hip (제거)
        # 2,3: Left/Right Knee ← 사용
        # 4,5: Left/Right Ankle ← 사용

        knee_ankle = X[:, :, 2:6, :]  # (N, T, 4, 3) - Knee(2) + Ankle(2)

        return knee_ankle

    def compute_knee_trajectory_features(self, knee_ankle):
        """Compute knee trajectory features"""
        N, T, J, C = knee_ankle.shape

        # Left knee trajectory (Y coordinate - 상하 움직임)
        left_knee_y = knee_ankle[:, :, 0, 1]  # (N, T)

        features = []

        for sample_idx in range(N):
            trajectory = left_knee_y[sample_idx]  # (T,)

            # 1. Range (얼마나 높이 올리는지)
            range_y = trajectory.max() - trajectory.min()

            # 2. Velocity (속도)
            velocity = np.diff(trajectory)
            mean_velocity = np.abs(velocity).mean()
            max_velocity = np.abs(velocity).max()

            # 3. Acceleration (가속도)
            acceleration = np.diff(velocity)
            mean_accel = np.abs(acceleration).mean()

            # 4. Periodicity (주기성)
            # Find peaks (무릎이 올라간 순간)
            peaks, _ = find_peaks(trajectory, distance=10)
            num_peaks = len(peaks)

            # Peak interval (리듬 일정성)
            if num_peaks > 1:
                peak_intervals = np.diff(peaks)
                rhythm_std = peak_intervals.std()
            else:
                rhythm_std = 0

            # 5. Knee-Ankle distance (무릎 굽힘 정도)
            left_knee = knee_ankle[sample_idx, :, 0, :2]  # (T, 2)
            left_ankle = knee_ankle[sample_idx, :, 2, :2]  # (T, 2)
            knee_ankle_dist = np.linalg.norm(left_knee - left_ankle, axis=-1)  # (T,)
            mean_knee_ankle_dist = knee_ankle_dist.mean()

            # 6. 초기/말기 속도 변화 (Fatigue, Decrement) ← 파킨슨 핵심!
            early_frames = min(30, T // 3)
            late_frames = min(30, T // 3)

            early_velocity = np.abs(velocity[:early_frames]).mean()
            late_velocity = np.abs(velocity[-late_frames:]).mean()

            # Velocity decrement ratio (초기 대비 말기 감소율)
            if early_velocity > 0:
                velocity_decrement = (early_velocity - late_velocity) / early_velocity
            else:
                velocity_decrement = 0

            # 7. 초기/말기 진폭 변화 (Amplitude decrement)
            early_range = trajectory[:early_frames].max() - trajectory[:early_frames].min()
            late_range = trajectory[-late_frames:].max() - trajectory[-late_frames:].min()

            if early_range > 0:
                amplitude_decrement = (early_range - late_range) / early_range
            else:
                amplitude_decrement = 0

            features.append([
                range_y,
                mean_velocity,
                max_velocity,
                mean_accel,
                num_peaks,
                rhythm_std,
                mean_knee_ankle_dist,
                early_velocity,
                late_velocity,
                velocity_decrement,
                amplitude_decrement
            ])

        features = np.array(features)  # (N, 11)

        return features, [
            'knee_range_y',
            'mean_velocity',
            'max_velocity',
            'mean_accel',
            'num_peaks',
            'rhythm_std',
            'knee_ankle_dist',
            'early_velocity',
            'late_velocity',
            'velocity_decrement',  # 파킨슨 핵심!
            'amplitude_decrement'  # 파킨슨 핵심!
        ]

    def analyze_knee_features(self, train_data, test_data):
        """Analyze knee-only features"""
        print(f"\n{'='*60}")
        print(f"Knee-Only Feature Analysis")
        print(f"{'='*60}")

        # Extract knee+ankle only
        train_X_knee = self.extract_knee_only_features(train_data['X'])
        test_X_knee = self.extract_knee_only_features(test_data['X'])

        train_y = train_data['y']
        test_y = test_data['y']

        print(f"\nKnee+Ankle shape: {train_X_knee.shape}")  # (N, T, 4, 3)

        # Compute trajectory features
        train_features, feature_names = self.compute_knee_trajectory_features(train_X_knee)
        test_features, _ = self.compute_knee_trajectory_features(test_X_knee)

        print(f"Trajectory features: {train_features.shape}")  # (N, 11)
        print(f"Feature names: {feature_names}")

        # Combine all data
        all_features = np.concatenate([train_features, test_features], axis=0)
        all_y = np.concatenate([train_y, test_y], axis=0)

        # Analyze feature correlation with score
        print(f"\n[Feature-Score Correlation Analysis]")

        for i, fname in enumerate(feature_names):
            feature_values = all_features[:, i]

            # Pearson correlation
            correlation = np.corrcoef(feature_values, all_y)[0, 1]

            print(f"  {fname:20s}: Pearson = {correlation:+.3f}")

        # Visualize
        self._visualize_knee_features(all_features, all_y, feature_names)

        # Compare with original approach
        self._compare_approaches(train_X_knee, train_features, train_y)

    def _visualize_knee_features(self, features, y, feature_names):
        """Visualize knee trajectory features by score"""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()

        for i, fname in enumerate(feature_names):
            ax = axes[i]

            feature_by_score = [features[y == s, i] for s in range(5)]

            ax.boxplot(feature_by_score, labels=['0', '1', '2', '3', '4'])
            ax.set_xlabel('UPDRS Score')
            ax.set_ylabel(fname)
            ax.set_title(f'{fname} by Score')
            ax.grid(True, alpha=0.3)

        # Hide unused subplot
        axes[11].axis('off')

        plt.tight_layout()
        output_path = self.output_dir / "leg_knee_trajectory_features.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] {output_path}")
        plt.close()

    def _compare_approaches(self, knee_ankle, knee_features, y):
        """Compare Hip+Knee+Ankle vs Knee+Ankle only"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Original (6 landmarks movement)
        ax = axes[0]

        # Compute movement from all 6 landmarks (for comparison with original)
        # 하지만 여기서는 knee+ankle(4)만 있으므로 이것만 계산
        movement = np.diff(knee_ankle, axis=1)  # (N, T-1, 4, 3)
        movement_mag = np.linalg.norm(movement, axis=-1)  # (N, T-1, 4)
        per_sample_movement = movement_mag.mean(axis=(1, 2))  # (N,)

        movement_by_score = [per_sample_movement[y == s] for s in range(5)]

        ax.boxplot(movement_by_score, labels=['0', '1', '2', '3', '4'])
        ax.set_xlabel('UPDRS Score')
        ax.set_ylabel('Movement Magnitude')
        ax.set_title('Original: Knee+Ankle Movement\n(문제: Score별 차이 없음)')
        ax.grid(True, alpha=0.3)

        # Right: New (Knee trajectory features)
        ax = axes[1]

        # Use knee_range_y (index 0) as representative
        knee_range_by_score = [knee_features[y == s, 0] for s in range(5)]

        ax.boxplot(knee_range_by_score, labels=['0', '1', '2', '3', '4'])
        ax.set_xlabel('UPDRS Score')
        ax.set_ylabel('Knee Range Y')
        ax.set_title('New: Knee Trajectory Range\n(Score별 패턴 있을까?)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "leg_approach_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {output_path}")
        plt.close()

    def run_analysis(self):
        """Run complete knee-only analysis"""
        train_data, test_data = self.load_data()

        self.analyze_knee_features(train_data, test_data)

        print(f"\n{'='*60}")
        print(f"[CONCLUSION]")
        print(f"{'='*60}")
        print(f"1. Hip 제거 → Hip-Knee overlap 문제 해결")
        print(f"2. Knee trajectory features:")
        print(f"   - Range, velocity, acceleration")
        print(f"   - Periodicity (num_peaks, rhythm)")
        print(f"   - Knee-Ankle distance")
        print(f"3. 이 features가 Score와 상관관계 있는지 확인")
        print(f"4. 상관관계 있으면 → 재학습 진행")
        print(f"   상관관계 없으면 → Leg Agility 포기")


def main():
    analyzer = KneeOnlyAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
