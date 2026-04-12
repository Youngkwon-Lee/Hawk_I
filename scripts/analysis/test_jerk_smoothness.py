"""
Leg Agility: Jerk & Smoothness Features 테스트
- Jerk (3차 미분): 부드러움의 직접 측정
- Movement smoothness: 경직성 측정
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class JerkSmoothnessAnalyzer:
    def __init__(self):
        self.base_dir = Path("C:/Users/YK/tulip/Hawkeye")
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "scripts/analysis/output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load train+test data"""
        train_path = self.data_dir / "leg_agility_train.pkl"
        test_path = self.data_dir / "leg_agility_test.pkl"

        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

        # Combine
        X = np.concatenate([train_data['X'], test_data['X']], axis=0)
        y = np.concatenate([train_data['y'], test_data['y']], axis=0)

        return X, y

    def compute_jerk_features(self, X):
        """Compute jerk (3rd derivative) features"""
        # X: (N, T, 18) or (N, T, 6, 3)

        if len(X.shape) == 3:
            N, T, F = X.shape
            X = X.reshape(N, T, 6, 3)

        N, T, J, C = X.shape

        # Left knee trajectory (Y coordinate)
        left_knee_y = X[:, :, 2, 1]  # (N, T)

        features = []

        for sample_idx in range(N):
            trajectory = left_knee_y[sample_idx]  # (T,)

            # 1st derivative: Velocity
            velocity = np.diff(trajectory)  # (T-1,)

            # 2nd derivative: Acceleration
            acceleration = np.diff(velocity)  # (T-2,)

            # 3rd derivative: Jerk (부드러움의 직접 측정!)
            jerk = np.diff(acceleration)  # (T-3,)

            # Jerk features
            mean_jerk = np.abs(jerk).mean()
            max_jerk = np.abs(jerk).max()
            jerk_std = np.abs(jerk).std()

            # Smoothness: Normalized jerk
            # SPARC (Spectral Arc Length) 대신 간단한 버전
            if len(jerk) > 0:
                smoothness = -np.log(mean_jerk + 1e-8)  # 낮을수록 부드러움
            else:
                smoothness = 0

            # Acceleration variance (경직성)
            accel_variance = acceleration.var()

            # Movement efficiency (경로 길이 / 직선 거리)
            path_length = np.sum(np.abs(velocity))
            start_end_dist = np.abs(trajectory[-1] - trajectory[0])
            if start_end_dist > 0:
                efficiency = path_length / (start_end_dist + 1e-8)
            else:
                efficiency = 0

            features.append([
                mean_jerk,
                max_jerk,
                jerk_std,
                smoothness,
                accel_variance,
                efficiency
            ])

        features = np.array(features)  # (N, 6)

        return features, [
            'mean_jerk',
            'max_jerk',
            'jerk_std',
            'smoothness',
            'accel_variance',
            'movement_efficiency'
        ]

    def analyze_correlation(self, features, y, feature_names):
        """Analyze feature-score correlation"""
        print(f"\n{'='*60}")
        print(f"Jerk & Smoothness Feature Correlation")
        print(f"{'='*60}")

        print(f"\n[Feature-Score Correlation]")

        for i, fname in enumerate(feature_names):
            feature_values = features[:, i]

            # Pearson correlation
            correlation = np.corrcoef(feature_values, y)[0, 1]

            print(f"  {fname:25s}: Pearson = {correlation:+.3f}")

        # Visualize
        self._visualize_features(features, y, feature_names)

    def _visualize_features(self, features, y, feature_names):
        """Visualize features by score"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, fname in enumerate(feature_names):
            ax = axes[i]

            feature_by_score = [features[y == s, i] for s in range(5)]

            bp = ax.boxplot(feature_by_score, tick_labels=['0', '1', '2', '3', '4'])
            ax.set_xlabel('UPDRS Score', fontsize=12)
            ax.set_ylabel(fname, fontsize=12)
            ax.set_title(f'{fname} by Score', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Highlight if correlation is strong
            corr = np.corrcoef(features[:, i], y)[0, 1]
            if abs(corr) > 0.2:
                ax.text(0.5, 0.95, f'Pearson: {corr:+.3f}',
                       transform=ax.transAxes, ha='center',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                       fontsize=10, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / "leg_jerk_smoothness_features.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] {output_path}")
        plt.close()

    def run_analysis(self):
        """Run complete jerk/smoothness analysis"""
        X, y = self.load_data()

        print(f"\nData shape: {X.shape}")
        print(f"Score distribution: {np.bincount(y.astype(int))}")

        # Compute jerk features
        features, feature_names = self.compute_jerk_features(X)

        print(f"\nJerk features shape: {features.shape}")

        # Analyze correlation
        self.analyze_correlation(features, y, feature_names)

        print(f"\n{'='*60}")
        print(f"[CONCLUSION]")
        print(f"{'='*60}")
        print(f"1. Jerk (3차 미분) = 부드러움의 직접 측정")
        print(f"2. Smoothness = 경직성 측정")
        print(f"3. 만약 이것도 상관관계 낮으면:")
        print(f"   - 라벨링 품질 문제 확정")
        print(f"   - 또는 skeleton 추출 자체가 부정확")
        print(f"   - Leg Agility Task 포기 권장")


def main():
    analyzer = JerkSmoothnessAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
