"""
Leg Agility: Full Analysis with ALL joints (Hip, Knee, Ankle, Both legs)
- Score 0,1,2 3-class classification
- Hip + Knee + Ankle (6 landmarks)
- Both legs (left + right)
- Proper tapping count analysis
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

class FullLegAnalyzer:
    def __init__(self):
        self.base_dir = Path("C:/Users/YK/tulip/Hawkeye")
        self.data_dir = self.base_dir / "data"

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

    def extract_full_features(self, X):
        """Extract features from ALL joints (hip, knee, ankle, both sides)"""
        if len(X.shape) == 3:
            N, T, F = X.shape
            X = X.reshape(N, T, 6, 3)

        N, T, J, C = X.shape

        # Landmark indices:
        # 0: Left Hip, 1: Right Hip
        # 2: Left Knee, 3: Right Knee
        # 4: Left Ankle, 5: Right Ankle

        features = []

        for sample_idx in range(N):
            sample = X[sample_idx]  # (T, 6, 3)

            # Extract trajectories (Y coordinate)
            left_hip_y = sample[:, 0, 1]
            right_hip_y = sample[:, 1, 1]
            left_knee_y = sample[:, 2, 1]
            right_knee_y = sample[:, 3, 1]
            left_ankle_y = sample[:, 4, 1]
            right_ankle_y = sample[:, 5, 1]

            # === Feature Group 1: Tapping Count (CRITICAL!) ===
            # Left leg
            left_knee_peaks, _ = find_peaks(left_knee_y, distance=10, height=left_knee_y.mean())
            left_tapping_count = len(left_knee_peaks)

            # Right leg
            right_knee_peaks, _ = find_peaks(right_knee_y, distance=10, height=right_knee_y.mean())
            right_tapping_count = len(right_knee_peaks)

            # Total
            total_tapping_count = left_tapping_count + right_tapping_count
            avg_tapping_count = (left_tapping_count + right_tapping_count) / 2

            # === Feature Group 2: Range (무릎 높이) ===
            left_knee_range = left_knee_y.max() - left_knee_y.min()
            right_knee_range = right_knee_y.max() - right_knee_y.min()
            avg_knee_range = (left_knee_range + right_knee_range) / 2

            # Ankle range
            left_ankle_range = left_ankle_y.max() - left_ankle_y.min()
            right_ankle_range = right_ankle_y.max() - right_ankle_y.min()
            avg_ankle_range = (left_ankle_range + right_ankle_range) / 2

            # === Feature Group 3: Velocity (속도) ===
            left_knee_vel = np.abs(np.diff(left_knee_y)).mean()
            right_knee_vel = np.abs(np.diff(right_knee_y)).mean()
            avg_knee_vel = (left_knee_vel + right_knee_vel) / 2

            left_ankle_vel = np.abs(np.diff(left_ankle_y)).mean()
            right_ankle_vel = np.abs(np.diff(right_ankle_y)).mean()
            avg_ankle_vel = (left_ankle_vel + right_ankle_vel) / 2

            # === Feature Group 4: Rhythm (리듬 일관성) ===
            # Left leg rhythm
            if left_tapping_count > 1:
                left_intervals = np.diff(left_knee_peaks)
                left_rhythm_std = left_intervals.std()
                left_rhythm_cv = left_rhythm_std / left_intervals.mean() if left_intervals.mean() > 0 else 0
            else:
                left_rhythm_std = 0
                left_rhythm_cv = 0

            # Right leg rhythm
            if right_tapping_count > 1:
                right_intervals = np.diff(right_knee_peaks)
                right_rhythm_std = right_intervals.std()
                right_rhythm_cv = right_rhythm_std / right_intervals.mean() if right_intervals.mean() > 0 else 0
            else:
                right_rhythm_std = 0
                right_rhythm_cv = 0

            avg_rhythm_std = (left_rhythm_std + right_rhythm_std) / 2
            avg_rhythm_cv = (left_rhythm_cv + right_rhythm_cv) / 2

            # === Feature Group 5: Tapping Rate (초당 횟수) ===
            # Assuming 30 FPS, T frames = T/30 seconds
            duration_sec = T / 30.0
            left_tapping_rate = left_tapping_count / duration_sec if duration_sec > 0 else 0
            right_tapping_rate = right_tapping_count / duration_sec if duration_sec > 0 else 0
            avg_tapping_rate = (left_tapping_rate + right_tapping_rate) / 2

            # === Feature Group 6: Hip-Knee-Ankle Coordination ===
            # Hip-Knee distance
            left_hip_knee_dist = np.abs(left_hip_y - left_knee_y).mean()
            right_hip_knee_dist = np.abs(right_hip_y - right_knee_y).mean()
            avg_hip_knee_dist = (left_hip_knee_dist + right_hip_knee_dist) / 2

            # Knee-Ankle distance
            left_knee_ankle_dist = np.abs(left_knee_y - left_ankle_y).mean()
            right_knee_ankle_dist = np.abs(right_knee_y - right_ankle_y).mean()
            avg_knee_ankle_dist = (left_knee_ankle_dist + right_knee_ankle_dist) / 2

            # Full leg extension (Hip-Ankle distance)
            left_hip_ankle_dist = np.abs(left_hip_y - left_ankle_y).mean()
            right_hip_ankle_dist = np.abs(right_hip_y - right_ankle_y).mean()
            avg_hip_ankle_dist = (left_hip_ankle_dist + right_hip_ankle_dist) / 2

            # === Feature Group 7: Decrement (초기→말기 변화) ===
            early_frames = min(30, T // 3)
            late_frames = min(30, T // 3)

            # Velocity decrement
            early_vel = np.abs(np.diff(left_knee_y[:early_frames])).mean()
            late_vel = np.abs(np.diff(left_knee_y[-late_frames:])).mean()
            velocity_decrement = (early_vel - late_vel) / early_vel if early_vel > 0 else 0

            # Amplitude decrement
            early_range = left_knee_y[:early_frames].max() - left_knee_y[:early_frames].min()
            late_range = left_knee_y[-late_frames:].max() - left_knee_y[-late_frames:].min()
            amplitude_decrement = (early_range - late_range) / early_range if early_range > 0 else 0

            # === Feature Group 8: Symmetry (Left-Right 대칭성) ===
            knee_range_asymmetry = np.abs(left_knee_range - right_knee_range)
            knee_vel_asymmetry = np.abs(left_knee_vel - right_knee_vel)
            tapping_count_asymmetry = np.abs(left_tapping_count - right_tapping_count)

            features.append([
                # Tapping count (가장 중요!)
                left_tapping_count,
                right_tapping_count,
                total_tapping_count,
                avg_tapping_count,
                avg_tapping_rate,

                # Range
                avg_knee_range,
                avg_ankle_range,

                # Velocity
                avg_knee_vel,
                avg_ankle_vel,

                # Rhythm
                avg_rhythm_std,
                avg_rhythm_cv,

                # Coordination
                avg_hip_knee_dist,
                avg_knee_ankle_dist,
                avg_hip_ankle_dist,

                # Decrement
                velocity_decrement,
                amplitude_decrement,

                # Symmetry
                knee_range_asymmetry,
                knee_vel_asymmetry,
                tapping_count_asymmetry,
            ])

        return np.array(features)

    def analyze_tapping_by_score(self, X, y):
        """Tapping count 분석"""
        print(f"\n{'='*60}")
        print(f"Tapping Count Analysis by Score")
        print(f"{'='*60}")

        if len(X.shape) == 3:
            N, T, F = X.shape
            X = X.reshape(N, T, 6, 3)

        for score in range(3):
            mask = (y == score)
            if mask.sum() == 0:
                continue

            samples = X[mask]
            tapping_counts = []

            for sample in samples:
                left_knee_y = sample[:, 2, 1]
                peaks, _ = find_peaks(left_knee_y, distance=10, height=left_knee_y.mean())
                tapping_counts.append(len(peaks))

            tapping_counts = np.array(tapping_counts)
            print(f"\nScore {score} (n={mask.sum()}):")
            print(f"  Tapping count: {tapping_counts.mean():.1f} ± {tapping_counts.std():.1f}")
            print(f"  Range: {tapping_counts.min()} - {tapping_counts.max()}")

    def test_full_classification(self):
        """Test with full features"""
        print(f"\n{'='*60}")
        print(f"Full Leg Analysis: Hip + Knee + Ankle, Both sides")
        print(f"{'='*60}")

        # Load data
        X, y = self.load_data()

        # Filter Score 0,1,2
        mask = (y <= 2)
        X_filtered = X[mask]
        y_filtered = y[mask].astype(int)

        print(f"\nFiltered data: {len(X_filtered)} samples")
        print(f"Score distribution:")
        for score in range(3):
            count = (y_filtered == score).sum()
            pct = count / len(y_filtered) * 100
            print(f"  Score {score}: {count} samples ({pct:.1f}%)")

        # Analyze tapping
        self.analyze_tapping_by_score(X_filtered, y_filtered)

        # Extract features
        features = self.extract_full_features(X_filtered)
        print(f"\nFeatures shape: {features.shape}")

        # Split
        np.random.seed(42)
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(indices))

        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        X_train = features[train_idx]
        X_test = features[test_idx]
        y_train = y_filtered[train_idx]
        y_test = y_filtered[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

        # Random Forest
        print(f"\n{'='*60}")
        print(f"Random Forest Classification")
        print(f"{'='*60}")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {acc:.3f}")

        cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Score 0', 'Score 1', 'Score 2']))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        print(f"\nPer-class Accuracy:")
        for i in range(3):
            class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"  Score {i}: {class_acc:.3f}")

        # Feature importance
        print(f"\n{'='*60}")
        print(f"Feature Importance (Top 10)")
        print(f"{'='*60}")
        feature_names = [
            'left_tapping_count',
            'right_tapping_count',
            'total_tapping_count',
            'avg_tapping_count',
            'avg_tapping_rate',
            'avg_knee_range',
            'avg_ankle_range',
            'avg_knee_vel',
            'avg_ankle_vel',
            'avg_rhythm_std',
            'avg_rhythm_cv',
            'avg_hip_knee_dist',
            'avg_knee_ankle_dist',
            'avg_hip_ankle_dist',
            'velocity_decrement',
            'amplitude_decrement',
            'knee_range_asymmetry',
            'knee_vel_asymmetry',
            'tapping_count_asymmetry',
        ]

        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        for idx in sorted_idx[:10]:
            print(f"  {feature_names[idx]:30s}: {importances[idx]:.3f}")

        # Baseline
        baseline_acc = (y_test == np.bincount(y_train).argmax()).mean()
        print(f"\n{'='*60}")
        print(f"Baseline: {baseline_acc:.3f}")
        print(f"Improvement: {(acc - baseline_acc):.3f}")
        print(f"{'='*60}")

        if acc > baseline_acc + 0.15:
            print(f"\n✅ SUCCESS: Full features WORK!")
            print(f"→ Score 0,1,2 구분 가능!")
        elif acc > baseline_acc + 0.05:
            print(f"\n⚠️ MARGINAL: 약한 구분 가능")
        else:
            print(f"\n❌ FAILED: 여전히 구분 불가")


def main():
    analyzer = FullLegAnalyzer()
    analyzer.test_full_classification()


if __name__ == "__main__":
    main()
