"""
Leg Agility: Score 0,1,2 3-Class Classification Test
- Score 3,4 제외 (샘플 부족)
- Score 0,1,2만 사용 (597 samples)
- 구분 가능 여부 테스트
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

class MildClassificationTester:
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

    def extract_all_features(self, X):
        """Extract all features (knee + jerk + smoothness)"""
        if len(X.shape) == 3:
            N, T, F = X.shape
            X = X.reshape(N, T, 6, 3)

        N, T, J, C = X.shape

        # Left knee trajectory (Y coordinate)
        left_knee_y = X[:, :, 2, 1]  # (N, T)

        features = []

        for sample_idx in range(N):
            trajectory = left_knee_y[sample_idx]  # (T,)

            # 1. Range
            range_y = trajectory.max() - trajectory.min()

            # 2. Velocity
            velocity = np.diff(trajectory)
            mean_velocity = np.abs(velocity).mean()
            max_velocity = np.abs(velocity).max()

            # 3. Acceleration
            acceleration = np.diff(velocity)
            mean_accel = np.abs(acceleration).mean()

            # 4. Jerk
            jerk = np.diff(acceleration)
            mean_jerk = np.abs(jerk).mean() if len(jerk) > 0 else 0
            max_jerk = np.abs(jerk).max() if len(jerk) > 0 else 0
            jerk_std = np.abs(jerk).std() if len(jerk) > 0 else 0

            # 5. Smoothness
            smoothness = -np.log(mean_jerk + 1e-8) if len(jerk) > 0 else 0

            # 6. Acceleration variance
            accel_variance = acceleration.var()

            # 7. Efficiency
            path_length = np.sum(np.abs(velocity))
            start_end_dist = np.abs(trajectory[-1] - trajectory[0])
            efficiency = path_length / (start_end_dist + 1e-8) if start_end_dist > 0 else 0

            # 8. Velocity decrement
            early_frames = min(30, T // 3)
            late_frames = min(30, T // 3)
            early_velocity = np.abs(velocity[:early_frames]).mean()
            late_velocity = np.abs(velocity[-late_frames:]).mean()
            velocity_decrement = (early_velocity - late_velocity) / early_velocity if early_velocity > 0 else 0

            # 9. Amplitude decrement
            early_range = trajectory[:early_frames].max() - trajectory[:early_frames].min()
            late_range = trajectory[-late_frames:].max() - trajectory[-late_frames:].min()
            amplitude_decrement = (early_range - late_range) / early_range if early_range > 0 else 0

            # 10. Rhythm (peaks)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(trajectory, distance=10)
            num_peaks = len(peaks)
            if num_peaks > 1:
                peak_intervals = np.diff(peaks)
                rhythm_std = peak_intervals.std()
            else:
                rhythm_std = 0

            features.append([
                range_y,
                mean_velocity,
                max_velocity,
                mean_accel,
                mean_jerk,
                max_jerk,
                jerk_std,
                smoothness,
                accel_variance,
                efficiency,
                velocity_decrement,
                amplitude_decrement,
                num_peaks,
                rhythm_std
            ])

        return np.array(features)

    def test_3class_classification(self):
        """Test 3-class classification (0,1,2)"""
        print(f"\n{'='*60}")
        print(f"3-Class Classification: Score 0, 1, 2 Only")
        print(f"{'='*60}")

        # Load data
        X, y = self.load_data()

        # Filter only Score 0,1,2
        mask = (y <= 2)
        X_filtered = X[mask]
        y_filtered = y[mask].astype(int)

        print(f"\nOriginal data: {len(X)} samples")
        print(f"Filtered data: {len(X_filtered)} samples")
        print(f"Score distribution:")
        for score in range(3):
            count = (y_filtered == score).sum()
            pct = count / len(y_filtered) * 100
            print(f"  Score {score}: {count} samples ({pct:.1f}%)")

        # Extract features
        features = self.extract_all_features(X_filtered)
        print(f"\nFeatures shape: {features.shape}")

        # Split 80/20 train/test
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

        # Test 1: Logistic Regression
        print(f"\n{'='*60}")
        print(f"[1] Logistic Regression (Multinomial)")
        print(f"{'='*60}")
        lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)

        acc_lr = accuracy_score(y_test, y_pred_lr)
        print(f"\nTest Accuracy: {acc_lr:.3f}")

        # Cross-validation
        cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_lr, target_names=['Score 0', 'Score 1', 'Score 2']))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_lr)
        print(cm)

        # Per-class accuracy
        print(f"\nPer-class Accuracy:")
        for i in range(3):
            class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"  Score {i}: {class_acc:.3f}")

        # Test 2: Random Forest
        print(f"\n{'='*60}")
        print(f"[2] Random Forest")
        print(f"{'='*60}")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)

        acc_rf = accuracy_score(y_test, y_pred_rf)
        print(f"\nTest Accuracy: {acc_rf:.3f}")

        cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_rf, target_names=['Score 0', 'Score 1', 'Score 2']))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_rf)
        print(cm)

        print(f"\nPer-class Accuracy:")
        for i in range(3):
            class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"  Score {i}: {class_acc:.3f}")

        # Test 3: Gradient Boosting
        print(f"\n{'='*60}")
        print(f"[3] Gradient Boosting")
        print(f"{'='*60}")
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_train_scaled, y_train)
        y_pred_gb = gb.predict(X_test_scaled)

        acc_gb = accuracy_score(y_test, y_pred_gb)
        print(f"\nTest Accuracy: {acc_gb:.3f}")

        cv_scores = cross_val_score(gb, X_train_scaled, y_train, cv=5)
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_gb, target_names=['Score 0', 'Score 1', 'Score 2']))

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_gb)
        print(cm)

        print(f"\nPer-class Accuracy:")
        for i in range(3):
            class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            print(f"  Score {i}: {class_acc:.3f}")

        # Feature importance (RF)
        print(f"\n{'='*60}")
        print(f"[Feature Importance] (Random Forest)")
        print(f"{'='*60}")
        feature_names = [
            'knee_range_y',
            'mean_velocity',
            'max_velocity',
            'mean_accel',
            'mean_jerk',
            'max_jerk',
            'jerk_std',
            'smoothness',
            'accel_variance',
            'movement_efficiency',
            'velocity_decrement',
            'amplitude_decrement',
            'num_peaks',
            'rhythm_std'
        ]

        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        for idx in sorted_idx:
            print(f"  {feature_names[idx]:25s}: {importances[idx]:.3f}")

        # Baseline comparison
        print(f"\n{'='*60}")
        print(f"[Baseline Comparison]")
        print(f"{'='*60}")
        majority_class = np.bincount(y_train).argmax()
        baseline_acc = (y_test == majority_class).mean()
        print(f"Majority baseline (always predict Score {majority_class}): {baseline_acc:.3f}")
        print(f"\nLogistic Regression improvement: {(acc_lr - baseline_acc):.3f}")
        print(f"Random Forest improvement: {(acc_rf - baseline_acc):.3f}")
        print(f"Gradient Boosting improvement: {(acc_gb - baseline_acc):.3f}")

        print(f"\n{'='*60}")
        print(f"[CONCLUSION]")
        print(f"{'='*60}")

        best_acc = max(acc_lr, acc_rf, acc_gb)
        best_model = ['Logistic Regression', 'Random Forest', 'Gradient Boosting'][[acc_lr, acc_rf, acc_gb].index(best_acc)]

        if best_acc > baseline_acc + 0.15:
            print(f"3-class classification WORKS (best={best_acc:.3f}, {best_model})")
            print(f"-> Score 0, 1, 2 구분 가능!")
            print(f"-> Baseline 대비 {(best_acc - baseline_acc):.3f} 향상")
        elif best_acc > baseline_acc + 0.05:
            print(f"3-class classification MARGINAL (best={best_acc:.3f}, {best_model})")
            print(f"-> 약한 구분 가능")
            print(f"-> Feature engineering 또는 모델 개선 필요")
        else:
            print(f"3-class classification FAILED (best={best_acc:.3f})")
            print(f"-> Score 0, 1, 2 구분 불가")
            print(f"-> Baseline과 차이 없음")


def main():
    tester = MildClassificationTester()
    tester.test_3class_classification()


if __name__ == "__main__":
    main()
