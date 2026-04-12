"""
Leg Agility Binary Classification Test
- Severe (Score 3,4) vs Mild (Score 0,1,2)
- Test if ML can at least distinguish severe from mild cases
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class BinaryClassificationTester:
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
        """Extract knee trajectory + jerk + smoothness features"""
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

            # 4. Jerk (3rd derivative)
            jerk = np.diff(acceleration)
            mean_jerk = np.abs(jerk).mean() if len(jerk) > 0 else 0
            max_jerk = np.abs(jerk).max() if len(jerk) > 0 else 0

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

            features.append([
                range_y,
                mean_velocity,
                max_velocity,
                mean_accel,
                mean_jerk,
                max_jerk,
                smoothness,
                accel_variance,
                efficiency,
                velocity_decrement,
                amplitude_decrement
            ])

        return np.array(features)

    def convert_to_binary(self, y):
        """Convert to binary: 0=Mild (0,1,2), 1=Severe (3,4)"""
        return (y >= 3).astype(int)

    def test_binary_classification(self):
        """Test binary classification"""
        print(f"\n{'='*60}")
        print(f"Binary Classification Test: Severe (3,4) vs Mild (0,1,2)")
        print(f"{'='*60}")

        # Load data
        X, y = self.load_data()
        print(f"\nData shape: {X.shape}")
        print(f"Original score distribution: {np.bincount(y.astype(int))}")

        # Convert to binary
        y_binary = self.convert_to_binary(y)
        print(f"\nBinary distribution:")
        print(f"  Mild (0,1,2): {(y_binary == 0).sum()} samples")
        print(f"  Severe (3,4): {(y_binary == 1).sum()} samples")

        # Extract features
        features = self.extract_all_features(X)
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
        y_train = y_binary[train_idx]
        y_test = y_binary[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

        # Test 1: Logistic Regression
        print(f"\n{'='*60}")
        print(f"[1] Logistic Regression")
        print(f"{'='*60}")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)

        acc_lr = accuracy_score(y_test, y_pred_lr)
        print(f"\nAccuracy: {acc_lr:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_lr, target_names=['Mild (0,1,2)', 'Severe (3,4)']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_lr))

        # Test 2: Random Forest
        print(f"\n{'='*60}")
        print(f"[2] Random Forest")
        print(f"{'='*60}")
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)

        acc_rf = accuracy_score(y_test, y_pred_rf)
        print(f"\nAccuracy: {acc_rf:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_rf, target_names=['Mild (0,1,2)', 'Severe (3,4)']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred_rf))

        # Feature importance
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
            'smoothness',
            'accel_variance',
            'movement_efficiency',
            'velocity_decrement',
            'amplitude_decrement'
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
        print(f"Majority baseline (always predict {['Mild', 'Severe'][majority_class]}): {baseline_acc:.3f}")
        print(f"\nLogistic Regression improvement: {(acc_lr - baseline_acc):.3f}")
        print(f"Random Forest improvement: {(acc_rf - baseline_acc):.3f}")

        print(f"\n{'='*60}")
        print(f"[CONCLUSION]")
        print(f"{'='*60}")
        if acc_rf > 0.7:
            print(f"Binary classification WORKS (RF acc={acc_rf:.3f})")
            print(f"-> Skeleton features CAN distinguish severe from mild")
            print(f"-> Problem: 5-class regression too fine-grained")
            print(f"-> Recommendation: Use binary or 3-class model")
        elif acc_rf > baseline_acc + 0.1:
            print(f"Binary classification MARGINAL (RF acc={acc_rf:.3f})")
            print(f"-> Features provide some signal but weak")
            print(f"-> Recommendation: Investigate labeling quality")
        else:
            print(f"Binary classification FAILED (RF acc={acc_rf:.3f})")
            print(f"-> Features provide NO useful signal")
            print(f"-> Fundamental problem: Labeling or skeleton extraction")
            print(f"-> Recommendation: ABANDON Leg Agility task")


def main():
    tester = BinaryClassificationTester()
    tester.test_binary_classification()


if __name__ == "__main__":
    main()
