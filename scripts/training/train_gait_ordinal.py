"""
Train Ordinal Regression Model for UPDRS Gait Score Prediction
Ordinal Regression leverages the natural ordering of UPDRS scores (0 < 1 < 2 < 3 < 4)
"""
import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR, TRAINED_MODELS_DIR, ensure_dirs

# Paths (from centralized config)
FEATURE_DIR = str(FEATURES_DIR)
MODEL_DIR = str(TRAINED_MODELS_DIR)
ensure_dirs()

# Feature columns (with time-series features)
FEATURE_COLS = [
    # Core gait metrics
    'step_count',
    'cadence',
    'gait_speed',
    'stride_length',
    'step_length_asymmetry',
    'stride_time_variability',
    'double_support_time',
    'arm_swing_amplitude',
    'arm_swing_asymmetry',
    'trunk_sway',
    # Angle features
    'hip_flexion_mean',
    'hip_flexion_range',
    'knee_flexion_mean',
    'knee_flexion_range',
    'ankle_flexion_mean',
    'ankle_flexion_range',
    # Time-series features
    'step_length_first_half',
    'step_length_second_half',
    'step_length_trend',
    'cadence_first_half',
    'cadence_second_half',
    'cadence_trend',
    'arm_swing_first_half',
    'arm_swing_second_half',
    'arm_swing_trend',
    'stride_variability_first_half',
    'stride_variability_second_half',
    'variability_trend',
    'step_height_first_half',
    'step_height_second_half',
    'step_height_trend',
]


class OrdinalClassifier:
    """
    Ordinal Regression using cumulative probability model.
    For K classes, we train K-1 binary classifiers:
    - Classifier 1: P(Y > 0)
    - Classifier 2: P(Y > 1)
    - ...
    - Classifier K-1: P(Y > K-2)

    Final prediction uses: P(Y = k) = P(Y > k-1) - P(Y > k)
    """

    def __init__(self, base_classifier=None):
        if base_classifier is None:
            self.base_classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.base_classifier = base_classifier
        self.classifiers = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        n_classes = len(self.classes_)

        # Train K-1 binary classifiers
        self.classifiers = []
        for k in range(n_classes - 1):
            # Binary target: 1 if y > class[k], else 0
            y_binary = (y > self.classes_[k]).astype(int)

            # Clone the base classifier
            from sklearn.base import clone
            clf = clone(self.base_classifier)
            clf.fit(X, y_binary)
            self.classifiers.append(clf)

        return self

    def predict_proba(self, X):
        """
        Calculate P(Y = k) for each class k.
        P(Y = k) = P(Y > k-1) - P(Y > k)
        where P(Y > -1) = 1 and P(Y > K-1) = 0
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Get cumulative probabilities P(Y > k)
        cumprobs = np.zeros((n_samples, n_classes + 1))
        cumprobs[:, 0] = 1  # P(Y > -1) = 1

        for k, clf in enumerate(self.classifiers):
            cumprobs[:, k + 1] = clf.predict_proba(X)[:, 1]

        # cumprobs[:, -1] = 0  # P(Y > K-1) = 0 (already zeros)

        # Calculate P(Y = k) = P(Y > k-1) - P(Y > k)
        probs = np.diff(-cumprobs, axis=1)

        # Ensure non-negative probabilities
        probs = np.clip(probs, 0, 1)

        # Normalize
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def load_data():
    """Load train, valid, test features"""
    train_df = pd.read_csv(f"{FEATURE_DIR}/gait_train_features.csv")
    valid_df = pd.read_csv(f"{FEATURE_DIR}/gait_valid_features.csv")
    test_df = pd.read_csv(f"{FEATURE_DIR}/gait_test_features.csv")

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Check score distribution
    for name, df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        print(f"\n{name} score distribution:")
        print(df['score'].value_counts().sort_index())

    return train_df, valid_df, test_df


def prepare_features(df, feature_cols):
    """Extract features and labels"""
    # Filter to only use columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")

    X = df[available_cols].values
    y = df['score'].values

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, available_cols


def evaluate_model(model, X, y, name=""):
    """Evaluate model"""
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    within_1 = np.mean(np.abs(y - y_pred) <= 1) * 100

    print(f"\n{name} Results:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  MAE: {mae:.3f}")
    print(f"  Within 1 Point: {within_1:.1f}%")
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    return {'mae': mae, 'accuracy': accuracy, 'within_1': within_1}


def main():
    print("="*60)
    print("Ordinal Regression for UPDRS Gait Prediction")
    print("="*60)

    # Load data
    train_df, valid_df, test_df = load_data()

    # Combine train and valid for final training
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)

    # Prepare features
    X_train, y_train, used_cols = prepare_features(train_df, FEATURE_COLS)
    X_valid, y_valid, _ = prepare_features(valid_df, FEATURE_COLS)
    X_test, y_test, _ = prepare_features(test_df, FEATURE_COLS)
    X_full_train, y_full_train, _ = prepare_features(full_train_df, FEATURE_COLS)

    print(f"\nFeature shape: {X_train.shape}")
    print(f"Features used: {len(used_cols)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    X_full_train_scaled = scaler.fit_transform(X_full_train)

    # Train Ordinal Regression
    print("\n" + "="*60)
    print("Training Ordinal Regression")
    print("="*60)

    ordinal_clf = OrdinalClassifier()
    ordinal_clf.fit(X_train_scaled, y_train)

    results = {}
    results['Ordinal_Valid'] = evaluate_model(ordinal_clf, X_valid_scaled, y_valid, "Ordinal Regression (Valid)")
    results['Ordinal_Test'] = evaluate_model(ordinal_clf, X_test_scaled, y_test, "Ordinal Regression (Test)")

    # Compare with standard Logistic Regression
    print("\n" + "="*60)
    print("Training Standard Logistic Regression (baseline)")
    print("="*60)

    lr_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_clf.fit(X_train_scaled, y_train)

    results['LR_Valid'] = evaluate_model(lr_clf, X_valid_scaled, y_valid, "Logistic Regression (Valid)")
    results['LR_Test'] = evaluate_model(lr_clf, X_test_scaled, y_test, "Logistic Regression (Test)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Model':<25} {'Test MAE':>10} {'Test Acc':>10} {'Within 1':>10}")
    print("-" * 55)
    for name, res in results.items():
        if 'Test' in name:
            model_name = name.replace('_Test', '')
            print(f"{model_name:<25} {res['mae']:>10.3f} {res['accuracy']*100:>9.1f}% {res['within_1']:>9.1f}%")

    # Save models
    print("\n" + "="*60)
    print("Saving Models")
    print("="*60)

    # Retrain on full data
    scaler_final = StandardScaler()
    X_full_scaled = scaler_final.fit_transform(X_full_train)

    ordinal_final = OrdinalClassifier()
    ordinal_final.fit(X_full_scaled, y_full_train)

    joblib.dump(ordinal_final, f"{MODEL_DIR}/ordinal_gait_scorer.pkl")
    print(f"Saved: {MODEL_DIR}/ordinal_gait_scorer.pkl")

    joblib.dump(scaler_final, f"{MODEL_DIR}/ordinal_gait_scaler.pkl")
    print(f"Saved: {MODEL_DIR}/ordinal_gait_scaler.pkl")


if __name__ == "__main__":
    main()
