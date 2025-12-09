"""
Train ML Model for UPDRS Finger Tapping Score Prediction
Using XGBoost and Random Forest on kinematic features
"""
import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR, TRAINED_MODELS_DIR, ensure_dirs

# Paths (from centralized config)
FEATURE_DIR = str(FEATURES_DIR)
MODEL_DIR = str(TRAINED_MODELS_DIR)
ensure_dirs()

# Feature columns (excluding metadata and bias features)
FEATURE_COLS = [
    # Speed
    'tapping_speed',
    # Amplitude features
    'amplitude_mean',
    'amplitude_std',
    'amplitude_decrement',
    'first_half_amplitude',
    'second_half_amplitude',
    # Velocity features (MDS-UPDRS critical)
    'opening_velocity_mean',
    'closing_velocity_mean',
    'peak_velocity_mean',
    'velocity_decrement',
    # Rhythm features
    'rhythm_variability',
    # Events
    'hesitation_count',
    'halt_count',
    'freeze_episodes',
    # Fatigue
    'fatigue_rate',
    # === TIME-SERIES FEATURES ===
    # Temporal segments
    'velocity_first_third',
    'velocity_mid_third',
    'velocity_last_third',
    'amplitude_first_third',
    'amplitude_mid_third',
    'amplitude_last_third',
    # Trend slopes
    'velocity_slope',
    'amplitude_slope',
    'rhythm_slope',
    # Variability progression
    'variability_first_half',
    'variability_second_half',
    'variability_change',
]

def load_data(use_stratified=True):
    """Load train, valid, test features

    Args:
        use_stratified: Use stratified split (recommended, no data leakage)
    """
    suffix = "_stratified" if use_stratified else ""

    train_df = pd.read_csv(f"{FEATURE_DIR}/finger_tapping_train_features{suffix}.csv")
    valid_df = pd.read_csv(f"{FEATURE_DIR}/finger_tapping_valid_features{suffix}.csv")
    test_df = pd.read_csv(f"{FEATURE_DIR}/finger_tapping_test_features{suffix}.csv")

    print(f"Using {'STRATIFIED' if use_stratified else 'ORIGINAL'} split")
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Check score distribution
    for name, df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        print(f"\n{name} score distribution:")
        print(df['score'].value_counts().sort_index())

    return train_df, valid_df, test_df

def prepare_features(df):
    """Extract features and labels"""
    X = df[FEATURE_COLS].values
    y = df['score'].values

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y

def train_xgboost_regressor(X_train, y_train, X_valid, y_valid):
    """Train XGBoost Regressor"""
    print("\n" + "="*60)
    print("Training XGBoost Regressor")
    print("="*60)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42,
        eval_metric='mae'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=True
    )

    return model

def train_random_forest_regressor(X_train, y_train):
    """Train Random Forest Regressor"""
    print("\n" + "="*60)
    print("Training Random Forest Regressor")
    print("="*60)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

def train_xgboost_classifier(X_train, y_train, X_valid, y_valid, num_classes):
    """Train XGBoost Classifier"""
    print("\n" + "="*60)
    print("Training XGBoost Classifier")
    print("="*60)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=num_classes,
        random_state=42,
        eval_metric='mlogloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=True
    )

    return model

def evaluate_regressor(model, X, y, name=""):
    """Evaluate regression model"""
    y_pred = model.predict(X)
    y_pred_rounded = np.clip(np.round(y_pred), 0, 4)

    mae = mean_absolute_error(y, y_pred)
    accuracy = accuracy_score(y, y_pred_rounded)

    # Within 1 point accuracy
    within_1 = np.mean(np.abs(y - y_pred_rounded) <= 1) * 100

    print(f"\n{name} Results:")
    print(f"  MAE: {mae:.3f}")
    print(f"  Exact Accuracy: {accuracy*100:.1f}%")
    print(f"  Within 1 Point: {within_1:.1f}%")

    return {'mae': mae, 'accuracy': accuracy, 'within_1': within_1}

def evaluate_classifier(model, X, y, name=""):
    """Evaluate classification model"""
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

def print_feature_importance(model, model_name):
    """Print feature importance"""
    print(f"\n{model_name} Feature Importance:")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print("-" * 40)
    for i, idx in enumerate(indices[:10]):
        print(f"  {i+1}. {FEATURE_COLS[idx]}: {importances[idx]:.4f}")

def main():
    print("="*60)
    print("ML Model Training for UPDRS Finger Tapping Prediction")
    print("="*60)

    # Load data
    train_df, valid_df, test_df = load_data()

    # Combine train and valid for final training
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)
    X_test, y_test = prepare_features(test_df)
    X_full_train, y_full_train = prepare_features(full_train_df)

    print(f"\nFeature shape: {X_train.shape}")

    # Get number of classes
    num_classes = len(np.unique(np.concatenate([y_train, y_valid, y_test])))
    print(f"Number of classes: {num_classes}")

    # Train models
    results = {}

    # 1. XGBoost Regressor
    xgb_reg = train_xgboost_regressor(X_train, y_train, X_valid, y_valid)
    results['XGB_Reg_Valid'] = evaluate_regressor(xgb_reg, X_valid, y_valid, "XGB Regressor (Valid)")
    results['XGB_Reg_Test'] = evaluate_regressor(xgb_reg, X_test, y_test, "XGB Regressor (Test)")
    print_feature_importance(xgb_reg, "XGBoost Regressor")

    # 2. Random Forest Regressor
    rf_reg = train_random_forest_regressor(X_train, y_train)
    results['RF_Reg_Valid'] = evaluate_regressor(rf_reg, X_valid, y_valid, "RF Regressor (Valid)")
    results['RF_Reg_Test'] = evaluate_regressor(rf_reg, X_test, y_test, "RF Regressor (Test)")
    print_feature_importance(rf_reg, "Random Forest Regressor")

    # 3. XGBoost Classifier
    xgb_clf = train_xgboost_classifier(X_train, y_train, X_valid, y_valid, num_classes)
    results['XGB_Clf_Valid'] = evaluate_classifier(xgb_clf, X_valid, y_valid, "XGB Classifier (Valid)")
    results['XGB_Clf_Test'] = evaluate_classifier(xgb_clf, X_test, y_test, "XGB Classifier (Test)")

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

    # Compare with Rule-based (97.0% within 1, MAE 0.364)
    print("\n" + "="*60)
    print("COMPARISON WITH RULE-BASED")
    print("="*60)
    print(f"Rule-based: MAE=0.364, Within 1=97.0%")
    best_ml_mae = min(results['XGB_Reg_Test']['mae'], results['RF_Reg_Test']['mae'])
    best_ml_within1 = max(results['XGB_Reg_Test']['within_1'], results['RF_Reg_Test']['within_1'])
    print(f"Best ML:    MAE={best_ml_mae:.3f}, Within 1={best_ml_within1:.1f}%")

    # Save best model (XGBoost Regressor typically best for ordinal prediction)
    print("\n" + "="*60)
    print("Saving Models")
    print("="*60)

    # Retrain on full data
    xgb_final = train_xgboost_regressor(X_full_train, y_full_train, X_test, y_test)
    joblib.dump(xgb_final, f"{MODEL_DIR}/xgb_finger_tapping_scorer.pkl")
    print(f"Saved: {MODEL_DIR}/xgb_finger_tapping_scorer.pkl")

    rf_final = train_random_forest_regressor(X_full_train, y_full_train)
    joblib.dump(rf_final, f"{MODEL_DIR}/rf_finger_tapping_scorer.pkl")
    print(f"Saved: {MODEL_DIR}/rf_finger_tapping_scorer.pkl")

    # Save feature columns for inference
    import json
    with open(f"{MODEL_DIR}/finger_tapping_feature_cols.json", 'w') as f:
        json.dump(FEATURE_COLS, f)
    print(f"Saved: {MODEL_DIR}/finger_tapping_feature_cols.json")

if __name__ == "__main__":
    main()
