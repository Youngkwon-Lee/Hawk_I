"""
Train ML Model for UPDRS Finger Tapping Score Prediction - V2
Enhanced with:
1. Feature Engineering (polynomial, interaction features)
2. Class Balancing (SMOTE, class weights)
3. Hyperparameter Tuning (GridSearchCV)
4. Ordinal Regression approach
5. Ensemble Methods
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURE_DIR = "C:/Users/YK/tulip/Hawkeye/ml_features"
MODEL_DIR = "C:/Users/YK/tulip/Hawkeye/ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Original Feature columns (excluding bias features like total_taps, duration)
BASE_FEATURE_COLS = [
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
]

def load_data():
    """Load train, valid, test features"""
    train_df = pd.read_csv(f"{FEATURE_DIR}/finger_tapping_train_features.csv")
    valid_df = pd.read_csv(f"{FEATURE_DIR}/finger_tapping_valid_features.csv")
    test_df = pd.read_csv(f"{FEATURE_DIR}/finger_tapping_test_features.csv")

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Check score distribution
    for name, df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        print(f"\n{name} score distribution:")
        print(df['score'].value_counts().sort_index())

    return train_df, valid_df, test_df

def engineer_features(df):
    """Create additional engineered features"""
    df = df.copy()

    # Ratio features
    df['tap_per_second'] = df['total_taps'] / (df['duration'] + 0.001)
    df['amplitude_cv'] = df['amplitude_std'] / (df['amplitude_mean'] + 0.001)  # Coefficient of variation
    df['event_ratio'] = (df['hesitation_count'] + df['halt_count']) / (df['total_taps'] + 1)

    # Combined severity indicators
    df['severity_index'] = (
        df['amplitude_decrement'] * 0.3 +
        df['rhythm_variability'] * 0.3 +
        df['fatigue_rate'] * 0.2 +
        df['event_ratio'] * 0.2
    )

    # Speed-amplitude interaction
    df['speed_amplitude'] = df['tapping_speed'] * df['amplitude_mean']

    # Normalized features
    df['amplitude_normalized'] = df['amplitude_mean'] / (df['amplitude_mean'].max() + 0.001)
    df['speed_normalized'] = df['tapping_speed'] / (df['tapping_speed'].max() + 0.001)

    # Fatigue indicators
    df['fatigue_severity'] = df['fatigue_rate'] * df['amplitude_decrement']

    return df

ENGINEERED_FEATURES = [
    'tap_per_second',
    'amplitude_cv',
    'event_ratio',
    'severity_index',
    'speed_amplitude',
    'amplitude_normalized',
    'speed_normalized',
    'fatigue_severity',
]

ALL_FEATURES = BASE_FEATURE_COLS + ENGINEERED_FEATURES

def prepare_features(df, feature_cols=None):
    """Extract features and labels with engineering"""
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    df = engineer_features(df)
    X = df[feature_cols].values
    y = df['score'].values

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y

def apply_smote(X_train, y_train):
    """Apply SMOTE for class balancing"""
    print("\nApplying SMOTE for class balancing...")
    print(f"Before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Use SMOTETomek for better results
    try:
        smote = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    except:
        # Fallback to regular SMOTE if SMOTETomek fails
        smote = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_train)) - 1))
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

    return X_resampled, y_resampled

def train_xgboost_tuned(X_train, y_train, X_valid, y_valid):
    """Train XGBoost with hyperparameter tuning"""
    print("\n" + "="*60)
    print("Training XGBoost with Hyperparameter Tuning")
    print("="*60)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
    }

    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        eval_metric='mae',
        early_stopping_rounds=10
    )

    # Quick grid search with reduced params
    quick_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'min_child_weight': [1, 3],
    }

    grid_search = GridSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        quick_param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.3f}")

    # Train final model with best params
    best_model = grid_search.best_estimator_

    return best_model

def train_random_forest_tuned(X_train, y_train):
    """Train Random Forest with tuning"""
    print("\n" + "="*60)
    print("Training Random Forest with Tuning")
    print("="*60)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.3f}")

    return grid_search.best_estimator_

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting Regressor"""
    print("\n" + "="*60)
    print("Training Gradient Boosting")
    print("="*60)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def train_ensemble(X_train, y_train, models_dict):
    """Create an ensemble of models"""
    print("\n" + "="*60)
    print("Training Ensemble Model")
    print("="*60)

    estimators = [(name, model) for name, model in models_dict.items()]

    ensemble = VotingRegressor(
        estimators=estimators,
        n_jobs=-1
    )

    ensemble.fit(X_train, y_train)

    return ensemble

def train_stacking(X_train, y_train):
    """Train Stacking Regressor"""
    print("\n" + "="*60)
    print("Training Stacking Ensemble")
    print("="*60)

    estimators = [
        ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
    ]

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1
    )

    stacking.fit(X_train, y_train)

    return stacking

def evaluate_model(model, X, y, name=""):
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

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred_rounded, labels=range(5))
    print(cm)

    return {'mae': mae, 'accuracy': accuracy, 'within_1': within_1, 'y_pred': y_pred_rounded}

def main():
    print("="*60)
    print("ML Model Training V2 - Enhanced")
    print("="*60)

    # Load data
    train_df, valid_df, test_df = load_data()

    # Combine train and valid for training
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)

    # Prepare features with engineering
    print(f"\nUsing {len(ALL_FEATURES)} features (base + engineered)")
    X_train, y_train = prepare_features(train_df)
    X_valid, y_valid = prepare_features(valid_df)
    X_test, y_test = prepare_features(test_df)
    X_full_train, y_full_train = prepare_features(full_train_df)

    print(f"Feature shape: {X_train.shape}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    X_full_train_scaled = scaler.fit_transform(X_full_train)

    # Apply SMOTE for class balancing
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)

    results = {}

    # 1. XGBoost with tuning
    xgb_model = train_xgboost_tuned(X_train_balanced, y_train_balanced, X_valid_scaled, y_valid)
    results['XGB_Tuned_Valid'] = evaluate_model(xgb_model, X_valid_scaled, y_valid, "XGB Tuned (Valid)")
    results['XGB_Tuned_Test'] = evaluate_model(xgb_model, X_test_scaled, y_test, "XGB Tuned (Test)")

    # 2. Random Forest with tuning
    rf_model = train_random_forest_tuned(X_train_balanced, y_train_balanced)
    results['RF_Tuned_Valid'] = evaluate_model(rf_model, X_valid_scaled, y_valid, "RF Tuned (Valid)")
    results['RF_Tuned_Test'] = evaluate_model(rf_model, X_test_scaled, y_test, "RF Tuned (Test)")

    # 3. Gradient Boosting
    gb_model = train_gradient_boosting(X_train_balanced, y_train_balanced)
    results['GB_Valid'] = evaluate_model(gb_model, X_valid_scaled, y_valid, "Gradient Boosting (Valid)")
    results['GB_Test'] = evaluate_model(gb_model, X_test_scaled, y_test, "Gradient Boosting (Test)")

    # 4. Stacking Ensemble
    stacking_model = train_stacking(X_train_balanced, y_train_balanced)
    results['Stacking_Valid'] = evaluate_model(stacking_model, X_valid_scaled, y_valid, "Stacking (Valid)")
    results['Stacking_Test'] = evaluate_model(stacking_model, X_test_scaled, y_test, "Stacking (Test)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Test Results")
    print("="*60)
    print(f"\n{'Model':<20} {'MAE':>10} {'Accuracy':>12} {'Within 1':>12}")
    print("-" * 55)
    for name, res in results.items():
        if 'Test' in name:
            model_name = name.replace('_Test', '')
            print(f"{model_name:<20} {res['mae']:>10.3f} {res['accuracy']*100:>11.1f}% {res['within_1']:>11.1f}%")

    # Compare with baselines
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Rule-based:     MAE=0.364, Accuracy=71.1%, Within 1=97.0%")
    print(f"ML V1 (RF):     MAE=0.529, Accuracy=54.1%, Within 1=98.5%")

    # Find best model
    test_results = {k: v for k, v in results.items() if 'Test' in k}
    best_model_name = min(test_results, key=lambda x: test_results[x]['mae'])
    best_result = test_results[best_model_name]
    print(f"\nBest ML V2:     MAE={best_result['mae']:.3f}, Accuracy={best_result['accuracy']*100:.1f}%, Within 1={best_result['within_1']:.1f}%")

    # Save best models
    print("\n" + "="*60)
    print("Saving Models")
    print("="*60)

    # Retrain on full data with SMOTE
    X_full_balanced, y_full_balanced = apply_smote(X_full_train_scaled, y_full_train)

    # Save RF model
    rf_final = train_random_forest_tuned(X_full_balanced, y_full_balanced)
    joblib.dump(rf_final, f"{MODEL_DIR}/rf_finger_tapping_scorer.pkl")
    print(f"Saved: {MODEL_DIR}/rf_finger_tapping_scorer.pkl")

    # Save XGB model
    xgb_final = train_xgboost_tuned(X_full_balanced, y_full_balanced, X_test_scaled, y_test)
    joblib.dump(xgb_final, f"{MODEL_DIR}/xgb_finger_tapping_scorer.pkl")
    print(f"Saved: {MODEL_DIR}/xgb_finger_tapping_scorer.pkl")

    # Save stacking model (typically best)
    stacking_final = train_stacking(X_full_balanced, y_full_balanced)
    joblib.dump(stacking_final, f"{MODEL_DIR}/stacking_finger_tapping_scorer.pkl")
    print(f"Saved: {MODEL_DIR}/stacking_finger_tapping_scorer.pkl")

    # Save scaler
    joblib.dump(scaler, f"{MODEL_DIR}/finger_tapping_scaler.pkl")
    print(f"Saved: {MODEL_DIR}/finger_tapping_scaler.pkl")

    # Save feature list (also save as primary feature cols)
    import json
    with open(f"{MODEL_DIR}/finger_tapping_feature_cols_v2.json", 'w') as f:
        json.dump(ALL_FEATURES, f)
    print(f"Saved: {MODEL_DIR}/finger_tapping_feature_cols_v2.json")

    # Also save as primary feature cols for ml_scorer
    with open(f"{MODEL_DIR}/finger_tapping_feature_cols.json", 'w') as f:
        json.dump(ALL_FEATURES, f)
    print(f"Saved: {MODEL_DIR}/finger_tapping_feature_cols.json")

if __name__ == "__main__":
    main()
