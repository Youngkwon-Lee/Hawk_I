"""
Train ML Model for UPDRS Gait Score Prediction
With Time-Series Features
"""
import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR, TRAINED_MODELS_DIR, ensure_dirs

# Paths (from centralized config)
FEATURE_DIR = str(FEATURES_DIR)
MODEL_DIR = str(TRAINED_MODELS_DIR)
ensure_dirs()

FEATURE_COLS = [
    "arm_swing_amplitude_mean", "arm_swing_asymmetry", "walking_speed", "cadence",
    "step_height_mean", "step_count", "stride_length", "stride_variability",
    "swing_time_mean", "stance_time_mean", "swing_stance_ratio", "double_support_percent",
    "step_length_asymmetry", "swing_time_asymmetry",
    "trunk_flexion_mean", "trunk_flexion_rom", "hip_flexion_rom_mean",
    "knee_flexion_rom_mean", "ankle_dorsiflexion_rom_mean",
    "step_length_first_half", "step_length_second_half", "step_length_trend",
    "cadence_first_half", "cadence_second_half", "cadence_trend",
    "arm_swing_first_half", "arm_swing_second_half", "arm_swing_trend",
    "stride_variability_first_half", "stride_variability_second_half", "variability_trend",
    "step_height_first_half", "step_height_second_half", "step_height_trend",
]

def load_data():
    train_df = pd.read_csv(f"{FEATURE_DIR}/gait_train_features.csv")
    valid_df = pd.read_csv(f"{FEATURE_DIR}/gait_valid_features.csv")
    test_df = pd.read_csv(f"{FEATURE_DIR}/gait_test_features.csv")
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        print(f"\n{name} score distribution:")
        print(df["score"].value_counts().sort_index())
    return train_df, valid_df, test_df

def prepare_features(df, feature_cols):
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].values
    y = df["score"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, available_cols

def evaluate_model(model, X, y, name="", is_classifier=False):
    if is_classifier:
        y_pred = model.predict(X)
    else:
        y_pred = np.clip(np.round(model.predict(X)), 0, 4).astype(int)
    accuracy = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    within_1 = np.mean(np.abs(y - y_pred) <= 1) * 100
    print(f"\n{name} Results:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  MAE: {mae:.3f}")
    print(f"  Within 1 Point: {within_1:.1f}%")
    print(classification_report(y, y_pred, zero_division=0))
    return {"mae": mae, "accuracy": accuracy, "within_1": within_1}

def main():
    print("="*60)
    print("ML Model Training for UPDRS Gait Prediction")
    print("="*60)
    train_df, valid_df, test_df = load_data()
    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)
    X_train, y_train, used_cols = prepare_features(train_df, FEATURE_COLS)
    X_valid, y_valid, _ = prepare_features(valid_df, FEATURE_COLS)
    X_test, y_test, _ = prepare_features(test_df, FEATURE_COLS)
    X_full_train, y_full_train, _ = prepare_features(full_train_df, FEATURE_COLS)
    print(f"\nFeature shape: {X_train.shape}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    results = {}

    print("\nTraining XGBoost Regressor...")
    xgb_reg = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_reg.fit(X_train_scaled, y_train)
    results["XGB_Reg_Test"] = evaluate_model(xgb_reg, X_test_scaled, y_test, "XGBoost Regressor (Test)")

    print("\nTraining Random Forest Regressor...")
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_scaled, y_train)
    results["RF_Reg_Test"] = evaluate_model(rf_reg, X_test_scaled, y_test, "Random Forest Regressor (Test)")

    print("\nTraining XGBoost Classifier...")
    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, objective="multi:softmax", num_class=5)
    xgb_clf.fit(X_train_scaled, y_train)
    results["XGB_Clf_Test"] = evaluate_model(xgb_clf, X_test_scaled, y_test, "XGBoost Classifier (Test)", is_classifier=True)

    importances = rf_reg.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop 10 Features:")
    for i in range(min(10, len(used_cols))):
        print(f"  {i+1}. {used_cols[indices[i]]}: {importances[indices[i]]:.4f}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, res in results.items():
        mae_val = res["mae"]
        acc_val = res["accuracy"]*100
        w1_val = res["within_1"]
        print(f"{name}: MAE={mae_val:.3f}, Acc={acc_val:.1f}%, Within1={w1_val:.1f}%")

    scaler_final = StandardScaler()
    X_full_scaled = scaler_final.fit_transform(X_full_train)
    rf_final = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf_final.fit(X_full_scaled, y_full_train)
    joblib.dump(rf_final, f"{MODEL_DIR}/rf_gait_scorer.pkl")
    joblib.dump(scaler_final, f"{MODEL_DIR}/gait_scaler.pkl")
    with open(f"{MODEL_DIR}/gait_feature_cols.json", "w") as f:
        json.dump(used_cols, f)
    print("\nModels saved!")

if __name__ == "__main__":
    main()
