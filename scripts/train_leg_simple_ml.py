"""
Train simple ML models (RF, XGBoost) for Leg Agility
Compare with deep learning models
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
from pathlib import Path

# Auto-detect data path
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
data_dir = base_dir / 'data'

# Load data
train_path = data_dir / 'leg_agility_train_v2.pkl'
test_path = data_dir / 'leg_agility_test_v2.pkl'

with open(train_path, 'rb') as f:
    train_data = pickle.load(f)
with open(test_path, 'rb') as f:
    test_data = pickle.load(f)

X_train = train_data['X']  # (N, 150, 18)
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

print("="*60)
print("Simple ML Models for Leg Agility")
print("="*60)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Flatten sequences to 2D
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (N, 150*18=2700)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"Flattened: Train {X_train_flat.shape}, Test {X_test_flat.shape}")

# 1. Random Forest
print("\n" + "="*60)
print("1. Random Forest Regressor")
print("="*60)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf.fit(X_train_flat, y_train)
y_pred_rf = rf.predict(X_test_flat)
y_pred_rf = np.clip(np.round(y_pred_rf), 0, 4).astype(int)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
exact_rf = np.mean(y_pred_rf == y_test) * 100
within1_rf = np.mean(np.abs(y_pred_rf - y_test) <= 1) * 100
pearson_rf, _ = pearsonr(y_test, y_pred_rf)
spearman_rf, _ = spearmanr(y_test, y_pred_rf)

print(f"\nRandom Forest Results:")
print(f"  MAE: {mae_rf:.3f}")
print(f"  Exact: {exact_rf:.1f}%")
print(f"  Within1: {within1_rf:.1f}%")
print(f"  Pearson: {pearson_rf:.3f}")
print(f"  Spearman: {spearman_rf:.3f}")

# 2. XGBoost
print("\n" + "="*60)
print("2. XGBoost Regressor")
print("="*60)

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

xgb_model.fit(X_train_flat, y_train)
y_pred_xgb = xgb_model.predict(X_test_flat)
y_pred_xgb = np.clip(np.round(y_pred_xgb), 0, 4).astype(int)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
exact_xgb = np.mean(y_pred_xgb == y_test) * 100
within1_xgb = np.mean(np.abs(y_pred_xgb - y_test) <= 1) * 100
pearson_xgb, _ = pearsonr(y_test, y_pred_xgb)
spearman_xgb, _ = spearmanr(y_test, y_pred_xgb)

print(f"\nXGBoost Results:")
print(f"  MAE: {mae_xgb:.3f}")
print(f"  Exact: {exact_xgb:.1f}%")
print(f"  Within1: {within1_xgb:.1f}%")
print(f"  Pearson: {pearson_xgb:.3f}")
print(f"  Spearman: {spearman_xgb:.3f}")

# Summary comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"{'Model':<30} {'MAE':<8} {'Exact':<10} {'Pearson':<10}")
print("-" * 60)
print(f"{'Random Forest':<30} {mae_rf:<8.3f} {exact_rf:<10.1f} {pearson_rf:<10.3f}")
print(f"{'XGBoost':<30} {mae_xgb:<8.3f} {exact_xgb:<10.1f} {pearson_xgb:<10.3f}")
print(f"{'Mamba + CORAL (HPC)':<30} {'0.458':<8} {'57.9%':<10} {'0.307':<10}")

# Save models
output_dir = base_dir / 'models' / 'trained'
output_dir.mkdir(parents=True, exist_ok=True)

rf_path = output_dir / 'rf_leg_agility_v2.pkl'
xgb_path = output_dir / 'xgb_leg_agility_v2.pkl'

with open(rf_path, 'wb') as f:
    pickle.dump(rf, f)
with open(xgb_path, 'wb') as f:
    pickle.dump(xgb_model, f)

print(f"\n✅ Models saved:")
print(f"  {rf_path}")
print(f"  {xgb_path}")

print("\n" + "="*60)
