#!/usr/bin/env python3
"""
Gait Optimized Training - Feature Selection + New Features
Based on feature importance analysis:
- Remove low importance features (hip_height, individual angles)
- Add new derived features (asymmetry ratios, variability indices)
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70, flush=True)
print("Gait Optimized Training - Feature Selection + New Features", flush=True)
print("=" * 70, flush=True)

# Load data
data_dir = Path("C:/Users/YK/tulip/Hawkeye/hpc/data")

with open(data_dir / "gait_train_v2.pkl", 'rb') as f:
    train_data = pickle.load(f)
with open(data_dir / "gait_valid_v2.pkl", 'rb') as f:
    valid_data = pickle.load(f)
with open(data_dir / "gait_test_v2.pkl", 'rb') as f:
    test_data = pickle.load(f)

features = train_data['features']
clinical_idx = list(range(99, 129))
clinical_features = [features[i] for i in clinical_idx]

print("\nOriginal clinical features (30):", flush=True)
for i, f in enumerate(clinical_features):
    print(f"  {i}: {f}", flush=True)

# Top features to KEEP (importance > 3%)
top_features = [
    'trunk_angle_vel',    # 7.9%
    'body_sway_vel',      # 7.6%
    'left_arm_swing_vel', # 5.1%
    'stride_proxy_vel',   # 5.1%
    'stride_proxy',       # 5.0%
    'right_hip_angle_vel',# 4.8%
    'step_width_vel',     # 4.8%
    'step_width',         # 4.6%
    'body_sway',          # 4.1%
    'right_knee_angle_vel', # 4.0%
    'right_arm_swing_vel',  # 3.2%
    'right_arm_swing',      # 3.1%
]

# Features to REMOVE (low importance < 2.5%)
remove_features = [
    'hip_height',           # 1.6% - almost constant
    'left_hip_angle',       # 1.8% - individual angle, asymmetry is more useful
    'right_hip_angle',      # 2.0%
    'shoulder_asymmetry',   # 2.0% - less useful than others
]

# Keep features
keep_features = [f for f in clinical_features if f not in remove_features]
keep_idx = [clinical_idx[i] for i, f in enumerate(clinical_features) if f not in remove_features]

print(f"\nKept features: {len(keep_features)}", flush=True)
print(f"Removed features: {remove_features}", flush=True)

def extract_features_with_derived(X_3d, keep_idx, all_idx, feature_names):
    """Extract features with additional derived features"""
    n_samples = X_3d.shape[0]
    all_features = []

    for sample_idx in range(n_samples):
        sample = X_3d[sample_idx]  # (seq_len, all_features)
        feats = []

        # 1. Standard aggregated features for kept features
        for idx in keep_idx:
            col = sample[:, idx]
            feats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75)
            ])

        # 2. NEW: Coefficient of Variation (CV) for key features
        # CV = std/mean - captures variability normalized by magnitude
        key_velocity_idx = [
            all_idx[feature_names.index('trunk_angle_vel')],
            all_idx[feature_names.index('body_sway_vel')],
            all_idx[feature_names.index('stride_proxy_vel')],
        ]
        for idx in key_velocity_idx:
            col = sample[:, idx]
            mean_val = np.mean(np.abs(col)) + 1e-6
            cv = np.std(col) / mean_val
            feats.append(cv)

        # 3. NEW: Arm swing asymmetry ratio
        left_arm_idx = all_idx[feature_names.index('left_arm_swing')]
        right_arm_idx = all_idx[feature_names.index('right_arm_swing')]
        left_arm = sample[:, left_arm_idx]
        right_arm = sample[:, right_arm_idx]
        arm_asymmetry = np.mean(np.abs(left_arm - right_arm)) / (np.mean(np.abs(left_arm) + np.abs(right_arm)) + 1e-6)
        feats.append(arm_asymmetry)

        # 4. NEW: Knee angle asymmetry
        left_knee_idx = all_idx[feature_names.index('left_knee_angle')]
        right_knee_idx = all_idx[feature_names.index('right_knee_angle')]
        left_knee = sample[:, left_knee_idx]
        right_knee = sample[:, right_knee_idx]
        knee_asymmetry = np.mean(np.abs(left_knee - right_knee)) / (np.mean(np.abs(left_knee) + np.abs(right_knee)) + 1e-6)
        feats.append(knee_asymmetry)

        # 5. NEW: Gait rhythm regularity (autocorrelation of stride_proxy)
        stride_idx = all_idx[feature_names.index('stride_proxy')]
        stride = sample[:, stride_idx]
        if len(stride) > 10:
            # Simple autocorrelation at lag 5
            autocorr = np.corrcoef(stride[:-5], stride[5:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
        else:
            autocorr = 0
        feats.append(autocorr)

        # 6. NEW: Movement smoothness (jerk - derivative of velocity)
        trunk_vel_idx = all_idx[feature_names.index('trunk_angle_vel')]
        trunk_vel = sample[:, trunk_vel_idx]
        jerk = np.diff(trunk_vel)
        jerk_rms = np.sqrt(np.mean(jerk**2)) if len(jerk) > 0 else 0
        feats.append(jerk_rms)

        # 7. NEW: Freezing index (ratio of low movement periods)
        body_sway_idx = all_idx[feature_names.index('body_sway')]
        body_sway = sample[:, body_sway_idx]
        threshold = np.percentile(np.abs(body_sway), 10)
        freeze_ratio = np.mean(np.abs(body_sway) < threshold)
        feats.append(freeze_ratio)

        # 8. NEW: Cadence variability
        step_vel_idx = all_idx[feature_names.index('step_width_vel')]
        step_vel = sample[:, step_vel_idx]
        # Detect zero crossings as proxy for steps
        zero_crossings = np.sum(np.abs(np.diff(np.sign(step_vel))) > 0)
        cadence = zero_crossings / (len(step_vel) / 30)  # Assuming 30fps
        feats.append(cadence)

        all_features.append(feats)

    return np.array(all_features)

print("\nExtracting optimized features...", flush=True)
X_train = extract_features_with_derived(train_data['X'], keep_idx, clinical_idx, clinical_features)
X_valid = extract_features_with_derived(valid_data['X'], keep_idx, clinical_idx, clinical_features)
X_test = extract_features_with_derived(test_data['X'], keep_idx, clinical_idx, clinical_features)

X = np.vstack([X_train, X_valid, X_test])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])

# Replace NaN/Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

n_base = len(keep_features) * 7
n_derived = X.shape[1] - n_base
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
print(f"  - Base features: {n_base} ({len(keep_features)} clinical × 7 stats)", flush=True)
print(f"  - Derived features: {n_derived} (CV, asymmetries, rhythm, smoothness, freeze, cadence)", flush=True)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test models
print("\n" + "=" * 70, flush=True)
print("5-Fold StratifiedKFold Cross-Validation", flush=True)
print("=" * 70, flush=True)

models = {
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'RF_500': RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'GB_200': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n[{name}]", flush=True)
    fold_scores = []
    fold_mae = []
    fold_within1 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        m = type(model)(**model.get_params())
        m.fit(X_scaled[train_idx], y[train_idx])
        y_pred = m.predict(X_scaled[val_idx])

        fold_scores.append(accuracy_score(y[val_idx], y_pred))
        fold_mae.append(mean_absolute_error(y[val_idx], y_pred))
        fold_within1.append(np.mean(np.abs(y[val_idx] - y_pred) <= 1))

    results[name] = {
        'accuracy': np.mean(fold_scores),
        'accuracy_std': np.std(fold_scores),
        'mae': np.mean(fold_mae),
        'within1': np.mean(fold_within1)
    }
    print(f"  Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}, MAE={np.mean(fold_mae):.3f}, Within-1={np.mean(fold_within1):.3f}", flush=True)

# Summary
print("\n" + "=" * 70, flush=True)
print("RESULTS COMPARISON", flush=True)
print("=" * 70, flush=True)

baseline_orig = 0.671
baseline_clinical = 0.725

for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    print(f"{name:<15} {acc:.1%} (vs orig: {acc-baseline_orig:+.1%}, vs clinical: {acc-baseline_clinical:+.1%})", flush=True)

best = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest: {best[0]} with {best[1]['accuracy']:.1%}", flush=True)
print(f"Previous best (clinical only): 72.5%", flush=True)

# Save
results_path = data_dir.parent / "results" / "gait_optimized_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'kept_features': keep_features,
        'removed_features': remove_features,
        'n_derived_features': n_derived
    }, f)
print(f"\nSaved to: {results_path}", flush=True)
