#!/usr/bin/env python3
"""
Gait v7 Training - Enhanced Asymmetry Features
Add comprehensive left-right differences for all bilateral features
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
print("Gait v7 Training - Enhanced Asymmetry Features", flush=True)
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

# Remove low importance, keep useful
remove_features = ['hip_height', 'left_hip_angle', 'right_hip_angle', 'shoulder_asymmetry']
keep_features = [f for f in clinical_features if f not in remove_features]
keep_idx = [clinical_idx[i] for i, f in enumerate(clinical_features) if f not in remove_features]

print(f"Base features: {len(keep_features)}", flush=True)

def extract_features_v7(X_3d, keep_idx, all_idx, feature_names):
    """Extract with comprehensive asymmetry features"""
    n_samples = X_3d.shape[0]
    all_features = []

    for sample_idx in range(n_samples):
        sample = X_3d[sample_idx]
        feats = []

        # 1. Standard aggregated features
        for idx in keep_idx:
            col = sample[:, idx]
            feats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75)
            ])

        # 2. CV for key velocity features
        for fname in ['trunk_angle_vel', 'body_sway_vel', 'stride_proxy_vel']:
            idx = all_idx[feature_names.index(fname)]
            col = sample[:, idx]
            cv = np.std(col) / (np.mean(np.abs(col)) + 1e-6)
            feats.append(cv)

        # 3. COMPREHENSIVE ASYMMETRY FEATURES
        bilateral_pairs = [
            ('left_arm_swing', 'right_arm_swing'),
            ('left_knee_angle', 'right_knee_angle'),
            ('left_ankle_height', 'right_ankle_height'),
            ('left_arm_swing_vel', 'right_arm_swing_vel'),
            ('left_knee_angle_vel', 'right_knee_angle_vel'),
            ('left_ankle_height_vel', 'right_ankle_height_vel'),
        ]

        for left_name, right_name in bilateral_pairs:
            left_idx = all_idx[feature_names.index(left_name)]
            right_idx = all_idx[feature_names.index(right_name)]
            left = sample[:, left_idx]
            right = sample[:, right_idx]

            # Asymmetry ratio (normalized difference)
            diff = left - right
            total = np.abs(left) + np.abs(right) + 1e-6
            asymmetry_ratio = np.mean(np.abs(diff) / total)
            feats.append(asymmetry_ratio)

            # Mean difference
            mean_diff = np.mean(diff)
            feats.append(mean_diff)

            # Std of difference (variability in asymmetry)
            std_diff = np.std(diff)
            feats.append(std_diff)

            # Phase difference (cross-correlation lag)
            if len(left) > 20:
                cross_corr = np.correlate(left - np.mean(left), right - np.mean(right), mode='same')
                lag = np.argmax(cross_corr) - len(cross_corr) // 2
                phase_diff = lag / len(left)  # Normalized
            else:
                phase_diff = 0
            feats.append(phase_diff)

        # 4. Gait rhythm (autocorrelation)
        stride_idx = all_idx[feature_names.index('stride_proxy')]
        stride = sample[:, stride_idx]
        if len(stride) > 10:
            autocorr = np.corrcoef(stride[:-5], stride[5:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
        else:
            autocorr = 0
        feats.append(autocorr)

        # 5. Movement smoothness (jerk)
        trunk_vel_idx = all_idx[feature_names.index('trunk_angle_vel')]
        trunk_vel = sample[:, trunk_vel_idx]
        jerk = np.diff(trunk_vel)
        jerk_rms = np.sqrt(np.mean(jerk**2)) if len(jerk) > 0 else 0
        feats.append(jerk_rms)

        # 6. Freezing index
        body_sway_idx = all_idx[feature_names.index('body_sway')]
        body_sway = sample[:, body_sway_idx]
        threshold = np.percentile(np.abs(body_sway), 10)
        freeze_ratio = np.mean(np.abs(body_sway) < threshold)
        feats.append(freeze_ratio)

        # 7. Cadence (step frequency proxy)
        step_vel_idx = all_idx[feature_names.index('step_width_vel')]
        step_vel = sample[:, step_vel_idx]
        zero_crossings = np.sum(np.abs(np.diff(np.sign(step_vel))) > 0)
        cadence = zero_crossings / (len(step_vel) / 30)
        feats.append(cadence)

        # 8. NEW: Arm-leg coordination (cross-correlation between arm and opposite leg)
        left_arm_idx = all_idx[feature_names.index('left_arm_swing')]
        right_knee_idx = all_idx[feature_names.index('right_knee_angle')]
        left_arm = sample[:, left_arm_idx]
        right_knee = sample[:, right_knee_idx]
        if len(left_arm) > 5:
            coord = np.corrcoef(left_arm, right_knee)[0, 1]
            if np.isnan(coord):
                coord = 0
        else:
            coord = 0
        feats.append(coord)

        # 9. NEW: Trunk stability (inverse of body_sway variability)
        trunk_stability = 1.0 / (np.std(body_sway) + 1e-6)
        feats.append(np.clip(trunk_stability, 0, 100))

        # 10. NEW: Step width consistency
        step_idx = all_idx[feature_names.index('step_width')]
        step = sample[:, step_idx]
        step_consistency = np.mean(step) / (np.std(step) + 1e-6)
        feats.append(np.clip(step_consistency, 0, 100))

        all_features.append(feats)

    return np.array(all_features)

print("\nExtracting v7 features...", flush=True)
X_train = extract_features_v7(train_data['X'], keep_idx, clinical_idx, clinical_features)
X_valid = extract_features_v7(valid_data['X'], keep_idx, clinical_idx, clinical_features)
X_test = extract_features_v7(test_data['X'], keep_idx, clinical_idx, clinical_features)

X = np.vstack([X_train, X_valid, X_test])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

n_base = len(keep_features) * 7
n_derived = X.shape[1] - n_base
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
print(f"  - Base features: {n_base}", flush=True)
print(f"  - Derived features: {n_derived}", flush=True)
print(f"    - CV features: 3", flush=True)
print(f"    - Asymmetry features: 6 pairs × 4 metrics = 24", flush=True)
print(f"    - Other: rhythm, jerk, freeze, cadence, coordination, stability, consistency", flush=True)

# Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'RF_500': RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'RF_500_d30': RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42, n_jobs=-1),
    'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'GB_200': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
}

print("\n" + "=" * 70, flush=True)
print("5-Fold StratifiedKFold Cross-Validation", flush=True)
print("=" * 70, flush=True)

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
baseline_prev = 0.739  # Previous best

for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    print(f"{name:<15} {acc:.1%} (vs orig: {acc-baseline_orig:+.1%}, vs prev: {acc-baseline_prev:+.1%})", flush=True)

best = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n*** BEST: {best[0]} with {best[1]['accuracy']:.1%} ***", flush=True)

# Save
results_path = data_dir.parent / "results" / "gait_v7_asymmetry_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'n_features': X.shape[1],
        'n_base': n_base,
        'n_derived': n_derived
    }, f)
print(f"Saved to: {results_path}", flush=True)
