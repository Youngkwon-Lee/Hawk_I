#!/usr/bin/env python3
"""
Finger Tapping Optimized Training
- Remove low importance features (freeze_count, hesitation_count)
- Add derived features (ratios, interactions)
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70, flush=True)
print("Finger Tapping Optimized Training", flush=True)
print("=" * 70, flush=True)

# Load data
data_dir = Path("C:/Users/YK/tulip/Hawkeye/hpc/data")

with open(data_dir / "finger_train_v4.pkl", 'rb') as f:
    train_data = pickle.load(f)
with open(data_dir / "finger_valid_v4.pkl", 'rb') as f:
    valid_data = pickle.load(f)
with open(data_dir / "finger_test_v4.pkl", 'rb') as f:
    test_data = pickle.load(f)

X_orig = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])
ids = np.hstack([train_data['ids'], valid_data['ids'], test_data['ids']])
subjects = np.array([id.split('_')[0] for id in ids])
features = train_data['features']

print(f"Original features: {len(features)}", flush=True)
print(f"Samples: {X_orig.shape[0]}", flush=True)

# Feature indices
feature_idx = {f: i for i, f in enumerate(features)}

# Remove low importance features
remove_features = ['freeze_count', 'hesitation_count']
keep_mask = [i for i, f in enumerate(features) if f not in remove_features]
keep_features = [features[i] for i in keep_mask]

X_filtered = X_orig[:, keep_mask]
print(f"\nAfter removing {remove_features}: {X_filtered.shape[1]} features", flush=True)

# Add derived features
def add_derived_features(X, feature_names, feature_idx_map):
    """Add interaction and ratio features"""
    derived = []

    for i in range(X.shape[0]):
        row = X[i]
        feats = list(row)  # Start with original features

        # Get indices in filtered feature set
        def get_val(name):
            if name in feature_idx_map:
                orig_idx = feature_idx_map[name]
                # Find new index after filtering
                new_idx = keep_mask.index(orig_idx) if orig_idx in keep_mask else None
                if new_idx is not None:
                    return row[new_idx]
            return 0

        # 1. Amplitude decay ratio (first5 / last5)
        first5 = get_val('first5_amp')
        last5 = get_val('last5_amp')
        decay_ratio = first5 / (last5 + 1e-6) if last5 > 0 else 0
        feats.append(np.clip(decay_ratio, 0, 10))

        # 2. Speed asymmetry (opening vs closing)
        open_speed = get_val('opening_speed_mean')
        close_speed = get_val('closing_speed_mean')
        speed_asym = np.abs(open_speed - close_speed) / (open_speed + close_speed + 1e-6)
        feats.append(speed_asym)

        # 3. Rhythm stability (1 / cycle_cv)
        cycle_cv = get_val('cycle_cv')
        rhythm_stability = 1.0 / (cycle_cv + 1e-6) if cycle_cv > 0 else 0
        feats.append(np.clip(rhythm_stability, 0, 100))

        # 4. Amplitude consistency (1 / amp_cv)
        amp_cv = get_val('amp_cv')
        amp_consistency = 1.0 / (amp_cv + 1e-6) if amp_cv > 0 else 0
        feats.append(np.clip(amp_consistency, 0, 100))

        # 5. Fatigue index (slope2 - slope1, negative = fatigue)
        slope1 = get_val('slope1')
        slope2 = get_val('slope2')
        fatigue_index = slope2 - slope1
        feats.append(fatigue_index)

        # 6. Bradykinesia score (combines amplitude and speed)
        amp_mean = get_val('amp_mean')
        max_speed = get_val('max_speed')
        bradykinesia = amp_mean * max_speed  # Higher = better
        feats.append(bradykinesia)

        # 7. Hypokinesia index (amplitude relative to speed)
        hypokinesia = amp_mean / (max_speed + 1e-6)
        feats.append(np.clip(hypokinesia, 0, 10))

        # 8. Movement efficiency (taps per unit duration normalized)
        taps_per_sec = get_val('taps_per_sec')
        amp_ratio = get_val('amp_ratio')
        efficiency = taps_per_sec * amp_ratio
        feats.append(efficiency)

        # 9. Sequence degradation (exp_decay_rate * duration)
        exp_decay = get_val('exp_decay_rate')
        duration = get_val('duration')
        seq_degrade = exp_decay * duration
        feats.append(seq_degrade)

        # 10. Overall motor score (composite)
        rhythm_reg = get_val('rhythm_regularity')
        motor_score = (amp_mean + max_speed + rhythm_reg) / 3
        feats.append(motor_score)

        derived.append(feats)

    return np.array(derived)

print("\nAdding derived features...", flush=True)
X = add_derived_features(X_filtered, keep_features, feature_idx)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

n_orig = len(keep_features)
n_derived = X.shape[1] - n_orig
print(f"Final features: {X.shape[1]} ({n_orig} base + {n_derived} derived)", flush=True)

# Derived feature names
derived_names = ['decay_ratio', 'speed_asymmetry', 'rhythm_stability', 'amp_consistency',
                 'fatigue_index', 'bradykinesia', 'hypokinesia', 'efficiency',
                 'seq_degradation', 'motor_score']
print(f"Derived features: {derived_names}", flush=True)

# Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'RF_500': RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'ET_500': ExtraTreesClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'GB_200': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
}

print("\n" + "=" * 70, flush=True)
print("5-Fold GroupKFold Cross-Validation", flush=True)
print("=" * 70, flush=True)

gkf = GroupKFold(n_splits=5)
results = {}

for name, model in models.items():
    print(f"\n[{name}]", flush=True)
    fold_scores = []
    fold_mae = []
    fold_within1 = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, subjects)):
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

baseline = 0.591
prev_best = 0.604

for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    print(f"{name:<15} {acc:.1%} (vs orig: {acc-baseline:+.1%}, vs prev: {acc-prev_best:+.1%})", flush=True)

best = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n*** BEST: {best[0]} with {best[1]['accuracy']:.1%} ***", flush=True)

# Save
results_path = data_dir.parent / "results" / "finger_optimized_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'kept_features': keep_features,
        'derived_features': derived_names,
        'removed_features': remove_features
    }, f)
print(f"Saved to: {results_path}", flush=True)
