#!/usr/bin/env python3
"""
Gait v5 Training - Without Face Features (nose, eyes, ears, mouth)
Keep only: shoulders, elbows, wrists, hips, knees, ankles, heels, feet + clinical features
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60, flush=True)
print("Gait v5 Training - NO FACE FEATURES", flush=True)
print("=" * 60, flush=True)

# Load v2 3D data
data_dir = Path("C:/Users/YK/tulip/Hawkeye/hpc/data")

with open(data_dir / "gait_train_v2.pkl", 'rb') as f:
    train_data = pickle.load(f)
with open(data_dir / "gait_valid_v2.pkl", 'rb') as f:
    valid_data = pickle.load(f)
with open(data_dir / "gait_test_v2.pkl", 'rb') as f:
    test_data = pickle.load(f)

features = train_data['features']
print(f"\nOriginal features: {len(features)}", flush=True)

# Define face-related feature indices to EXCLUDE (0-32)
# nose(0-2), eyes(3-20), ears(21-26), mouth(27-32)
face_indices = list(range(0, 33))  # indices 0-32

# Keep indices 33-128 (shoulders onwards + clinical features)
keep_indices = [i for i in range(len(features)) if i not in face_indices]
keep_features = [features[i] for i in keep_indices]

print(f"Removed face features: {len(face_indices)} (nose, eyes, ears, mouth)", flush=True)
print(f"Keeping features: {len(keep_indices)}", flush=True)
print(f"\nKept feature groups:", flush=True)
print(f"  - Upper body: shoulders, elbows, wrists, hands (33-68)", flush=True)
print(f"  - Lower body: hips, knees, ankles, heels, feet (69-98)", flush=True)
print(f"  - Clinical position: step_width, hip_height, etc. (99-113)", flush=True)
print(f"  - Clinical velocity: derivatives (114-128)", flush=True)

# Extract only non-face features from 3D data
def extract_features_no_face(X_3d, keep_idx):
    """Extract 2D aggregated features from 3D, excluding face"""
    # X_3d shape: (n_samples, seq_len, n_features)
    X_filtered = X_3d[:, :, keep_idx]  # Filter features

    n_samples = X_filtered.shape[0]
    features_list = []

    for i in range(n_samples):
        sample = X_filtered[i]  # (seq_len, n_features)
        feats = []

        for j in range(sample.shape[1]):
            col = sample[:, j]
            feats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75)
            ])
        features_list.append(feats)

    return np.array(features_list)

print("\nExtracting features (excluding face)...", flush=True)
X_train = extract_features_no_face(train_data['X'], keep_indices)
X_valid = extract_features_no_face(valid_data['X'], keep_indices)
X_test = extract_features_no_face(test_data['X'], keep_indices)

X = np.vstack([X_train, X_valid, X_test])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
print(f"  (96 body features × 7 stats = {96*7} expected)", flush=True)
print(f"Labels: {np.unique(y, return_counts=True)}", flush=True)

# Preprocessing
print("\nPreprocessing...", flush=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA - keep 99% variance
pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} features (99% variance)", flush=True)

# Models
models = {
    'RF_200': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'RF_500': RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'GB_200': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
}

# 5-fold StratifiedKFold CV
print("\n" + "=" * 60, flush=True)
print("5-Fold StratifiedKFold Cross-Validation", flush=True)
print("=" * 60, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n[{name}] Training...", flush=True)
    fold_scores = []
    fold_mae = []
    fold_within1 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pca, y)):
        X_train, X_val = X_pca[train_idx], X_pca[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)

        fold_scores.append(accuracy_score(y_val, y_pred))
        fold_mae.append(mean_absolute_error(y_val, y_pred))
        fold_within1.append(np.mean(np.abs(y_val - y_pred) <= 1))

    results[name] = {
        'accuracy': np.mean(fold_scores),
        'accuracy_std': np.std(fold_scores),
        'mae': np.mean(fold_mae),
        'within1': np.mean(fold_within1)
    }
    print(f"  Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}, MAE={np.mean(fold_mae):.3f}, Within-1={np.mean(fold_within1):.3f}", flush=True)

# Stacking Ensemble
print("\n[StackingEnsemble] Training...", flush=True)
stacking_base = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
]

fold_scores = []
fold_mae = []
fold_within1 = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_pca, y)):
    X_train, X_val = X_pca[train_idx], X_pca[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    stacking = StackingClassifier(
        estimators=stacking_base,
        final_estimator=LogisticRegression(max_iter=500, random_state=42),
        cv=3
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_val)

    fold_scores.append(accuracy_score(y_val, y_pred))
    fold_mae.append(mean_absolute_error(y_val, y_pred))
    fold_within1.append(np.mean(np.abs(y_val - y_pred) <= 1))

results['StackingEnsemble'] = {
    'accuracy': np.mean(fold_scores),
    'accuracy_std': np.std(fold_scores),
    'mae': np.mean(fold_mae),
    'within1': np.mean(fold_within1)
}
print(f"  Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}, MAE={np.mean(fold_mae):.3f}, Within-1={np.mean(fold_within1):.3f}", flush=True)

# Summary
print("\n" + "=" * 60, flush=True)
print("RESULTS SUMMARY (NO FACE FEATURES)", flush=True)
print("=" * 60, flush=True)
print(f"{'Model':<20} {'Accuracy':<15} {'MAE':<10} {'Within-1':<10}", flush=True)
print("-" * 55, flush=True)

baseline_v4 = 0.700  # Previous best with face features
baseline_orig = 0.671  # Original baseline

for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    vs_v4 = acc - baseline_v4
    vs_orig = acc - baseline_orig
    print(f"{name:<20} {acc:.1%} (vs v4: {vs_v4:+.1%}, vs orig: {vs_orig:+.1%})", flush=True)

best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest: {best_model[0]} with {best_model[1]['accuracy']:.1%} accuracy", flush=True)
print(f"Original baseline (v4 with face): 70.0%", flush=True)
print(f"Improvement vs v4: {best_model[1]['accuracy'] - baseline_v4:+.1%}", flush=True)

# Save results
results_path = data_dir.parent / "results" / "gait_v5_no_face_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'features_used': keep_features,
        'n_features_original': len(features),
        'n_features_used': len(keep_features),
        'excluded': ['nose', 'eyes', 'ears', 'mouth']
    }, f)
print(f"\nResults saved to: {results_path}", flush=True)
