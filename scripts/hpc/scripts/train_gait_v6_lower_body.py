#!/usr/bin/env python3
"""
Gait v6 Training - Lower Body + Clinical Features Only
Focus on: hips, knees, ankles, heels, feet + all clinical features
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
print("Gait v6 Training - LOWER BODY + CLINICAL ONLY", flush=True)
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

# Select lower body + clinical features only
# Lower body (hips 69-74, knees 75-80, ankles 81-86, heels 87-92, feet 93-98)
# Clinical position (99-113), Clinical velocity (114-128)
lower_body_indices = list(range(69, 99))  # hips to feet
clinical_indices = list(range(99, 129))    # all clinical features

keep_indices = lower_body_indices + clinical_indices
keep_features = [features[i] for i in keep_indices]

print(f"\nSelected features: {len(keep_indices)}", flush=True)
print(f"  - Lower body landmarks: {len(lower_body_indices)} (hips, knees, ankles, heels, feet)", flush=True)
print(f"  - Clinical features: {len(clinical_indices)} (position + velocity)", flush=True)

print("\nLower body features:", flush=True)
for i in lower_body_indices[:12]:  # Show first few
    print(f"  {i}: {features[i]}", flush=True)
print("  ...")

print("\nClinical features:", flush=True)
for i in clinical_indices[:10]:
    print(f"  {i}: {features[i]}", flush=True)
print("  ...")

# Extract features
def extract_features_selected(X_3d, keep_idx):
    X_filtered = X_3d[:, :, keep_idx]
    n_samples = X_filtered.shape[0]
    features_list = []

    for i in range(n_samples):
        sample = X_filtered[i]
        feats = []
        for j in range(sample.shape[1]):
            col = sample[:, j]
            feats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75)
            ])
        features_list.append(feats)
    return np.array(features_list)

print("\nExtracting features...", flush=True)
X_train = extract_features_selected(train_data['X'], keep_indices)
X_valid = extract_features_selected(valid_data['X'], keep_indices)
X_test = extract_features_selected(test_data['X'], keep_indices)

X = np.vstack([X_train, X_valid, X_test])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
print(f"Labels: {np.unique(y, return_counts=True)}", flush=True)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} features (99% variance)", flush=True)

# Models
models = {
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'RF_500': RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'MLP_deeper': MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=500, random_state=42),
}

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

# Stacking
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
print(f"  Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}", flush=True)

# Summary
print("\n" + "=" * 60, flush=True)
print("RESULTS SUMMARY (LOWER BODY + CLINICAL)", flush=True)
print("=" * 60, flush=True)

baseline_v4 = 0.700
baseline_orig = 0.671

for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    print(f"{name:<20} {acc:.1%} (vs v4: {acc-baseline_v4:+.1%}, vs orig: {acc-baseline_orig:+.1%})", flush=True)

best = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest: {best[0]} with {best[1]['accuracy']:.1%}", flush=True)

# Save
results_path = data_dir.parent / "results" / "gait_v6_lower_body_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump({'results': results, 'features_used': keep_features}, f)
print(f"Saved to: {results_path}", flush=True)
