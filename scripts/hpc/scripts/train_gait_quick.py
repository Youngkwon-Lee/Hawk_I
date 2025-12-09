#!/usr/bin/env python3
"""
Quick Gait Ensemble Training - 3-fold CV with progress output
Target: Improve from 67.1% baseline
"""
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60, flush=True)
print("Gait Quick Ensemble Training - PD4T Dataset", flush=True)
print("=" * 60, flush=True)

# Load v4 data (pkl files)
data_dir = Path("C:/Users/YK/tulip/Hawkeye/hpc/data")

# Check for v4 pkl files
v4_train = data_dir / "gait_train_v4.pkl"
v4_valid = data_dir / "gait_valid_v4.pkl"
v4_test = data_dir / "gait_test_v4.pkl"

if v4_train.exists():
    print("Loading v4 features from pkl files...", flush=True)
    with open(v4_train, 'rb') as f:
        train_data = pickle.load(f)
    with open(v4_valid, 'rb') as f:
        valid_data = pickle.load(f)
    with open(v4_test, 'rb') as f:
        test_data = pickle.load(f)

    X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
    y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])
    ids = np.hstack([train_data['ids'], valid_data['ids'], test_data['ids']])
    subjects = np.array([id.split('_')[0] for id in ids])
else:
    # Fallback to v2 pkl files
    print("v4 not found, loading v2 pkl files...", flush=True)
    with open(data_dir / "gait_train_v2.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir / "gait_valid_v2.pkl", 'rb') as f:
        valid_data = pickle.load(f)
    with open(data_dir / "gait_test_v2.pkl", 'rb') as f:
        test_data = pickle.load(f)

    def extract_features_2d(X_3d):
        """Extract 2D aggregated features from 3D sequence"""
        n_samples = X_3d.shape[0]
        features_list = []

        for i in range(n_samples):
            sample = X_3d[i]  # (seq_len, n_features)
            feats = []

            for j in range(sample.shape[1]):
                col = sample[:, j]
                feats.extend([
                    np.mean(col), np.std(col), np.min(col), np.max(col),
                    np.median(col), np.percentile(col, 25), np.percentile(col, 75)
                ])
            features_list.append(feats)

        return np.array(features_list)

    print("Extracting features from train...", flush=True)
    X_train = extract_features_2d(train_data['X'])
    print("Extracting features from valid...", flush=True)
    X_valid = extract_features_2d(valid_data['X'])
    print("Extracting features from test...", flush=True)
    X_test = extract_features_2d(test_data['X'])

    X = np.vstack([X_train, X_valid, X_test])
    y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])
    ids = np.hstack([train_data['ids'], valid_data['ids'], test_data['ids']])
    subjects = np.array([id.split('_')[0] for id in ids])

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
print(f"Labels: {np.unique(y, return_counts=True)}", flush=True)
print(f"Subjects: {len(np.unique(subjects))}", flush=True)

# Preprocessing
print("\nPreprocessing...", flush=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
n_components = min(100, X.shape[1], X.shape[0] - 1)
print(f"PCA: {X.shape[1]} -> {n_components} features", flush=True)
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}", flush=True)

# Models to test (reduced set for speed)
models = {
    'RF_200': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'GradBoost': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
}

# 3-fold GroupKFold CV
print("\n" + "=" * 60, flush=True)
print("3-Fold GroupKFold Cross-Validation", flush=True)
print("=" * 60, flush=True)

gkf = GroupKFold(n_splits=3)
results = {}

for name, model in models.items():
    print(f"\n[{name}] Training...", flush=True)
    fold_scores = []
    fold_mae = []
    fold_within1 = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_pca, y, subjects)):
        X_train, X_val = X_pca[train_idx], X_pca[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        within1 = np.mean(np.abs(y_val - y_pred) <= 1)

        fold_scores.append(acc)
        fold_mae.append(mae)
        fold_within1.append(within1)

        print(f"  Fold {fold+1}: Acc={acc:.3f}, MAE={mae:.3f}, Within-1={within1:.3f}", flush=True)

    mean_acc = np.mean(fold_scores)
    std_acc = np.std(fold_scores)
    mean_mae = np.mean(fold_mae)
    mean_within1 = np.mean(fold_within1)

    results[name] = {
        'accuracy': mean_acc,
        'accuracy_std': std_acc,
        'mae': mean_mae,
        'within1': mean_within1
    }

    print(f"  >> Mean: Acc={mean_acc:.3f}+/-{std_acc:.3f}, MAE={mean_mae:.3f}, Within-1={mean_within1:.3f}", flush=True)

# Summary
print("\n" + "=" * 60, flush=True)
print("RESULTS SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"{'Model':<15} {'Accuracy':<12} {'MAE':<8} {'Within-1':<10}", flush=True)
print("-" * 45, flush=True)

baseline = 0.671  # 67.1% baseline
for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    improvement = acc - baseline
    sign = "+" if improvement > 0 else ""
    print(f"{name:<15} {acc:.1%} ({sign}{improvement:.1%})  {res['mae']:.3f}    {res['within1']:.1%}", flush=True)

best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest: {best_model[0]} with {best_model[1]['accuracy']:.1%} accuracy", flush=True)
print(f"Baseline: {baseline:.1%}, Improvement: {best_model[1]['accuracy'] - baseline:+.1%}", flush=True)

# Save results
results_path = data_dir.parent / "results" / "gait_v4_quick_results.pkl"
results_path.parent.mkdir(exist_ok=True)
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to: {results_path}", flush=True)
