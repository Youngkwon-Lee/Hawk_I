#!/usr/bin/env python3
"""
Finger Tapping v4 Full Ensemble Training - StratifiedKFold CV
Target: Improve from 59.1% baseline using Stacking approach (like Gait success)
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60, flush=True)
print("Finger Tapping v4 Full Ensemble Training - PD4T Dataset", flush=True)
print("=" * 60, flush=True)

# Load v4 data
data_dir = Path("C:/Users/YK/tulip/Hawkeye/hpc/data")

with open(data_dir / "finger_train_v4.pkl", 'rb') as f:
    train_data = pickle.load(f)
with open(data_dir / "finger_valid_v4.pkl", 'rb') as f:
    valid_data = pickle.load(f)
with open(data_dir / "finger_test_v4.pkl", 'rb') as f:
    test_data = pickle.load(f)

X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])
ids = np.hstack([train_data['ids'], valid_data['ids'], test_data['ids']])

# Extract subjects - format: '13-002356_r_003' -> '13-002356'
subjects = np.array([id.split('_')[0] for id in ids])

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
print(f"Labels: {np.unique(y, return_counts=True)}", flush=True)
print(f"Unique subjects: {len(np.unique(subjects))}", flush=True)

# Preprocessing
print("\nPreprocessing...", flush=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA - keep 99% variance
pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X.shape[1]} -> {X_pca.shape[1]} features (99% variance)", flush=True)

# Base models
base_models = {
    'RF_100': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'RF_200': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'RF_500': RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'ET_200': ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'GB_100': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'GB_200': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'Ada_100': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'MLP_100_50': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'MLP_200_100': MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=500, random_state=42),
}

# Use GroupKFold since we have multiple samples per subject
n_unique_subjects = len(np.unique(subjects))
n_splits = min(5, n_unique_subjects)
print(f"\nUsing {n_splits}-Fold GroupKFold CV (subject-level splits)", flush=True)

gkf = GroupKFold(n_splits=n_splits)
results = {}

# Test each base model
print("\n" + "=" * 60, flush=True)
print(f"{n_splits}-Fold GroupKFold Cross-Validation", flush=True)
print("=" * 60, flush=True)

for name, model in base_models.items():
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

    print(f"  Acc={mean_acc:.3f}+/-{std_acc:.3f}, MAE={mean_mae:.3f}, Within-1={mean_within1:.3f}", flush=True)

# Ensemble methods
print("\n" + "=" * 60, flush=True)
print("Ensemble Methods", flush=True)
print("=" * 60, flush=True)

# Voting Ensemble
print("\n[VotingEnsemble] Training...", flush=True)
voting_models = [
    ('rf300', RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
    ('rf500', RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1)),
    ('et300', ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
    ('gb200', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)),
]

fold_scores = []
fold_mae = []
fold_within1 = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_pca, y, subjects)):
    X_train, X_val = X_pca[train_idx], X_pca[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    voting = VotingClassifier(estimators=voting_models, voting='soft')
    voting.fit(X_train, y_train)
    y_pred = voting.predict(X_val)

    fold_scores.append(accuracy_score(y_val, y_pred))
    fold_mae.append(mean_absolute_error(y_val, y_pred))
    fold_within1.append(np.mean(np.abs(y_val - y_pred) <= 1))

results['VotingEnsemble'] = {
    'accuracy': np.mean(fold_scores),
    'accuracy_std': np.std(fold_scores),
    'mae': np.mean(fold_mae),
    'within1': np.mean(fold_within1)
}
print(f"  Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}, MAE={np.mean(fold_mae):.3f}, Within-1={np.mean(fold_within1):.3f}", flush=True)

# Stacking Ensemble (worked well for Gait)
print("\n[StackingEnsemble] Training...", flush=True)
stacking_base = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
]

fold_scores = []
fold_mae = []
fold_within1 = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_pca, y, subjects)):
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

# Stacking with MLP meta-learner
print("\n[StackingMLP] Training...", flush=True)
fold_scores = []
fold_mae = []
fold_within1 = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_pca, y, subjects)):
    X_train, X_val = X_pca[train_idx], X_pca[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    stacking = StackingClassifier(
        estimators=stacking_base,
        final_estimator=MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
        cv=3
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_val)

    fold_scores.append(accuracy_score(y_val, y_pred))
    fold_mae.append(mean_absolute_error(y_val, y_pred))
    fold_within1.append(np.mean(np.abs(y_val - y_pred) <= 1))

results['StackingMLP'] = {
    'accuracy': np.mean(fold_scores),
    'accuracy_std': np.std(fold_scores),
    'mae': np.mean(fold_mae),
    'within1': np.mean(fold_within1)
}
print(f"  Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}, MAE={np.mean(fold_mae):.3f}, Within-1={np.mean(fold_within1):.3f}", flush=True)

# Summary
print("\n" + "=" * 60, flush=True)
print("FINAL RESULTS SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"{'Model':<20} {'Accuracy':<15} {'MAE':<10} {'Within-1':<10}", flush=True)
print("-" * 55, flush=True)

baseline = 0.591  # 59.1% baseline
for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    improvement = acc - baseline
    sign = "+" if improvement > 0 else ""
    print(f"{name:<20} {acc:.1%} ({sign}{improvement:.1%})  {res['mae']:.3f}      {res['within1']:.1%}", flush=True)

best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest: {best_model[0]} with {best_model[1]['accuracy']:.1%} accuracy", flush=True)
print(f"Baseline: {baseline:.1%}, Improvement: {best_model[1]['accuracy'] - baseline:+.1%}", flush=True)

# Save results
results_path = data_dir.parent / "results" / "finger_v4_full_results.pkl"
results_path.parent.mkdir(exist_ok=True)
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved to: {results_path}", flush=True)
