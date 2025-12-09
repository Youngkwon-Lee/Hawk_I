#!/usr/bin/env python3
"""
Gait Training - Clinical Features Only (Best performing configuration)
30 clinical features: step_width, hip_height, trunk_angle, knee_angles, etc.
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60, flush=True)
print("Gait Training - CLINICAL FEATURES ONLY", flush=True)
print("=" * 60, flush=True)

# Load data
data_dir = Path("C:/Users/YK/tulip/Hawkeye/hpc/data")

with open(data_dir / "gait_train_v2.pkl", 'rb') as f:
    train_data = pickle.load(f)
with open(data_dir / "gait_valid_v2.pkl", 'rb') as f:
    valid_data = pickle.load(f)
with open(data_dir / "gait_test_v2.pkl", 'rb') as f:
    test_data = pickle.load(f)

features = train_data['features']

# Clinical features only (99-128)
clinical_idx = list(range(99, 129))
clinical_features = [features[i] for i in clinical_idx]

print(f"\nClinical features ({len(clinical_features)}):", flush=True)
for i, f in enumerate(clinical_features):
    print(f"  {i+1}. {f}", flush=True)

# Extract aggregated features
def extract_clinical(X_3d, idx):
    X_f = X_3d[:, :, idx]
    result = []
    for sample in X_f:
        feats = []
        for j in range(sample.shape[1]):
            col = sample[:, j]
            feats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75)
            ])
        result.append(feats)
    return np.array(result)

print("\nExtracting features...", flush=True)
X = np.vstack([
    extract_clinical(train_data['X'], clinical_idx),
    extract_clinical(valid_data['X'], clinical_idx),
    extract_clinical(test_data['X'], clinical_idx)
])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
print(f"  (30 clinical × 7 stats = 210)", flush=True)
print(f"Labels: {np.unique(y, return_counts=True)}", flush=True)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models with various configurations
models = {
    'RF_200': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'RF_500': RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'RF_500_d30': RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42, n_jobs=-1),
    'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'ET_500': ExtraTreesClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'GB_200': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'GB_300': GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42),
    'Ada_200': AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
}

print("\n" + "=" * 60, flush=True)
print("5-Fold StratifiedKFold Cross-Validation", flush=True)
print("=" * 60, flush=True)

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

# Ensemble methods
print("\n" + "=" * 60, flush=True)
print("Ensemble Methods", flush=True)
print("=" * 60, flush=True)

# Voting
print("\n[VotingEnsemble]", flush=True)
voting_models = [
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)),
]

fold_scores = []
for train_idx, val_idx in skf.split(X_scaled, y):
    voting = VotingClassifier(estimators=voting_models, voting='soft')
    voting.fit(X_scaled[train_idx], y[train_idx])
    fold_scores.append(accuracy_score(y[val_idx], voting.predict(X_scaled[val_idx])))

results['VotingEnsemble'] = {'accuracy': np.mean(fold_scores), 'accuracy_std': np.std(fold_scores)}
print(f"  Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}", flush=True)

# Stacking with LR
print("\n[StackingLR]", flush=True)
stacking_base = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
    ('et', ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)),
]

fold_scores = []
for train_idx, val_idx in skf.split(X_scaled, y):
    stacking = StackingClassifier(
        estimators=stacking_base,
        final_estimator=LogisticRegression(max_iter=500, random_state=42),
        cv=3
    )
    stacking.fit(X_scaled[train_idx], y[train_idx])
    fold_scores.append(accuracy_score(y[val_idx], stacking.predict(X_scaled[val_idx])))

results['StackingLR'] = {'accuracy': np.mean(fold_scores), 'accuracy_std': np.std(fold_scores)}
print(f"  Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}", flush=True)

# Stacking with RF
print("\n[StackingRF]", flush=True)
fold_scores = []
for train_idx, val_idx in skf.split(X_scaled, y):
    stacking = StackingClassifier(
        estimators=stacking_base,
        final_estimator=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        cv=3
    )
    stacking.fit(X_scaled[train_idx], y[train_idx])
    fold_scores.append(accuracy_score(y[val_idx], stacking.predict(X_scaled[val_idx])))

results['StackingRF'] = {'accuracy': np.mean(fold_scores), 'accuracy_std': np.std(fold_scores)}
print(f"  Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}", flush=True)

# Summary
print("\n" + "=" * 60, flush=True)
print("FINAL RESULTS SUMMARY", flush=True)
print("=" * 60, flush=True)

baseline = 0.671
prev_best = 0.700

for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    acc = res['accuracy']
    print(f"{name:<20} {acc:.1%} (vs orig: {acc-baseline:+.1%}, vs v4: {acc-prev_best:+.1%})", flush=True)

best = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n*** BEST: {best[0]} with {best[1]['accuracy']:.1%} ***", flush=True)
print(f"Original baseline: 67.1%", flush=True)
print(f"Previous best (v4 all features): 70.0%", flush=True)
print(f"Improvement: {best[1]['accuracy'] - baseline:+.1%} vs original", flush=True)

# Save results
results_path = data_dir.parent / "results" / "gait_clinical_only_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'features': clinical_features,
        'best_model': best[0],
        'best_accuracy': best[1]['accuracy']
    }, f)
print(f"\nSaved to: {results_path}", flush=True)
