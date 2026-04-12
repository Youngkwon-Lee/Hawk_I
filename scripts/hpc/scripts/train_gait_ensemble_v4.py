#!/usr/bin/env python3
"""
PD4T Gait - Ensemble Training with v4 2D Aggregated Features
Target: 67.1% -> 70%+
"""

import pickle
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PD4T Gait - Ensemble Training with v4 Features")
print("Target: Improve from 67.1% to 70%+")
print("=" * 70)

# Load data
with open('../data/gait_train_v4.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('../data/gait_valid_v4.pkl', 'rb') as f:
    valid_data = pickle.load(f)
with open('../data/gait_test_v4.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Combine all data
X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])

# Extract subjects from ids
ids = np.hstack([train_data['ids'], valid_data['ids'], test_data['ids']])
subjects = np.array([str(id).split('_')[0] if '_' in str(id) else str(id) for id in ids])

print(f"\nTotal samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Unique subjects: {len(np.unique(subjects))}")
print(f"Class distribution: {np.bincount(y)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality (907 -> 100)
print("\nApplying PCA (907 -> 100 features)...")
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_)*100:.1f}%")

def evaluate_model(name, model, X, y, groups):
    """Evaluate with GroupKFold CV"""
    gkf = GroupKFold(n_splits=5)
    y_pred = cross_val_predict(model, X, y, cv=gkf, groups=groups)

    acc = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    within1 = np.mean(np.abs(y - y_pred) <= 1)
    r, _ = pearsonr(y, y_pred)

    return {
        'name': name,
        'accuracy': acc,
        'mae': mae,
        'within1': within1,
        'pearson_r': r
    }

# Define models
base_models = {
    'RF': RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGB': XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=500,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
}

print("\n" + "=" * 70)
print("Phase 1: Raw Features (907 dims)")
print("=" * 70)

results = []
for name, model in base_models.items():
    result = evaluate_model(name, model, X_scaled, y, subjects)
    results.append(result)
    print(f"{name:20s}: Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

print("\n" + "=" * 70)
print("Phase 2: PCA Features (100 dims)")
print("=" * 70)

for name, model in base_models.items():
    result = evaluate_model(f"{name}_PCA", model, X_pca, y, subjects)
    results.append(result)
    print(f"{name+'_PCA':20s}: Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

print("\n" + "=" * 70)
print("Phase 3: Voting Ensemble")
print("=" * 70)

# Soft Voting on raw features
voting_raw = VotingClassifier(
    estimators=[
        ('rf', base_models['RF']),
        ('xgb', base_models['XGB']),
        ('et', base_models['ExtraTrees'])
    ],
    voting='soft'
)

result = evaluate_model('Voting_Raw', voting_raw, X_scaled, y, subjects)
results.append(result)
print(f"{'Voting_Raw':20s}: Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

# Voting on PCA features
result = evaluate_model('Voting_PCA', voting_raw, X_pca, y, subjects)
results.append(result)
print(f"{'Voting_PCA':20s}: Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

print("\n" + "=" * 70)
print("FINAL RANKING")
print("=" * 70)

results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

print(f"{'Rank':<5}{'Model':<25}{'Accuracy':<12}{'MAE':<10}{'Within-1':<12}{'Pearson r':<10}")
print("-" * 70)
for i, r in enumerate(results_sorted, 1):
    print(f"{i:<5}{r['name']:<25}{r['accuracy']*100:.1f}%{'':<5}{r['mae']:.3f}{'':<4}{r['within1']*100:.1f}%{'':<5}{r['pearson_r']:.3f}")

best = results_sorted[0]
print(f"\nBEST: {best['name']} with {best['accuracy']*100:.1f}% accuracy")

# Improvement check
baseline = 0.671
improvement = best['accuracy'] - baseline
print(f"\n{'='*70}")
print(f"Baseline (TCN v2): 67.1%")
print(f"Best achieved: {best['accuracy']*100:.1f}%")
print(f"Improvement: {'+' if improvement > 0 else ''}{improvement*100:.1f}%")
print(f"{'='*70}")
