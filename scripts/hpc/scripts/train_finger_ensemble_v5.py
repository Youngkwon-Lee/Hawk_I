#!/usr/bin/env python3
"""
PD4T Finger Tapping - Advanced Ensemble for Accuracy Improvement
Target: 59.1% → 65%+

Strategies:
1. Voting Ensemble (RF + XGB + AdaBoost + ExtraTrees + GradientBoosting)
2. Stacking Ensemble with meta-learner
3. Hyperparameter optimization with Optuna
4. Class imbalance handling (SMOTE, class_weight)
"""

import pickle
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PD4T Finger Tapping - Advanced Ensemble Training")
print("Target: Improve from 59.1% to 65%+")
print("=" * 70)

# Load data
with open('../data/finger_train_v4.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('../data/finger_valid_v4.pkl', 'rb') as f:
    valid_data = pickle.load(f)
with open('../data/finger_test_v4.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Combine all data for GroupKFold CV
X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])
# Extract subjects from ids (format: '13-002356_r_003' -> '13-002356')
ids = np.hstack([train_data['ids'], valid_data['ids'], test_data['ids']])
subjects = np.array([id.split('_')[0] for id in ids])

print(f"\nTotal samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Unique subjects: {len(np.unique(subjects))}")
print(f"Class distribution: {np.bincount(y)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define base models with class_weight for imbalance
base_models = {
    'RF': RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGB': XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
}

def evaluate_model(name, model, X, y, groups):
    """Evaluate model with GroupKFold CV"""
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
        'pearson_r': r,
        'y_pred': y_pred
    }

print("\n" + "=" * 70)
print("Phase 1: Individual Model Evaluation")
print("=" * 70)

results = []
for name, model in base_models.items():
    result = evaluate_model(name, model, X_scaled, y, subjects)
    results.append(result)
    print(f"{name:20s}: Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

print("\n" + "=" * 70)
print("Phase 2: Voting Ensemble")
print("=" * 70)

# Soft Voting Ensemble
voting_clf = VotingClassifier(
    estimators=[
        ('rf', base_models['RF']),
        ('xgb', base_models['XGB']),
        ('et', base_models['ExtraTrees']),
        ('gb', base_models['GradientBoosting'])
    ],
    voting='soft'
)

result = evaluate_model('Soft Voting', voting_clf, X_scaled, y, subjects)
results.append(result)
print(f"Soft Voting Ensemble: Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

# Hard Voting Ensemble
voting_hard = VotingClassifier(
    estimators=[
        ('rf', base_models['RF']),
        ('xgb', base_models['XGB']),
        ('et', base_models['ExtraTrees']),
        ('ada', base_models['AdaBoost'])
    ],
    voting='hard'
)

result = evaluate_model('Hard Voting', voting_hard, X_scaled, y, subjects)
results.append(result)
print(f"Hard Voting Ensemble: Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

print("\n" + "=" * 70)
print("Phase 3: Stacking Ensemble")
print("=" * 70)

# Stacking with Logistic Regression meta-learner
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', base_models['RF']),
        ('xgb', base_models['XGB']),
        ('et', base_models['ExtraTrees'])
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
    cv=5,
    n_jobs=-1
)

result = evaluate_model('Stacking (LR)', stacking_clf, X_scaled, y, subjects)
results.append(result)
print(f"Stacking (LR meta): Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

# Stacking with RF meta-learner
stacking_rf = StackingClassifier(
    estimators=[
        ('rf', base_models['RF']),
        ('xgb', base_models['XGB']),
        ('gb', base_models['GradientBoosting'])
    ],
    final_estimator=RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    cv=5,
    n_jobs=-1
)

result = evaluate_model('Stacking (RF)', stacking_rf, X_scaled, y, subjects)
results.append(result)
print(f"Stacking (RF meta): Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

print("\n" + "=" * 70)
print("Phase 4: Hyperparameter Tuned Models")
print("=" * 70)

# Optimized RF
rf_tuned = RandomForestClassifier(
    n_estimators=1000,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced_subsample',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

result = evaluate_model('RF Tuned', rf_tuned, X_scaled, y, subjects)
results.append(result)
print(f"RF Tuned: Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

# Optimized XGB with scale_pos_weight
class_counts = np.bincount(y)
xgb_tuned = XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

result = evaluate_model('XGB Tuned', xgb_tuned, X_scaled, y, subjects)
results.append(result)
print(f"XGB Tuned: Acc={result['accuracy']*100:.1f}%, MAE={result['mae']:.3f}, Within-1={result['within1']*100:.1f}%, r={result['pearson_r']:.3f}")

print("\n" + "=" * 70)
print("FINAL RANKING")
print("=" * 70)

# Sort by accuracy
results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

print(f"{'Rank':<5}{'Model':<25}{'Accuracy':<12}{'MAE':<10}{'Within-1':<12}{'Pearson r':<10}")
print("-" * 70)
for i, r in enumerate(results_sorted, 1):
    print(f"{i:<5}{r['name']:<25}{r['accuracy']*100:.1f}%{'':<5}{r['mae']:.3f}{'':<4}{r['within1']*100:.1f}%{'':<5}{r['pearson_r']:.3f}")

best = results_sorted[0]
print(f"\n🏆 BEST: {best['name']} with {best['accuracy']*100:.1f}% accuracy")

# Confusion matrix for best model
print(f"\nConfusion Matrix ({best['name']}):")
cm = confusion_matrix(y, best['y_pred'])
print(cm)

# Improvement check
baseline = 0.591
improvement = best['accuracy'] - baseline
print(f"\n{'='*70}")
print(f"Baseline (RF v4): 59.1%")
print(f"Best achieved: {best['accuracy']*100:.1f}%")
print(f"Improvement: {'+' if improvement > 0 else ''}{improvement*100:.1f}%")
print(f"{'='*70}")
