#!/usr/bin/env python3
"""
Gait Multi-Model Training - Test various models on optimized 191 features
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 75, flush=True)
print("Gait Multi-Model Training - 191 Optimized Features", flush=True)
print("=" * 75, flush=True)

# Load and prepare data (same as train_gait_optimized.py)
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

remove_features = ['hip_height', 'left_hip_angle', 'right_hip_angle', 'shoulder_asymmetry']
keep_features = [f for f in clinical_features if f not in remove_features]
keep_idx = [clinical_idx[i] for i, f in enumerate(clinical_features) if f not in remove_features]

def extract_optimized_features(X_3d, keep_idx, all_idx, feature_names):
    n_samples = X_3d.shape[0]
    all_features = []

    for sample_idx in range(n_samples):
        sample = X_3d[sample_idx]
        feats = []

        # Standard aggregated features
        for idx in keep_idx:
            col = sample[:, idx]
            feats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75)
            ])

        # CV features
        for fname in ['trunk_angle_vel', 'body_sway_vel', 'stride_proxy_vel']:
            idx = all_idx[feature_names.index(fname)]
            col = sample[:, idx]
            cv = np.std(col) / (np.mean(np.abs(col)) + 1e-6)
            feats.append(cv)

        # Arm asymmetry
        left_arm = sample[:, all_idx[feature_names.index('left_arm_swing')]]
        right_arm = sample[:, all_idx[feature_names.index('right_arm_swing')]]
        feats.append(np.mean(np.abs(left_arm - right_arm)) / (np.mean(np.abs(left_arm) + np.abs(right_arm)) + 1e-6))

        # Knee asymmetry
        left_knee = sample[:, all_idx[feature_names.index('left_knee_angle')]]
        right_knee = sample[:, all_idx[feature_names.index('right_knee_angle')]]
        feats.append(np.mean(np.abs(left_knee - right_knee)) / (np.mean(np.abs(left_knee) + np.abs(right_knee)) + 1e-6))

        # Rhythm
        stride = sample[:, all_idx[feature_names.index('stride_proxy')]]
        autocorr = np.corrcoef(stride[:-5], stride[5:])[0, 1] if len(stride) > 10 else 0
        feats.append(0 if np.isnan(autocorr) else autocorr)

        # Jerk
        trunk_vel = sample[:, all_idx[feature_names.index('trunk_angle_vel')]]
        jerk = np.diff(trunk_vel)
        feats.append(np.sqrt(np.mean(jerk**2)) if len(jerk) > 0 else 0)

        # Freeze
        body_sway = sample[:, all_idx[feature_names.index('body_sway')]]
        threshold = np.percentile(np.abs(body_sway), 10)
        feats.append(np.mean(np.abs(body_sway) < threshold))

        # Cadence
        step_vel = sample[:, all_idx[feature_names.index('step_width_vel')]]
        feats.append(np.sum(np.abs(np.diff(np.sign(step_vel))) > 0) / (len(step_vel) / 30))

        all_features.append(feats)

    return np.array(all_features)

print("Extracting features...", flush=True)
X = np.vstack([
    extract_optimized_features(train_data['X'], keep_idx, clinical_idx, clinical_features),
    extract_optimized_features(valid_data['X'], keep_idx, clinical_idx, clinical_features),
    extract_optimized_features(test_data['X'], keep_idx, clinical_idx, clinical_features)
])
y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features", flush=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define all models to test
models = {
    # Tree-based
    'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'RF_500': RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
    'ET_500': ExtraTreesClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1),
    'GB_200': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    'GB_300': GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42),
    'XGB_200': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0),
    'XGB_300': xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.03, random_state=42, verbosity=0),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    'Bagging': BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1),

    # Neural Networks
    'MLP_100_50': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'MLP_200_100': MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=500, random_state=42),
    'MLP_deep': MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=500, random_state=42),

    # Linear
    'LogReg': LogisticRegression(max_iter=500, random_state=42),
    'Ridge': RidgeClassifier(random_state=42),

    # SVM
    'SVM_rbf': SVC(kernel='rbf', C=1.0, random_state=42),
    'SVM_poly': SVC(kernel='poly', degree=3, C=1.0, random_state=42),

    # Others
    'KNN_5': KNeighborsClassifier(n_neighbors=5),
    'KNN_10': KNeighborsClassifier(n_neighbors=10),
    'NaiveBayes': GaussianNB(),
}

print(f"\nTesting {len(models)} models...", flush=True)
print("=" * 75, flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    fold_scores = []
    fold_mae = []
    fold_within1 = []

    try:
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
        print(f"[{name:<15}] Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}, MAE={np.mean(fold_mae):.3f}, Within-1={np.mean(fold_within1):.3f}", flush=True)
    except Exception as e:
        print(f"[{name:<15}] ERROR: {e}", flush=True)

# Ensemble methods
print("\n" + "=" * 75, flush=True)
print("Ensemble Methods", flush=True)
print("=" * 75, flush=True)

# Voting
voting_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
    ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)),
    ('et', ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
]

fold_scores = []
for train_idx, val_idx in skf.split(X_scaled, y):
    voting = VotingClassifier(estimators=voting_models, voting='soft')
    voting.fit(X_scaled[train_idx], y[train_idx])
    fold_scores.append(accuracy_score(y[val_idx], voting.predict(X_scaled[val_idx])))
results['VotingSoft'] = {'accuracy': np.mean(fold_scores), 'accuracy_std': np.std(fold_scores)}
print(f"[VotingSoft      ] Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}", flush=True)

# Stacking with various meta-learners
for meta_name, meta in [('LR', LogisticRegression(max_iter=500)),
                         ('RF', RandomForestClassifier(n_estimators=100, max_depth=10)),
                         ('XGB', xgb.XGBClassifier(n_estimators=100, max_depth=5, verbosity=0))]:
    fold_scores = []
    for train_idx, val_idx in skf.split(X_scaled, y):
        stacking = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
                ('et', ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
                ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=5, random_state=42, verbosity=0)),
            ],
            final_estimator=meta,
            cv=3
        )
        stacking.fit(X_scaled[train_idx], y[train_idx])
        fold_scores.append(accuracy_score(y[val_idx], stacking.predict(X_scaled[val_idx])))
    results[f'Stacking_{meta_name}'] = {'accuracy': np.mean(fold_scores), 'accuracy_std': np.std(fold_scores)}
    print(f"[Stacking_{meta_name:<7}] Acc={np.mean(fold_scores):.3f}+/-{np.std(fold_scores):.3f}", flush=True)

# Summary
print("\n" + "=" * 75, flush=True)
print("FINAL RANKING", flush=True)
print("=" * 75, flush=True)

baseline = 0.739  # Previous best

sorted_results = sorted(results.items(), key=lambda x: -x[1]['accuracy'])
for rank, (name, res) in enumerate(sorted_results[:15], 1):
    acc = res['accuracy']
    diff = acc - baseline
    sign = '+' if diff >= 0 else ''
    print(f"{rank:2d}. {name:<20} {acc:.1%} ({sign}{diff:.1%})", flush=True)

best = sorted_results[0]
print(f"\n*** BEST: {best[0]} with {best[1]['accuracy']:.1%} ***", flush=True)

# Save
results_path = data_dir.parent / "results" / "gait_multi_model_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved to: {results_path}", flush=True)
