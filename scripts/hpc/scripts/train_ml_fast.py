#!/usr/bin/env python3
"""
Fast ML for PD4T - Quick experiment with core models only
Goal: Beat current best (Gait 70.4%, Finger 58.3%)
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

data_dir = Path(__file__).parent.parent / 'data'

# ===================== FEATURE EXTRACTION =====================

def extract_features(X_3d, task='gait'):
    """Extract aggregated features from 3D time series"""
    n_samples = X_3d.shape[0]
    all_features = []

    for i in range(n_samples):
        sample = X_3d[i]  # (T, F)
        feats = []

        for j in range(sample.shape[1]):
            col = sample[:, j]
            # Basic stats
            feats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75)
            ])
            # Temporal features
            diff = np.diff(col)
            feats.extend([
                np.mean(diff), np.std(diff),  # velocity stats
                np.mean(np.diff(diff)) if len(diff) > 1 else 0,  # acceleration
            ])
            # Range and IQR
            feats.extend([
                np.max(col) - np.min(col),  # range
                np.percentile(col, 75) - np.percentile(col, 25),  # IQR
            ])

        all_features.append(feats)

    return np.array(all_features)

def load_data(task):
    """Load and combine all data"""
    X_all, y_all, ids_all = [], [], []
    for split in ['train', 'valid', 'test']:
        with open(data_dir / f'{task}_{split}_v2.pkl', 'rb') as f:
            d = pickle.load(f)
        X_all.append(d['X'])
        y_all.append(d['y'])
        ids_all.append(d['ids'])

    X_3d = np.vstack(X_all)
    y = np.hstack(y_all)
    ids = np.hstack(ids_all)
    subjects = np.array([id.rsplit('_', 1)[-1] for id in ids])

    # Extract features
    print(f"  Extracting features from {X_3d.shape}...", end=' ', flush=True)
    X = extract_features(X_3d, task)
    print(f"-> {X.shape}", flush=True)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, subjects

# ===================== ORDINAL REGRESSION =====================

class OrdinalClassifier:
    """Ordinal regression using cumulative model"""
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or LogisticRegression(max_iter=1000)
        self.classifiers = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.classifiers = []

        # Create K-1 binary classifiers for K classes
        for k in self.classes_[:-1]:
            y_binary = (y > k).astype(int)
            clf = type(self.base_estimator)(**self.base_estimator.get_params())
            clf.fit(X, y_binary)
            self.classifiers.append(clf)

        return self

    def predict_proba(self, X):
        # Get cumulative probabilities P(Y > k)
        cumprobs = np.column_stack([
            clf.predict_proba(X)[:, 1] for clf in self.classifiers
        ])

        # Convert to class probabilities
        probs = np.zeros((X.shape[0], len(self.classes_)))
        probs[:, 0] = 1 - cumprobs[:, 0]
        for k in range(1, len(self.classes_) - 1):
            probs[:, k] = cumprobs[:, k-1] - cumprobs[:, k]
        probs[:, -1] = cumprobs[:, -1]

        # Clip and normalize
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# ===================== EVALUATION =====================

def run_loso_cv(model, X, y, subjects, model_name='Model'):
    """Run LOSO cross-validation"""
    logo = LeaveOneGroupOut()
    fold_accs = []
    all_preds, all_targets = [], []

    for train_idx, test_idx in logo.split(X, y, subjects):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train & predict
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        m.fit(X_train_scaled, y_train)
        y_pred = m.predict(X_test_scaled)

        fold_accs.append(accuracy_score(y_test, y_pred))
        all_preds.extend(y_pred)
        all_targets.extend(y_test)

    overall_acc = accuracy_score(all_targets, all_preds)
    overall_mae = mean_absolute_error(all_targets, all_preds)
    within1 = np.mean(np.abs(np.array(all_targets) - np.array(all_preds)) <= 1)

    return {
        'model': model_name,
        'mean_acc': np.mean(fold_accs),
        'std_acc': np.std(fold_accs),
        'overall_acc': overall_acc,
        'mae': overall_mae,
        'within1': within1
    }

def select_features_importance(X, y, n_features=100):
    """Select top features by importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-n_features:]
    return top_idx

def run_task(task):
    print('=' * 60, flush=True)
    print(f'{task.upper()} - Fast ML', flush=True)
    print('=' * 60, flush=True)

    X, y, subjects = load_data(task)
    print(f'Samples: {len(y)}, Subjects: {len(np.unique(subjects))}, Features: {X.shape[1]}', flush=True)
    print(flush=True)

    results = []

    # 1. Random Forest with different configs
    print('--- Random Forest Variants ---', flush=True)
    configs = [
        ('RF_100', 100, 10, None),
        ('RF_200', 200, 15, None),
        ('RF_300', 300, 20, None),
        ('RF_500', 500, 25, None),
        ('RF_300_d30', 300, 30, None),
        ('RF_300_d40', 300, 40, None),
    ]

    for name, n_est, depth, max_feat in configs:
        model = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth,
            max_features=max_feat, random_state=42, n_jobs=-1
        )
        r = run_loso_cv(model, X, y, subjects, name)
        print(f'  {name}: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%} (MAE: {r["mae"]:.2f})', flush=True)
        results.append(r)

    # 2. Extra Trees Variants
    print('\n--- Extra Trees Variants ---', flush=True)
    for name, n_est, depth in [('ET_300', 300, 20), ('ET_500', 500, 25), ('ET_300_d30', 300, 30)]:
        model = ExtraTreesClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
        r = run_loso_cv(model, X, y, subjects, name)
        print(f'  {name}: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%} (MAE: {r["mae"]:.2f})', flush=True)
        results.append(r)

    # 3. Feature Selection + Best Model
    print('\n--- Feature Selection ---', flush=True)
    for n_feat in [50, 100, 200, 300]:
        top_idx = select_features_importance(X, y, n_features=min(n_feat, X.shape[1]))
        X_selected = X[:, top_idx]
        model = ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        r = run_loso_cv(model, X_selected, y, subjects, f'ET_top{n_feat}')
        print(f'  ET_top{n_feat}: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%} (MAE: {r["mae"]:.2f})', flush=True)
        results.append(r)

    # 4. Ordinal Regression
    print('\n--- Ordinal Regression ---', flush=True)
    for C in [0.1, 1.0, 10.0]:
        ordinal = OrdinalClassifier(LogisticRegression(max_iter=1000, C=C))
        r = run_loso_cv(ordinal, X, y, subjects, f'Ordinal_C{C}')
        print(f'  Ordinal_C{C}: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%} (MAE: {r["mae"]:.2f})', flush=True)
        results.append(r)

    # Ridge Ordinal
    ordinal_ridge = OrdinalClassifier(RidgeClassifier(alpha=1.0))
    r = run_loso_cv(ordinal_ridge, X, y, subjects, 'Ordinal_Ridge')
    print(f'  Ordinal_Ridge: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%} (MAE: {r["mae"]:.2f})', flush=True)
    results.append(r)

    # Best result
    best = max(results, key=lambda x: x['mean_acc'])
    print(f'\n★ Best: {best["model"]} - {best["mean_acc"]:.1%} +/- {best["std_acc"]:.1%}', flush=True)

    return results

def main():
    print('=' * 60, flush=True)
    print('PD4T - Fast ML (RF/ET + Feature Selection + Ordinal)', flush=True)
    print('=' * 60, flush=True)
    print(flush=True)

    all_results = {}
    all_results['gait'] = run_task('gait')
    print(flush=True)
    all_results['finger'] = run_task('finger')

    # Final Summary
    print(flush=True)
    print('=' * 60, flush=True)
    print('FINAL SUMMARY', flush=True)
    print('=' * 60, flush=True)

    for task in ['gait', 'finger']:
        baseline = 67.1 if task == 'gait' else 59.1
        prev_best = 70.4 if task == 'gait' else 58.3

        print(f'\n{task.upper()} (Baseline: {baseline}%, Previous Best: {prev_best}%):', flush=True)
        sorted_results = sorted(all_results[task], key=lambda x: x['mean_acc'], reverse=True)
        for r in sorted_results[:5]:  # Top 5
            marker = '★' if r['mean_acc'] * 100 > prev_best else ''
            print(f'  {r["model"]}: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%} (MAE: {r["mae"]:.2f}) {marker}', flush=True)

    print('\nDone!', flush=True)

if __name__ == '__main__':
    main()
