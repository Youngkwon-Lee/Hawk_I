#!/usr/bin/env python3
"""
Advanced ML for PD4T - Ensemble + Feature Selection + Ordinal Regression
Goal: Beat current best (Gait 70.4%, Finger 58.3%)
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
    except:
        pass
    return False

USE_GPU = check_gpu()

try:
    import lightgbm as lgb
    HAS_LGBM = True
except:
    HAS_LGBM = False
    print("LightGBM not available, skipping")

data_dir = Path(__file__).parent.parent / 'data'
results_dir = Path(__file__).parent.parent / 'results'

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
    print(f"  Extracting features from {X_3d.shape}...", end=' ')
    X = extract_features(X_3d, task)
    print(f"-> {X.shape}")

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

# ===================== ENSEMBLE METHODS =====================

def get_base_models(use_gpu=True):
    """Get base models for ensemble"""
    # XGBoost GPU config
    xgb_params = {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
        'random_state': 42, 'verbosity': 0
    }
    if use_gpu:
        xgb_params.update({'tree_method': 'hist', 'device': 'cuda'})
    else:
        xgb_params['n_jobs'] = -1

    models = {
        'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        'XGB': xgb.XGBClassifier(**xgb_params),
        # GBM removed - too slow without parallelization
    }
    if HAS_LGBM:
        lgb_params = {'n_estimators': 200, 'max_depth': 6, 'random_state': 42, 'verbose': -1}
        if use_gpu:
            lgb_params['device'] = 'gpu'
        else:
            lgb_params['n_jobs'] = -1
        models['LGBM'] = lgb.LGBMClassifier(**lgb_params)
    return models

def get_stacking_model(use_gpu=True):
    """Create stacking ensemble"""
    xgb_params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 42, 'verbosity': 0}
    if use_gpu:
        xgb_params.update({'tree_method': 'hist', 'device': 'cuda'})
    else:
        xgb_params['n_jobs'] = -1

    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBClassifier(**xgb_params)),
    ]
    if HAS_LGBM:
        lgb_params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 42, 'verbose': -1}
        if use_gpu:
            lgb_params['device'] = 'gpu'
        else:
            lgb_params['n_jobs'] = -1
        estimators.append(('lgbm', lgb.LGBMClassifier(**lgb_params)))

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1
    )

def get_voting_model(use_gpu=True):
    """Create voting ensemble"""
    xgb_params = {'n_estimators': 200, 'max_depth': 6, 'random_state': 42, 'verbosity': 0}
    if use_gpu:
        xgb_params.update({'tree_method': 'hist', 'device': 'cuda'})
    else:
        xgb_params['n_jobs'] = -1

    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBClassifier(**xgb_params)),
    ]
    return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

# ===================== FEATURE SELECTION =====================

def select_features_importance(X, y, n_features=100):
    """Select top features by importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-n_features:]
    return top_idx

def select_features_mutual_info(X, y, n_features=100):
    """Select features by mutual information"""
    mi = mutual_info_classif(X, y, random_state=42)
    top_idx = np.argsort(mi)[-n_features:]
    return top_idx

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

def run_task(task):
    print('=' * 60)
    print(f'{task.upper()} - Advanced ML')
    print('=' * 60)

    X, y, subjects = load_data(task)
    print(f'Samples: {len(y)}, Subjects: {len(np.unique(subjects))}, Features: {X.shape[1]}')
    print()

    results = []

    # 1. Base models
    print('--- Base Models ---')
    print(f'  (GPU mode: {USE_GPU})')
    for name, model in get_base_models(USE_GPU).items():
        r = run_loso_cv(model, X, y, subjects, name)
        print(f'  {name}: {r["mean_acc"]:.1%} ± {r["std_acc"]:.1%}')
        results.append(r)

    # 2. Stacking Ensemble
    print('\n--- Ensemble Models ---')
    stacking = get_stacking_model(USE_GPU)
    r = run_loso_cv(stacking, X, y, subjects, 'Stacking')
    print(f'  Stacking: {r["mean_acc"]:.1%} ± {r["std_acc"]:.1%}')
    results.append(r)

    voting = get_voting_model(USE_GPU)
    r = run_loso_cv(voting, X, y, subjects, 'Voting')
    print(f'  Voting: {r["mean_acc"]:.1%} ± {r["std_acc"]:.1%}')
    results.append(r)

    # 3. Feature Selection
    print('\n--- Feature Selection (top 100) ---')
    top_idx = select_features_importance(X, y, n_features=min(100, X.shape[1]))
    X_selected = X[:, top_idx]

    et_selected = ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    r = run_loso_cv(et_selected, X_selected, y, subjects, 'ET_Selected')
    print(f'  ET_Selected: {r["mean_acc"]:.1%} ± {r["std_acc"]:.1%}')
    results.append(r)

    # 4. Ordinal Regression
    print('\n--- Ordinal Regression ---')
    ordinal = OrdinalClassifier(LogisticRegression(max_iter=1000, C=1.0))
    r = run_loso_cv(ordinal, X, y, subjects, 'Ordinal-LR')
    print(f'  Ordinal-LR: {r["mean_acc"]:.1%} ± {r["std_acc"]:.1%}')
    results.append(r)

    # Best result
    best = max(results, key=lambda x: x['mean_acc'])
    print(f'\n★ Best: {best["model"]} - {best["mean_acc"]:.1%} ± {best["std_acc"]:.1%}')

    return results

def main():
    print('=' * 60)
    print('PD4T - Advanced ML (Ensemble + Feature Selection + Ordinal)')
    print('=' * 60)
    print()

    all_results = {}
    all_results['gait'] = run_task('gait')
    print()
    all_results['finger'] = run_task('finger')

    # Final Summary
    print()
    print('=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)

    for task in ['gait', 'finger']:
        baseline = 67.1 if task == 'gait' else 59.1
        prev_best = 70.4 if task == 'gait' else 58.3

        print(f'\n{task.upper()} (Baseline: {baseline}%, Previous Best: {prev_best}%):')
        for r in all_results[task]:
            marker = '★' if r['mean_acc'] * 100 > prev_best else ''
            print(f'  {r["model"]}: {r["mean_acc"]:.1%} ± {r["std_acc"]:.1%} (MAE: {r["mae"]:.2f}) {marker}')

    # Save
    with open(results_dir / 'advanced_ml_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print('\nDone!')

if __name__ == '__main__':
    main()
