"""
PD4T LOSO (Leave-One-Subject-Out) Cross-Validation
30명 피험자 → 30-fold CV (가장 엄격한 평가)
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

data_dir = Path(__file__).parent.parent / 'data'
results_dir = Path(__file__).parent.parent / 'results'

def load_data(task='gait'):
    """Load and combine train/valid/test data"""
    with open(data_dir / f'{task}_train_v2.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir / f'{task}_valid_v2.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open(data_dir / f'{task}_test_v2.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
    y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])
    ids = np.hstack([train_data['ids'], valid_data['ids'], test_data['ids']])
    features = train_data['features']

    return X, y, ids, features

def extract_subject_ids(ids, task='gait'):
    """Extract patient IDs from video IDs
    Gait: 15-001760_009 -> 009
    Finger: 13-002356_r_003 -> 003
    """
    if task == 'gait':
        return np.array([id.rsplit('_', 1)[-1] for id in ids])
    else:  # finger
        return np.array([id.rsplit('_', 1)[-1] for id in ids])

def extract_gait_features(X_3d, features):
    """Extract optimized features for Gait"""
    clinical_idx = list(range(99, 129))
    clinical_features = [features[i] for i in clinical_idx]

    remove_features = ['hip_height', 'left_hip_angle', 'right_hip_angle', 'shoulder_asymmetry']
    keep_idx = [clinical_idx[i] for i, f in enumerate(clinical_features) if f not in remove_features]

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

        # Derived features
        # CV features
        for fname in ['trunk_angle_vel', 'body_sway_vel', 'stride_proxy_vel']:
            idx = clinical_idx[clinical_features.index(fname)]
            col = sample[:, idx]
            feats.append(np.std(col) / (np.mean(np.abs(col)) + 1e-6))

        # Arm asymmetry
        left_arm = sample[:, clinical_idx[clinical_features.index('left_arm_swing')]]
        right_arm = sample[:, clinical_idx[clinical_features.index('right_arm_swing')]]
        feats.append(np.mean(np.abs(left_arm - right_arm)) / (np.mean(np.abs(left_arm) + np.abs(right_arm)) + 1e-6))

        # Knee asymmetry
        left_knee = sample[:, clinical_idx[clinical_features.index('left_knee_angle')]]
        right_knee = sample[:, clinical_idx[clinical_features.index('right_knee_angle')]]
        feats.append(np.mean(np.abs(left_knee - right_knee)) / (np.mean(np.abs(left_knee) + np.abs(right_knee)) + 1e-6))

        # Rhythm (autocorrelation)
        stride = sample[:, clinical_idx[clinical_features.index('stride_proxy')]]
        autocorr = np.corrcoef(stride[:-5], stride[5:])[0, 1] if len(stride) > 10 else 0
        feats.append(0 if np.isnan(autocorr) else autocorr)

        # Jerk
        trunk_vel = sample[:, clinical_idx[clinical_features.index('trunk_angle_vel')]]
        jerk = np.diff(trunk_vel)
        feats.append(np.sqrt(np.mean(jerk**2)) if len(jerk) > 0 else 0)

        # Freeze ratio
        body_sway = sample[:, clinical_idx[clinical_features.index('body_sway')]]
        threshold = np.percentile(np.abs(body_sway), 10)
        feats.append(np.mean(np.abs(body_sway) < threshold))

        # Cadence
        step_vel = sample[:, clinical_idx[clinical_features.index('step_width_vel')]]
        feats.append(np.sum(np.abs(np.diff(np.sign(step_vel))) > 0) / (len(step_vel) / 30))

        all_features.append(feats)

    return np.array(all_features)

def extract_finger_features(X_3d):
    """Extract clinical features for Finger Tapping"""
    clinical_idx = list(range(63, 73))  # 10 clinical features

    n_samples = X_3d.shape[0]
    all_features = []

    for sample_idx in range(n_samples):
        sample = X_3d[sample_idx]
        feats = []

        for idx in clinical_idx:
            col = sample[:, idx]
            feats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75)
            ])

        all_features.append(feats)

    return np.array(all_features)

def run_loso_cv(task='gait'):
    """Run Leave-One-Subject-Out Cross-Validation"""
    print('=' * 70)
    print(f'{task.upper()} - LOSO (Leave-One-Subject-Out) 30-Fold CV')
    print('=' * 70)

    # Load data
    X_raw, y, ids, features = load_data(task)
    subjects = extract_subject_ids(ids, task)

    # Extract features
    if task == 'gait':
        X = extract_gait_features(X_raw, features)
    else:
        X = extract_finger_features(X_raw)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    unique_subjects = np.unique(subjects)
    print(f'Total samples: {len(y)}')
    print(f'Unique subjects: {len(unique_subjects)}')
    print(f'Features: {X.shape[1]}')
    print()

    # Models
    models = {
        'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        'XGB': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0),
    }

    # LOSO CV
    logo = LeaveOneGroupOut()

    results = {}

    for name, model in models.items():
        fold_accs = []
        fold_maes = []
        fold_within1 = []
        y_true_all = []
        y_pred_all = []

        print(f'[{name}]')

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, subjects)):
            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            # Train
            m = type(model)(**model.get_params())
            m.fit(X_train, y[train_idx])

            # Predict
            y_pred = m.predict(X_test)

            # Metrics
            acc = accuracy_score(y[test_idx], y_pred)
            mae = mean_absolute_error(y[test_idx], y_pred)
            within1 = np.mean(np.abs(y[test_idx] - y_pred) <= 1)

            fold_accs.append(acc)
            fold_maes.append(mae)
            fold_within1.append(within1)

            y_true_all.extend(y[test_idx])
            y_pred_all.extend(y_pred)

            test_subject = subjects[test_idx[0]]
            if fold < 5 or fold >= 25:  # Print first 5 and last 5
                print(f'  Fold {fold+1:2d} (Patient {test_subject}): '
                      f'Acc={acc:.1%}, Samples={len(test_idx)}')

        if len(unique_subjects) > 10:
            print(f'  ... (showing first 5 and last 5 of {len(unique_subjects)} folds)')

        # Overall metrics
        overall_acc = accuracy_score(y_true_all, y_pred_all)
        overall_mae = mean_absolute_error(y_true_all, y_pred_all)
        overall_within1 = np.mean(np.abs(np.array(y_true_all) - np.array(y_pred_all)) <= 1)

        results[name] = {
            'fold_accs': fold_accs,
            'fold_maes': fold_maes,
            'fold_within1': fold_within1,
            'mean_acc': np.mean(fold_accs),
            'std_acc': np.std(fold_accs),
            'mean_mae': np.mean(fold_maes),
            'mean_within1': np.mean(fold_within1),
            'overall_acc': overall_acc,
            'overall_mae': overall_mae,
            'overall_within1': overall_within1,
            'y_true': y_true_all,
            'y_pred': y_pred_all
        }

        print(f'  >> Mean: {np.mean(fold_accs):.1%} +/- {np.std(fold_accs):.1%}')
        print(f'  >> Overall: Acc={overall_acc:.1%}, MAE={overall_mae:.3f}, Within-1={overall_within1:.1%}')
        print()

    return results

def main():
    print('=' * 70)
    print('PD4T LOSO (Leave-One-Subject-Out) Cross-Validation')
    print('30 subjects → 30-fold CV (strictest evaluation)')
    print('=' * 70)
    print()

    # Gait
    gait_results = run_loso_cv('gait')

    print()

    # Finger Tapping
    finger_results = run_loso_cv('finger')

    # Summary
    print()
    print('=' * 70)
    print('FINAL SUMMARY - LOSO 30-Fold CV')
    print('=' * 70)
    print()

    print('GAIT:')
    print(f'  Baseline: 67.1%')
    for name, res in gait_results.items():
        print(f'  {name}: {res["mean_acc"]:.1%} +/- {res["std_acc"]:.1%} '
              f'(Overall: {res["overall_acc"]:.1%}, Within-1: {res["overall_within1"]:.1%})')

    print()
    print('FINGER TAPPING:')
    print(f'  Baseline: 59.1%')
    for name, res in finger_results.items():
        print(f'  {name}: {res["mean_acc"]:.1%} +/- {res["std_acc"]:.1%} '
              f'(Overall: {res["overall_acc"]:.1%}, Within-1: {res["overall_within1"]:.1%})')

    # Save results
    all_results = {
        'gait': gait_results,
        'finger': finger_results
    }

    with open(results_dir / 'loso_cv_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print()
    print(f'Results saved to {results_dir / "loso_cv_results.pkl"}')

if __name__ == '__main__':
    main()
