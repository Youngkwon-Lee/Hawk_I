"""
PD4T LOSO CV with Data Augmentation (Optimized Fast Version)
- Reduced augmentation multiplier (1x instead of 2-4x)
- Vectorized operations where possible
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import xgboost as xgb
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

data_dir = Path(__file__).parent.parent / 'data'
results_dir = Path(__file__).parent.parent / 'results'

# ===================== DATA AUGMENTATION =====================

def temporal_shift(X, max_shift=5):
    """Shift time series by random amount"""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(X, shift, axis=0)

def add_gaussian_noise(X, noise_std=0.01):
    """Add Gaussian noise to features"""
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

def speed_variation(X, speed_range=(0.9, 1.1)):
    """Change speed by resampling - always return original length"""
    original_len = len(X)
    speed = np.random.uniform(*speed_range)

    resampled = np.zeros_like(X)
    for i in range(X.shape[1]):
        zoomed = zoom(X[:, i], speed, order=1)
        if len(zoomed) >= original_len:
            resampled[:, i] = zoomed[:original_len]
        else:
            resampled[:len(zoomed), i] = zoomed
            resampled[len(zoomed):, i] = zoomed[-1]

    return resampled

def augment_sample(X):
    """Apply random augmentation to a single sample"""
    X_aug = X.copy()

    # Apply 1-2 random augmentations
    aug_choice = np.random.randint(0, 4)

    if aug_choice == 0:
        X_aug = temporal_shift(X_aug)
    elif aug_choice == 1:
        X_aug = add_gaussian_noise(X_aug)
    elif aug_choice == 2:
        X_aug = speed_variation(X_aug)
    else:
        # Combine shift + noise (most common augmentation)
        X_aug = temporal_shift(X_aug)
        X_aug = add_gaussian_noise(X_aug)

    return X_aug

def augment_dataset(X_raw, y, n_augment=1):
    """Augment entire dataset"""
    X_aug_list = [X_raw]
    y_aug_list = [y]

    for _ in range(n_augment):
        X_new = np.array([augment_sample(x) for x in X_raw])
        X_aug_list.append(X_new)
        y_aug_list.append(y.copy())

    return np.vstack(X_aug_list), np.hstack(y_aug_list)

# ===================== FEATURE EXTRACTION =====================

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

def extract_subject_ids(ids):
    """Extract patient IDs from video IDs"""
    return np.array([id.rsplit('_', 1)[-1] for id in ids])

def extract_gait_features(X_3d, features):
    """Extract features for Gait - vectorized where possible"""
    clinical_idx = list(range(99, 129))
    clinical_features = [features[i] for i in clinical_idx]

    remove_features = ['hip_height', 'left_hip_angle', 'right_hip_angle', 'shoulder_asymmetry']
    keep_idx = [clinical_idx[i] for i, f in enumerate(clinical_features) if f not in remove_features]

    n_samples = X_3d.shape[0]
    all_features = []

    for sample_idx in range(n_samples):
        sample = X_3d[sample_idx]
        feats = []

        for idx in keep_idx:
            col = sample[:, idx]
            feats.extend([
                np.mean(col), np.std(col), np.min(col), np.max(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75)
            ])

        # Derived features
        for fname in ['trunk_angle_vel', 'body_sway_vel', 'stride_proxy_vel']:
            idx = clinical_idx[clinical_features.index(fname)]
            col = sample[:, idx]
            feats.append(np.std(col) / (np.mean(np.abs(col)) + 1e-6))

        left_arm = sample[:, clinical_idx[clinical_features.index('left_arm_swing')]]
        right_arm = sample[:, clinical_idx[clinical_features.index('right_arm_swing')]]
        feats.append(np.mean(np.abs(left_arm - right_arm)) / (np.mean(np.abs(left_arm) + np.abs(right_arm)) + 1e-6))

        left_knee = sample[:, clinical_idx[clinical_features.index('left_knee_angle')]]
        right_knee = sample[:, clinical_idx[clinical_features.index('right_knee_angle')]]
        feats.append(np.mean(np.abs(left_knee - right_knee)) / (np.mean(np.abs(left_knee) + np.abs(right_knee)) + 1e-6))

        stride = sample[:, clinical_idx[clinical_features.index('stride_proxy')]]
        autocorr = np.corrcoef(stride[:-5], stride[5:])[0, 1] if len(stride) > 10 else 0
        feats.append(0 if np.isnan(autocorr) else autocorr)

        trunk_vel = sample[:, clinical_idx[clinical_features.index('trunk_angle_vel')]]
        jerk = np.diff(trunk_vel)
        feats.append(np.sqrt(np.mean(jerk**2)) if len(jerk) > 0 else 0)

        body_sway = sample[:, clinical_idx[clinical_features.index('body_sway')]]
        threshold = np.percentile(np.abs(body_sway), 10)
        feats.append(np.mean(np.abs(body_sway) < threshold))

        step_vel = sample[:, clinical_idx[clinical_features.index('step_width_vel')]]
        feats.append(np.sum(np.abs(np.diff(np.sign(step_vel))) > 0) / (len(step_vel) / 30))

        all_features.append(feats)

    return np.array(all_features)

def extract_finger_features(X_3d):
    """Extract clinical features for Finger Tapping"""
    clinical_idx = list(range(63, 73))

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

# ===================== TRAINING =====================

def run_loso_augmented(task='gait', n_augment=1):
    """Run LOSO CV with data augmentation on training set"""
    print('=' * 60)
    print(f'{task.upper()} - LOSO CV + Augmentation (x{n_augment+1})')
    print('=' * 60)

    # Load raw data
    X_raw, y, ids, features = load_data(task)
    subjects = extract_subject_ids(ids)

    unique_subjects = np.unique(subjects)
    print(f'Total samples: {len(y)}, Subjects: {len(unique_subjects)}')

    # Best model only (RF_300 or ET_300 based on previous results)
    if task == 'gait':
        model = ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        model_name = 'ET_300'
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        model_name = 'RF_300'

    logo = LeaveOneGroupOut()

    fold_accs = []
    y_true_all = []
    y_pred_all = []

    print(f'\nRunning [{model_name}] with augmentation...')

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_raw, y, subjects)):
        # Get raw data for train and test
        X_train_raw = X_raw[train_idx]
        y_train = y[train_idx]
        X_test_raw = X_raw[test_idx]
        y_test = y[test_idx]

        # Augment training data (raw 3D sequences)
        X_train_aug, y_train_aug = augment_dataset(X_train_raw, y_train, n_augment=n_augment)

        # Extract features after augmentation
        if task == 'gait':
            X_train_feat = extract_gait_features(X_train_aug, features)
            X_test_feat = extract_gait_features(X_test_raw, features)
        else:
            X_train_feat = extract_finger_features(X_train_aug)
            X_test_feat = extract_finger_features(X_test_raw)

        X_train_feat = np.nan_to_num(X_train_feat, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_feat)
        X_test_scaled = scaler.transform(X_test_feat)

        # Train
        m = type(model)(**model.get_params())
        m.fit(X_train_scaled, y_train_aug)

        # Predict
        y_pred = m.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        if fold % 10 == 0:
            print(f'  Fold {fold+1:2d}/30: Acc={acc:.1%}')

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_within1 = np.mean(np.abs(np.array(y_true_all) - np.array(y_pred_all)) <= 1)

    print()
    print(f'  >> Mean: {np.mean(fold_accs):.1%} +/- {np.std(fold_accs):.1%}')
    print(f'  >> Overall: Acc={overall_acc:.1%}, Within-1={overall_within1:.1%}')

    return {
        'model': model_name,
        'mean_acc': np.mean(fold_accs),
        'std_acc': np.std(fold_accs),
        'overall_acc': overall_acc,
        'overall_within1': overall_within1,
    }

def main():
    print('=' * 60)
    print('PD4T LOSO CV with Data Augmentation (Fast Version)')
    print('=' * 60)
    print()

    results = {}

    # Run with 1x augmentation (2x total data)
    results['gait'] = run_loso_augmented('gait', n_augment=1)
    print()
    results['finger'] = run_loso_augmented('finger', n_augment=1)

    # Summary
    print()
    print('=' * 60)
    print('SUMMARY: LOSO + Augmentation (x2 data)')
    print('=' * 60)
    print()
    print(f'GAIT (Baseline: 67.1%, LOSO no-aug: 70.4%):')
    print(f'  {results["gait"]["model"]}: {results["gait"]["mean_acc"]:.1%} +/- {results["gait"]["std_acc"]:.1%}')
    print()
    print(f'FINGER (Baseline: 59.1%, LOSO no-aug: 58.3%):')
    print(f'  {results["finger"]["model"]}: {results["finger"]["mean_acc"]:.1%} +/- {results["finger"]["std_acc"]:.1%}')

    # Save results
    with open(results_dir / 'loso_augmented_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
