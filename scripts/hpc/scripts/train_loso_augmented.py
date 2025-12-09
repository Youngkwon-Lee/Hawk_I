"""
PD4T LOSO CV with Data Augmentation
- Temporal shift (시간축 이동)
- Gaussian noise (노이즈 추가)
- Speed variation (속도 변화)
- Random crop (랜덤 자르기)
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, mean_absolute_error
import xgboost as xgb
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

data_dir = Path(__file__).parent.parent / 'data'
results_dir = Path(__file__).parent.parent / 'results'

# ===================== DATA AUGMENTATION =====================

def temporal_shift(X, max_shift=10):
    """Shift time series by random amount"""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(X, shift, axis=0)

def add_gaussian_noise(X, noise_std=0.02):
    """Add Gaussian noise to features"""
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

def speed_variation(X, speed_range=(0.8, 1.2)):
    """Change speed by resampling - always return original length"""
    original_len = len(X)
    speed = np.random.uniform(*speed_range)

    # Resample each feature to new length, then resize back to original
    resampled = np.zeros_like(X)
    for i in range(X.shape[1]):
        # Zoom to different length
        zoomed = zoom(X[:, i], speed, order=1)
        # Resize back to original length
        if len(zoomed) >= original_len:
            resampled[:, i] = zoomed[:original_len]
        else:
            resampled[:len(zoomed), i] = zoomed
            resampled[len(zoomed):, i] = zoomed[-1]  # Pad with last value

    return resampled

def random_crop(X, crop_ratio=0.9):
    """Randomly crop a portion of the sequence"""
    crop_len = int(len(X) * crop_ratio)
    start = np.random.randint(0, len(X) - crop_len + 1)
    cropped = X[start:start+crop_len]

    # Pad back to original length
    padded = np.zeros_like(X)
    padded[:crop_len] = cropped
    padded[crop_len:] = cropped[-1]  # Repeat last
    return padded

def augment_sample(X, augment_type='all'):
    """Apply augmentation to a single sample"""
    if augment_type == 'shift':
        return temporal_shift(X)
    elif augment_type == 'noise':
        return add_gaussian_noise(X)
    elif augment_type == 'speed':
        return speed_variation(X)
    elif augment_type == 'crop':
        return random_crop(X)
    elif augment_type == 'all':
        # Apply random combination
        X_aug = X.copy()
        if np.random.random() > 0.5:
            X_aug = temporal_shift(X_aug, max_shift=5)
        if np.random.random() > 0.5:
            X_aug = add_gaussian_noise(X_aug, noise_std=0.01)
        if np.random.random() > 0.5:
            X_aug = speed_variation(X_aug, speed_range=(0.9, 1.1))
        return X_aug
    return X

def augment_dataset(X_raw, y, n_augment=2):
    """Augment entire dataset"""
    X_aug_list = [X_raw]
    y_aug_list = [y]

    for _ in range(n_augment):
        X_new = np.array([augment_sample(x, 'all') for x in X_raw])
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

def extract_subject_ids(ids, task='gait'):
    """Extract patient IDs from video IDs"""
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

def run_loso_augmented(task='gait', n_augment=3):
    """Run LOSO CV with data augmentation on training set"""
    print('=' * 70)
    print(f'{task.upper()} - LOSO CV with Data Augmentation (x{n_augment+1})')
    print('=' * 70)

    # Load raw data
    X_raw, y, ids, features = load_data(task)
    subjects = extract_subject_ids(ids, task)

    unique_subjects = np.unique(subjects)
    print(f'Total samples: {len(y)}')
    print(f'Unique subjects: {len(unique_subjects)}')
    print()

    # Models
    models = {
        'RF_300': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        'ET_300': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        'XGB': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0),
    }

    logo = LeaveOneGroupOut()
    results = {}

    for name, model in models.items():
        fold_accs = []
        y_true_all = []
        y_pred_all = []

        print(f'[{name}]')

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

            test_subject = subjects[test_idx[0]]
            if fold < 3 or fold >= 27:
                print(f'  Fold {fold+1:2d} (Patient {test_subject}): Acc={acc:.1%}')

        print(f'  ... ({len(unique_subjects)} folds total)')

        overall_acc = accuracy_score(y_true_all, y_pred_all)
        overall_within1 = np.mean(np.abs(np.array(y_true_all) - np.array(y_pred_all)) <= 1)

        results[name] = {
            'mean_acc': np.mean(fold_accs),
            'std_acc': np.std(fold_accs),
            'overall_acc': overall_acc,
            'overall_within1': overall_within1,
        }

        print(f'  >> Mean: {np.mean(fold_accs):.1%} +/- {np.std(fold_accs):.1%}')
        print(f'  >> Overall: Acc={overall_acc:.1%}, Within-1={overall_within1:.1%}')
        print()

    return results

def main():
    print('=' * 70)
    print('PD4T LOSO CV with Data Augmentation')
    print('Augmentation: Temporal shift, Gaussian noise, Speed variation')
    print('=' * 70)
    print()

    # Test different augmentation levels
    for n_aug in [2, 4]:
        print(f'\n{"="*70}')
        print(f'AUGMENTATION LEVEL: x{n_aug+1} (original + {n_aug} augmented)')
        print(f'{"="*70}\n')

        gait_results = run_loso_augmented('gait', n_augment=n_aug)
        finger_results = run_loso_augmented('finger', n_augment=n_aug)

        print()
        print(f'=== SUMMARY (Augmentation x{n_aug+1}) ===')
        print()
        print('GAIT (Baseline: 67.1%, LOSO no-aug: 70.4%):')
        for name, res in gait_results.items():
            print(f'  {name}: {res["mean_acc"]:.1%} +/- {res["std_acc"]:.1%}')

        print()
        print('FINGER (Baseline: 59.1%, LOSO no-aug: 58.3%):')
        for name, res in finger_results.items():
            print(f'  {name}: {res["mean_acc"]:.1%} +/- {res["std_acc"]:.1%}')

if __name__ == '__main__':
    main()
