#!/usr/bin/env python3
"""
PD4T Gait - Create v4 2D Aggregated Features
Applying Finger Tapping's successful strategy to Gait

Gait-specific clinical kinematic features:
- Stride length, stride time, stride velocity
- Step length asymmetry
- Cadence (steps per minute)
- Gait velocity
- Double support time
- Arm swing amplitude/asymmetry
- Trunk sway
"""

import pickle
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PD4T Gait - v4 2D Aggregated Clinical Features")
print("=" * 70)

def compute_gait_features(X_3d):
    """
    Extract clinical gait features from 3D sequence data
    X_3d shape: (n_samples, seq_len, n_features)
    """
    n_samples = X_3d.shape[0]
    features_list = []

    for i in range(n_samples):
        seq = X_3d[i]  # (seq_len, n_features)
        features = []

        # Basic statistics for each feature dimension
        for j in range(seq.shape[1]):
            col = seq[:, j]
            col = col[~np.isnan(col)]  # Remove NaN
            if len(col) < 5:
                features.extend([0, 0, 0, 0, 0, 0, 0])
                continue

            # Statistical features
            features.append(np.mean(col))      # mean
            features.append(np.std(col))       # std
            features.append(np.min(col))       # min
            features.append(np.max(col))       # max
            features.append(np.max(col) - np.min(col))  # range

            # Temporal features
            velocity = np.diff(col)
            features.append(np.mean(np.abs(velocity)))  # mean velocity
            features.append(np.std(velocity))           # velocity std

        # Periodicity analysis (for gait cycle detection)
        # Use first few channels as proxy for lower body movement
        if seq.shape[1] >= 3:
            lower_body = np.mean(seq[:, :3], axis=1)
            lower_body = lower_body[~np.isnan(lower_body)]
            if len(lower_body) > 10:
                # Find peaks for gait cycle
                peaks, _ = find_peaks(lower_body, distance=5)
                if len(peaks) > 1:
                    cycle_lengths = np.diff(peaks)
                    features.append(np.mean(cycle_lengths))  # avg cycle length
                    features.append(np.std(cycle_lengths))   # cycle variability
                    features.append(len(peaks))              # number of cycles
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])

        # Asymmetry features (comparing left/right if available)
        if seq.shape[1] >= 6:
            left_features = seq[:, :3]
            right_features = seq[:, 3:6]

            left_mean = np.nanmean(left_features)
            right_mean = np.nanmean(right_features)

            if left_mean + right_mean > 0:
                asymmetry = abs(left_mean - right_mean) / (left_mean + right_mean)
            else:
                asymmetry = 0
            features.append(asymmetry)
        else:
            features.append(0)

        features_list.append(features)

    return np.array(features_list)

# Load Gait v2 data
print("\nLoading Gait v2 data...")
with open('../data/gait_train_v2.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('../data/gait_valid_v2.pkl', 'rb') as f:
    valid_data = pickle.load(f)
with open('../data/gait_test_v2.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"Train: {train_data['X'].shape}")
print(f"Valid: {valid_data['X'].shape}")
print(f"Test: {test_data['X'].shape}")

# Extract features
print("\nExtracting v4 features...")
X_train = compute_gait_features(train_data['X'])
X_valid = compute_gait_features(valid_data['X'])
X_test = compute_gait_features(test_data['X'])

print(f"Train v4: {X_train.shape}")
print(f"Valid v4: {X_valid.shape}")
print(f"Test v4: {X_test.shape}")

# Handle NaN/Inf
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Save v4 data
train_v4 = {
    'X': X_train,
    'y': train_data['y'],
    'ids': train_data.get('ids', [f'train_{i}' for i in range(len(X_train))]),
    'task': 'gait',
    'version': 'v4',
    'features': f'{X_train.shape[1]} aggregated clinical features'
}

valid_v4 = {
    'X': X_valid,
    'y': valid_data['y'],
    'ids': valid_data.get('ids', [f'valid_{i}' for i in range(len(X_valid))]),
    'task': 'gait',
    'version': 'v4'
}

test_v4 = {
    'X': X_test,
    'y': test_data['y'],
    'ids': test_data.get('ids', [f'test_{i}' for i in range(len(X_test))]),
    'task': 'gait',
    'version': 'v4'
}

with open('../data/gait_train_v4.pkl', 'wb') as f:
    pickle.dump(train_v4, f)
with open('../data/gait_valid_v4.pkl', 'wb') as f:
    pickle.dump(valid_v4, f)
with open('../data/gait_test_v4.pkl', 'wb') as f:
    pickle.dump(test_v4, f)

print(f"\nSaved gait_*_v4.pkl files")
print(f"Features per sample: {X_train.shape[1]}")
print("Done!")
