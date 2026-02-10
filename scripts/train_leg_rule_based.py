"""
Rule-based Leg Agility Scoring
Based on physical features: velocity, amplitude, rhythm, fatigue
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error

# Auto-detect data path
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
data_dir = base_dir / 'data'

# Load data
train_path = data_dir / 'leg_agility_train_v2.pkl'
test_path = data_dir / 'leg_agility_test_v2.pkl'

with open(train_path, 'rb') as f:
    train_data = pickle.load(f)
with open(test_path, 'rb') as f:
    test_data = pickle.load(f)

X_train = train_data['X']  # (N, 150, 18) - 150 frames, 6 landmarks x 3 coords
y_train = train_data['y']
X_test = test_data['X']
y_test = test_data['y']

print("="*60)
print("Rule-Based Leg Agility Scoring")
print("="*60)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


def extract_physical_features(X):
    """
    Extract physical features from skeleton sequences

    Features:
    1. Peak velocity (max speed of leg movement)
    2. Mean amplitude (average movement range)
    3. Rhythm variability (consistency of movement)
    4. Fatigue rate (velocity decline over time)
    5. Movement frequency (tapping speed)
    """
    N, T, F = X.shape  # (samples, frames, features)

    features = []

    for i in range(N):
        seq = X[i]  # (150, 18)

        # Reshape to (150, 6, 3) - 6 landmarks, 3 coords (x,y,z)
        landmarks = seq.reshape(T, 6, 3)

        # Focus on ankle movement (most dynamic in leg agility)
        # Landmarks: 0-1: Hip, 2-3: Knee, 4-5: Ankle
        left_ankle = landmarks[:, 4, :]   # (150, 3)
        right_ankle = landmarks[:, 5, :]  # (150, 3)

        # 1. Velocity (frame-to-frame displacement)
        left_vel = np.sqrt(np.sum(np.diff(left_ankle, axis=0)**2, axis=1))  # (149,)
        right_vel = np.sqrt(np.sum(np.diff(right_ankle, axis=0)**2, axis=1))

        peak_vel_left = np.max(left_vel) if len(left_vel) > 0 else 0
        peak_vel_right = np.max(right_vel) if len(right_vel) > 0 else 0
        mean_vel_left = np.mean(left_vel) if len(left_vel) > 0 else 0
        mean_vel_right = np.mean(right_vel) if len(right_vel) > 0 else 0

        # 2. Amplitude (vertical movement range - focus on y-axis)
        amp_left_y = np.ptp(left_ankle[:, 1])  # peak-to-peak
        amp_right_y = np.ptp(right_ankle[:, 1])

        # 3. Rhythm variability (std of velocity)
        rhythm_left = np.std(left_vel) if len(left_vel) > 0 else 0
        rhythm_right = np.std(right_vel) if len(right_vel) > 0 else 0

        # 4. Fatigue rate (velocity decline over time)
        # Compare first half vs second half
        mid = len(left_vel) // 2
        if mid > 0:
            first_half_left = np.mean(left_vel[:mid])
            second_half_left = np.mean(left_vel[mid:])
            fatigue_left = (first_half_left - second_half_left) / (first_half_left + 1e-8)

            first_half_right = np.mean(right_vel[:mid])
            second_half_right = np.mean(right_vel[mid:])
            fatigue_right = (first_half_right - second_half_right) / (first_half_right + 1e-8)
        else:
            fatigue_left = 0
            fatigue_right = 0

        # 5. Movement frequency (count peaks)
        from scipy.signal import find_peaks
        peaks_left, _ = find_peaks(left_vel, height=np.mean(left_vel))
        peaks_right, _ = find_peaks(right_vel, height=np.mean(right_vel))
        freq_left = len(peaks_left)
        freq_right = len(peaks_right)

        # Combine left and right (average)
        feature_vec = [
            (peak_vel_left + peak_vel_right) / 2,    # 0: peak velocity
            (mean_vel_left + mean_vel_right) / 2,    # 1: mean velocity
            (amp_left_y + amp_right_y) / 2,          # 2: amplitude
            (rhythm_left + rhythm_right) / 2,        # 3: rhythm variability
            (fatigue_left + fatigue_right) / 2,      # 4: fatigue rate
            (freq_left + freq_right) / 2,            # 5: frequency
        ]

        features.append(feature_vec)

    return np.array(features)


print("\n[1/4] Extracting physical features from train set...")
train_features = extract_physical_features(X_train)
print(f"  Train features: {train_features.shape}")

print("\n[2/4] Extracting physical features from test set...")
test_features = extract_physical_features(X_test)
print(f"  Test features: {test_features.shape}")

print("\n[3/4] Analyzing feature distributions...")
print("\nFeature ranges (train set):")
feature_names = ['Peak Velocity', 'Mean Velocity', 'Amplitude', 'Rhythm Var', 'Fatigue Rate', 'Frequency']
for i, name in enumerate(feature_names):
    print(f"  {name:20s}: min={train_features[:, i].min():.4f}, max={train_features[:, i].max():.4f}, mean={train_features[:, i].mean():.4f}")

print("\n[4/4] Applying rule-based scoring...")

# Analyze correlation with ground truth
print("\nFeature correlation with UPDRS score (train set):")
for i, name in enumerate(feature_names):
    corr, _ = pearsonr(train_features[:, i], y_train)
    print(f"  {name:20s}: {corr:+.3f}")


def rule_based_score(features):
    """
    Rule-based scoring based on physical features

    UPDRS Leg Agility criteria:
    0: Normal - fast, large amplitude
    1: Slight - slightly slow or small amplitude
    2: Mild - clearly slow or moderate amplitude reduction
    3: Moderate - very slow or small amplitude
    4: Severe - barely able or very small movement
    """
    scores = []

    # Normalize features (using train statistics)
    peak_vel = features[:, 0]
    mean_vel = features[:, 1]
    amplitude = features[:, 2]
    rhythm_var = features[:, 3]
    fatigue = features[:, 4]
    frequency = features[:, 5]

    for i in range(len(features)):
        score = 0

        # Rule 1: Velocity-based scoring (inverse relationship)
        # Higher velocity = lower score (better)
        if mean_vel[i] < 0.005:
            score += 2  # Very slow
        elif mean_vel[i] < 0.010:
            score += 1  # Slow
        elif mean_vel[i] < 0.015:
            score += 0.5  # Slightly slow

        # Rule 2: Amplitude-based scoring
        # Lower amplitude = higher score (worse)
        if amplitude[i] < 0.05:
            score += 2  # Very small
        elif amplitude[i] < 0.10:
            score += 1  # Small
        elif amplitude[i] < 0.15:
            score += 0.5  # Slightly small

        # Rule 3: Rhythm variability (higher var = worse)
        if rhythm_var[i] > 0.015:
            score += 1

        # Rule 4: Fatigue (positive fatigue = worse)
        if fatigue[i] > 0.2:
            score += 1

        # Rule 5: Frequency (lower freq = worse)
        if frequency[i] < 5:
            score += 1
        elif frequency[i] < 10:
            score += 0.5

        # Clip to 0-4 range
        score = np.clip(round(score), 0, 4)
        scores.append(score)

    return np.array(scores)


# Apply rules
y_pred_train = rule_based_score(train_features)
y_pred_test = rule_based_score(test_features)

# Evaluate
mae_train = mean_absolute_error(y_train, y_pred_train)
exact_train = np.mean(y_pred_train == y_train) * 100
within1_train = np.mean(np.abs(y_pred_train - y_train) <= 1) * 100
pearson_train, _ = pearsonr(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
exact_test = np.mean(y_pred_test == y_test) * 100
within1_test = np.mean(np.abs(y_pred_test - y_test) <= 1) * 100
pearson_test, _ = pearsonr(y_test, y_pred_test)

print("\n" + "="*60)
print("RULE-BASED RESULTS")
print("="*60)

print("\nTRAIN SET:")
print(f"  MAE: {mae_train:.3f}")
print(f"  Exact: {exact_train:.1f}%")
print(f"  Within1: {within1_train:.1f}%")
print(f"  Pearson: {pearson_train:.3f}")

print("\nTEST SET:")
print(f"  MAE: {mae_test:.3f}")
print(f"  Exact: {exact_test:.1f}%")
print(f"  Within1: {within1_test:.1f}%")
print(f"  Pearson: {pearson_test:.3f}")

# Comparison
print("\n" + "="*60)
print("COMPARISON WITH ML/DL MODELS")
print("="*60)
print(f"{'Model':<30} {'MAE':<8} {'Exact':<10} {'Pearson':<10}")
print("-" * 60)
print(f"{'Random Forest':<30} {'0.584':<8} {'46.0%':<10} {'-0.024':<10}")
print(f"{'XGBoost':<30} {'0.531':<8} {'49.6%':<10} {'0.197':<10}")
print(f"{'Mamba + CORAL':<30} {'0.458':<8} {'57.9%':<10} {'0.307':<10}")
print(f"{'Rule-Based (this)':<30} {mae_test:<8.3f} {f'{exact_test:.1f}%':<10} {pearson_test:<10.3f}")

print("\n" + "="*60)
