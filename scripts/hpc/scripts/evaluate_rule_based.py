"""
Evaluate Rule-Based UPDRS Scorer on PD4T Finger Tapping Data
Compare with ML models
"""
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import pearsonr


def rule_based_score_finger_tapping(features):
    """
    Rule-based UPDRS scoring for finger tapping
    Based on backend/services/updrs_scorer.py thresholds

    Input: v4 aggregated features (35 features)
    Features:
    - Amplitude: amp_mean, amp_std, amp_cv, amp_decay_linear, breakpoint_norm, slope1, slope2
    - Speed: opening_mean, closing_mean, speed_cv, opening_cv, max_speed, min_speed, closing_cv
    - Frequency: tap_frequency, cycle_mean, cycle_std, cycle_cv, num_taps, duration, taps_per_sec
    - Hesitation: hesitation_count, hesitation_ratio, longest_pause, mean_pause, intertap_cv, freeze_count, movement_time_ratio
    - Temporal: first5_amp, last5_amp, amp_ratio, exp_decay_rate, exp_r2, sequence_trend, rhythm_regularity
    """
    # Extract key features (indices based on v4 feature order)
    # Frequency features
    tap_frequency = features[14]  # tap_frequency
    taps_per_sec = features[20]   # taps_per_sec

    # Amplitude decrement
    amp_mean = features[0]
    first5_amp = features[28]
    last5_amp = features[29]
    amp_decay = features[3]  # amp_decay_linear
    amp_ratio = features[30]  # first5/last5 ratio

    # Rhythm variability
    cycle_cv = features[17]  # cycle coefficient of variation
    rhythm_regularity = features[34] if len(features) > 34 else 0.5

    # Hesitation/freezing
    hesitation_count = features[21]
    freeze_count = features[26]

    # Speed Score (0-4)
    # Normal >= 4.0 Hz, Slight >= 3.0, Mild >= 2.0, Moderate >= 1.2
    speed = taps_per_sec if taps_per_sec > 0 else tap_frequency
    if speed >= 4.0:
        speed_score = 0
    elif speed >= 3.0:
        speed_score = 1
    elif speed >= 2.0:
        speed_score = 2
    elif speed >= 1.2:
        speed_score = 3
    else:
        speed_score = 4

    # Amplitude Decrement Score (0-4)
    # Calculate decrement percentage
    if first5_amp > 0:
        decrement_pct = max(0, (1 - amp_ratio) * 100) if amp_ratio > 0 else amp_decay * 100
    else:
        decrement_pct = amp_decay * 100

    if decrement_pct < 10:
        decrement_score = 0
    elif decrement_pct < 25:
        decrement_score = 1
    elif decrement_pct < 50:
        decrement_score = 2
    elif decrement_pct < 70:
        decrement_score = 3
    else:
        decrement_score = 4

    # Rhythm Variability Score (0-4)
    # CV thresholds: Normal < 0.08, Slight < 0.18, Mild < 0.32, Moderate < 0.50
    rhythm_var = cycle_cv
    if rhythm_var < 0.08:
        rhythm_score = 0
    elif rhythm_var < 0.18:
        rhythm_score = 1
    elif rhythm_var < 0.32:
        rhythm_score = 2
    elif rhythm_var < 0.50:
        rhythm_score = 3
    else:
        rhythm_score = 4

    # Weighted combination
    # Speed 30%, Decrement 25%, Rhythm 25%, Velocity 20%
    weighted = speed_score * 0.35 + decrement_score * 0.30 + rhythm_score * 0.35
    base_score = min(4, int(weighted + 0.5))

    # Penalties for halts, hesitations, freezes
    penalty = 0.0
    if hesitation_count > 2:
        penalty += min(0.3, (hesitation_count - 2) / 5 * 0.3)
    if freeze_count > 1:
        penalty += min(0.3, (freeze_count - 1) * 0.15)

    total_score = min(4.0, base_score + penalty)

    return int(round(total_score))


def evaluate_rule_based():
    """Evaluate rule-based scoring on PD4T data"""
    print("=" * 70)
    print("Rule-Based UPDRS Scoring Evaluation")
    print("=" * 70)

    # Load v4 data (aggregated features)
    data_dir = './data'

    with open(f'{data_dir}/finger_train_v4.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(f'{data_dir}/finger_valid_v4.pkl', 'rb') as f:
        valid = pickle.load(f)
    with open(f'{data_dir}/finger_test_v4.pkl', 'rb') as f:
        test = pickle.load(f)

    # Combine all data
    X = np.vstack([np.array(train['X']), np.array(valid['X']), np.array(test['X'])])
    y_true = np.hstack([np.array(train['y']), np.array(valid['y']), np.array(test['y'])])

    print(f"\nTotal samples: {len(y_true)}")
    print(f"Features per sample: {X.shape[1]}")

    # Label distribution
    print("\nLabel distribution:")
    for i in range(5):
        count = np.sum(y_true == i)
        print(f"  UPDRS {i}: {count} ({100*count/len(y_true):.1f}%)")

    # Rule-based predictions
    y_pred = []
    for i in range(len(X)):
        pred = rule_based_score_finger_tapping(X[i])
        y_pred.append(pred)
    y_pred = np.array(y_pred)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    within1 = np.mean(np.abs(y_true - y_pred) <= 1)
    r, _ = pearsonr(y_true, y_pred)

    print(f"\nRule-Based Results:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  MAE: {mae:.3f}")
    print(f"  Within-1: {within1*100:.1f}%")
    print(f"  Pearson r: {r:.3f}")

    # Per-class accuracy
    print(f"\nPer-class Accuracy:")
    for i in range(5):
        mask = y_true == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == i)
            class_w1 = np.mean(np.abs(y_pred[mask] - i) <= 1)
            print(f"  UPDRS {i}: {class_acc*100:.1f}% (n={np.sum(mask)}, W1={class_w1*100:.1f}%)")

    # Confusion matrix
    print(f"\nPrediction Distribution:")
    for i in range(5):
        count = np.sum(y_pred == i)
        print(f"  Predicted {i}: {count} ({100*count/len(y_pred):.1f}%)")

    # Confusion matrix details
    print(f"\nConfusion Matrix:")
    print(f"{'True/Pred':<10}", end="")
    for j in range(5):
        print(f"{j:>8}", end="")
    print()
    print("-" * 50)
    for i in range(5):
        print(f"UPDRS {i:<4}", end="")
        for j in range(5):
            count = np.sum((y_true == i) & (y_pred == j))
            print(f"{count:>8}", end="")
        print()

    return {
        'accuracy': accuracy,
        'mae': mae,
        'within1': within1,
        'pearson': r
    }


if __name__ == "__main__":
    evaluate_rule_based()
