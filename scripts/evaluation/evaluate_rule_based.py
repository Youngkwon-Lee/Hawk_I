"""
Rule-Based Scorer Accuracy Evaluation
PD4T stratified dataset
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from dataclasses import dataclass

@dataclass
class SimpleFingerTappingMetrics:
    tapping_speed: float
    amplitude_decrement: float
    rhythm_variability: float
    velocity_decrement: float
    hesitation_count: int
    halt_count: int
    freeze_episodes: int

@dataclass
class SimpleGaitMetrics:
    walking_speed: float
    cadence: float
    stride_length: float
    stride_variability: float
    step_height_mean: float
    arm_swing_amplitude_mean: float
    arm_swing_asymmetry: float
    festination_index: float
    step_count: int

def score_finger_tapping_rule(m):
    sp = m.tapping_speed
    ss = 0 if sp >= 4.0 else (1 if sp >= 3.0 else (2 if sp >= 2.0 else (3 if sp >= 1.2 else 4)))
    dec = m.amplitude_decrement
    ds = 0 if dec < 10 else (1 if dec < 25 else (2 if dec < 50 else (3 if dec < 70 else 4)))
    rv = m.rhythm_variability
    rs = 0 if rv < 0.08 else (1 if rv < 0.18 else (2 if rv < 0.32 else (3 if rv < 0.50 else 4)))
    vd = m.velocity_decrement
    vs = 0 if vd < 10 else (1 if vd < 25 else (2 if vd < 45 else (3 if vd < 60 else 4)))
    w = ss * 0.30 + ds * 0.25 + rs * 0.20 + vs * 0.25
    bs = min(4, int(w + 0.5))
    pen = 0.0
    if m.hesitation_count > 2: pen += min(0.2, (m.hesitation_count - 2) / 5 * 0.2)
    if m.halt_count > 1: pen += min(0.3, (m.halt_count - 1) / 3 * 0.3)
    if m.freeze_episodes > 1: pen += min(0.2, (m.freeze_episodes - 1) * 0.1)
    return int(round(min(4.0, bs + pen)))

def score_gait_rule(m):
    arm = m.arm_swing_amplitude_mean
    if arm < 1.0:
        asc = 0 if arm >= 0.15 else (1 if arm >= 0.12 else (2 if arm >= 0.08 else (3 if arm >= 0.05 else 4)))
    else:
        asc = 0 if arm >= 20 else (1 if arm >= 15 else (2 if arm >= 10 else (3 if arm >= 5 else 4)))
    sp = m.walking_speed
    ssc = 0 if sp >= 1.0 else (1 if sp >= 0.8 else (2 if sp >= 0.6 else (3 if sp >= 0.4 else 4)))
    cad = m.cadence
    if cad > 150: csc = 2
    elif cad > 135: csc = 1
    elif 100 <= cad <= 120: csc = 0
    elif 85 <= cad < 100 or 120 < cad <= 135: csc = 1
    else: csc = 2
    sh = m.step_height_mean
    if sh < 0.1:
        shc = 0 if sh >= 0.06 else (1 if sh >= 0.05 else (2 if sh >= 0.04 else (3 if sh >= 0.03 else 4)))
    else:
        shc = 0 if sh >= 0.12 else (1 if sh >= 0.10 else (2 if sh >= 0.07 else (3 if sh >= 0.04 else 4)))
    sl = m.stride_length
    if sl < 0.5:
        slc = 0 if sl >= 0.35 else (1 if sl >= 0.28 else (2 if sl >= 0.22 else 3))
    else:
        slc = 0 if sl >= 1.2 else (1 if sl >= 0.9 else (2 if sl >= 0.6 else 3))
    w = asc * 0.30 + ssc * 0.20 + csc * 0.15 + shc * 0.15 + slc * 0.20
    bs = min(4, int(w + 0.5))
    pen = 0.0
    var_threshold = 25 if m.stride_variability > 1 else 0.08
    if m.stride_variability > var_threshold:
        pen += min(0.3, (m.stride_variability - var_threshold) / (var_threshold * 2.5) * 0.3)
    if m.arm_swing_asymmetry > 0.20: pen += min(0.4, (m.arm_swing_asymmetry - 0.20) / 0.30 * 0.4)
    if m.festination_index > 0.05: pen += min(0.3, m.festination_index * 2)
    if m.step_count < 10: pen += 0.2
    return int(round(min(4.0, bs + pen)))

def evaluate_finger_tapping():
    print("=" * 60)
    print("FINGER TAPPING Rule-Based Scorer")
    print("=" * 60)
    test_df = pd.read_csv("data/processed/features/finger_tapping_test_features_stratified.csv")
    print(f"Test samples: {len(test_df)}")
    predictions, actuals = [], []
    for _, row in test_df.iterrows():
        m = SimpleFingerTappingMetrics(
            tapping_speed=row.get("tapping_speed", 3.0),
            amplitude_decrement=row.get("amplitude_decrement", 0),
            rhythm_variability=row.get("rhythm_variability", 0.15),
            velocity_decrement=row.get("velocity_decrement", 0),
            hesitation_count=int(row.get("hesitation_count", 0)),
            halt_count=int(row.get("halt_count", 0)),
            freeze_episodes=int(row.get("freeze_episodes", 0))
        )
        predictions.append(score_finger_tapping_rule(m))
        actuals.append(int(row["score"]))
    predictions, actuals = np.array(predictions), np.array(actuals)
    exact_acc = accuracy_score(actuals, predictions)
    within_1 = np.mean(np.abs(predictions - actuals) <= 1)
    mae = np.mean(np.abs(predictions - actuals))
    print(f"
Exact Acc: {exact_acc:.1%}, +/-1: {within_1:.1%}, MAE: {mae:.2f}")
    print(f"Actual: {np.bincount(actuals, minlength=5)}")
    print(f"Pred:   {np.bincount(predictions, minlength=5)}")
    print("
Confusion Matrix:")
    print(confusion_matrix(actuals, predictions, labels=[0,1,2,3,4]))
    return exact_acc, within_1, mae

def evaluate_gait():
    print("
" + "=" * 60)
    print("GAIT Rule-Based Scorer")
    print("=" * 60)
    test_df = pd.read_csv("data/processed/features/gait_test_features_stratified.csv")
    print(f"Test samples: {len(test_df)}")
    predictions, actuals = [], []
    for _, row in test_df.iterrows():
        m = SimpleGaitMetrics(
            walking_speed=row.get("walking_speed", 0.8),
            cadence=row.get("cadence", 110),
            stride_length=row.get("stride_length", 0.3),
            stride_variability=row.get("stride_variability", 30),
            step_height_mean=row.get("step_height_mean", 0.05),
            arm_swing_amplitude_mean=row.get("arm_swing_amplitude_mean", 0.12),
            arm_swing_asymmetry=row.get("arm_swing_asymmetry", 0.15),
            festination_index=row.get("festination_index", 0),
            step_count=int(row.get("step_count", 20))
        )
        predictions.append(score_gait_rule(m))
        actuals.append(int(row["score"]))
    predictions, actuals = np.array(predictions), np.array(actuals)
    exact_acc = accuracy_score(actuals, predictions)
    within_1 = np.mean(np.abs(predictions - actuals) <= 1)
    mae = np.mean(np.abs(predictions - actuals))
    print(f"
Exact Acc: {exact_acc:.1%}, +/-1: {within_1:.1%}, MAE: {mae:.2f}")
    print(f"Actual: {np.bincount(actuals, minlength=5)}")
    print(f"Pred:   {np.bincount(predictions, minlength=5)}")
    print("
Confusion Matrix:")
    print(confusion_matrix(actuals, predictions, labels=[0,1,2,3,4]))
    return exact_acc, within_1, mae

if __name__ == "__main__":
    ft_acc, ft_w1, ft_mae = evaluate_finger_tapping()
    gait_acc, gait_w1, gait_mae = evaluate_gait()
    print("
" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Finger Tapping: Exact={ft_acc:.1%}, +/-1={ft_w1:.1%}, MAE={ft_mae:.2f}")
    print(f"Gait:           Exact={gait_acc:.1%}, +/-1={gait_w1:.1%}, MAE={gait_mae:.2f}")
