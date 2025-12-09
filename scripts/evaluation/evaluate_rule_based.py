"""
Rule-Based Scorer 정확도 평가
PD4T stratified 데이터셋 기반
"""
import sys
sys.path.insert(0, 'backend')

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from services.metrics_calculator import FingerTappingMetrics, GaitMetrics
from services.updrs_scorer import UPDRSScorer

def evaluate_finger_tapping():
    print("=" * 60)
    print("FINGER TAPPING Rule-Based Scorer 평가")
    print("=" * 60)
    
    # Load stratified test data
    test_df = pd.read_csv('data/processed/features/finger_tapping_test_features_stratified.csv')
    print(f"테스트 샘플: {len(test_df)}개")
    
    scorer = UPDRSScorer(method="rule")
    predictions = []
    actuals = []
    
    for _, row in test_df.iterrows():
        # Create metrics from features
        metrics = FingerTappingMetrics(
            tapping_speed=row.get('tapping_freq', 3.0),
            amplitude_mean=row.get('amplitude_ratio', 0.5),
            amplitude_std=row.get('amplitude_variability', 0.1),
            amplitude_decrement=row.get('fatigue_index', 0) * 100,  # Convert to %
            rhythm_variability=row.get('velocity_variability', 0.15),
            velocity_decrement=row.get('fatigue_index', 0) * 100,
            hesitation_count=int(row.get('hesitation_rate', 0) * 10),
            halt_count=0,
            freeze_episodes=0
        )
        
        result = scorer.score_finger_tapping(metrics)
        predictions.append(result.base_score)
        actuals.append(int(row['score']))
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Metrics
    exact_acc = accuracy_score(actuals, predictions)
    within_1 = np.mean(np.abs(predictions - actuals) <= 1)
    mae = np.mean(np.abs(predictions - actuals))
    
    print(f"\n정확도 (Exact): {exact_acc:.1%}")
    print(f"정확도 (±1점): {within_1:.1%}")
    print(f"MAE: {mae:.2f}")
    
    print("\n점수별 분포:")
    print(f"실제: {np.bincount(actuals, minlength=5)}")
    print(f"예측: {np.bincount(predictions, minlength=5)}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(actuals, predictions, labels=[0,1,2,3,4])
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(actuals, predictions, labels=[0,1,2,3,4], zero_division=0))
    
    return exact_acc, within_1, mae

def evaluate_gait():
    print("\n" + "=" * 60)
    print("GAIT Rule-Based Scorer 평가")
    print("=" * 60)
    
    # Load stratified test data
    test_df = pd.read_csv('data/processed/features/gait_test_features_stratified.csv')
    print(f"테스트 샘플: {len(test_df)}개")
    
    scorer = UPDRSScorer(method="rule")
    predictions = []
    actuals = []
    
    for _, row in test_df.iterrows():
        metrics = GaitMetrics(
            walking_speed=row.get('walking_speed', 0.8),
            cadence=row.get('cadence', 110),
            stride_length=row.get('stride_length', 0.3),
            stride_variability=row.get('stride_variability', 30),
            step_height_mean=row.get('step_height_mean', 0.05),
            step_height_std=row.get('step_height_std', 0.01),
            arm_swing_amplitude_mean=row.get('arm_swing_amplitude_mean', 0.12),
            arm_swing_asymmetry=row.get('arm_swing_asymmetry', 0.15),
            trunk_sway=row.get('trunk_sway', 0.02),
            festination_index=row.get('festination_index', 0),
            step_count=int(row.get('step_count', 20))
        )
        
        result = scorer.score_gait(metrics)
        predictions.append(result.base_score)
        actuals.append(int(row['score']))
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Metrics
    exact_acc = accuracy_score(actuals, predictions)
    within_1 = np.mean(np.abs(predictions - actuals) <= 1)
    mae = np.mean(np.abs(predictions - actuals))
    
    print(f"\n정확도 (Exact): {exact_acc:.1%}")
    print(f"정확도 (±1점): {within_1:.1%}")
    print(f"MAE: {mae:.2f}")
    
    print("\n점수별 분포:")
    print(f"실제: {np.bincount(actuals, minlength=5)}")
    print(f"예측: {np.bincount(predictions, minlength=5)}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(actuals, predictions, labels=[0,1,2,3,4])
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(actuals, predictions, labels=[0,1,2,3,4], zero_division=0))
    
    return exact_acc, within_1, mae

if __name__ == "__main__":
    ft_acc, ft_w1, ft_mae = evaluate_finger_tapping()
    gait_acc, gait_w1, gait_mae = evaluate_gait()
    
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"Finger Tapping: Exact={ft_acc:.1%}, ±1점={ft_w1:.1%}, MAE={ft_mae:.2f}")
    print(f"Gait:           Exact={gait_acc:.1%}, ±1점={gait_w1:.1%}, MAE={gait_mae:.2f}")
