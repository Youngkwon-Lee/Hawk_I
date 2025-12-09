"""
UPDRS Scoring Accuracy Test
Compares Rule-based vs ML vs Ensemble scoring
"""
import sys
sys.path.insert(0, '.')
from services.updrs_scorer import UPDRSScorer
from services.metrics_calculator import FingerTappingMetrics, GaitMetrics
import numpy as np

print('=' * 60)
print('UPDRS Scoring Accuracy Test')
print('=' * 60)

# Test cases based on MDS-UPDRS clinical criteria
finger_cases = [
    {
        'name': 'Normal',
        'expected': 0,
        'metrics': FingerTappingMetrics(
            tapping_speed=4.5, total_taps=45, duration=10.0,
            amplitude_mean=0.8, amplitude_std=0.08,
            opening_velocity_mean=2.0, closing_velocity_mean=2.0,
            peak_velocity_mean=2.5, velocity_decrement=5.0,
            amplitude_decrement=5.0, decrement_pattern='stable',
            first_half_amplitude=0.82, second_half_amplitude=0.78,
            rhythm_variability=0.05, halt_count=0, hesitation_count=0, freeze_episodes=0,
            fatigue_rate=0.03,
            velocity_first_third=2.5, velocity_mid_third=2.4, velocity_last_third=2.3,
            amplitude_first_third=0.82, amplitude_mid_third=0.80, amplitude_last_third=0.78,
            velocity_slope=-0.01, amplitude_slope=-0.01, rhythm_slope=0.0,
            variability_first_half=0.04, variability_second_half=0.05, variability_change=0.01
        )
    },
    {
        'name': 'Slight',
        'expected': 1,
        'metrics': FingerTappingMetrics(
            tapping_speed=3.5, total_taps=35, duration=10.0,
            amplitude_mean=0.6, amplitude_std=0.12,
            opening_velocity_mean=1.5, closing_velocity_mean=1.5,
            peak_velocity_mean=1.8, velocity_decrement=15.0,
            amplitude_decrement=20.0, decrement_pattern='gradual',
            first_half_amplitude=0.65, second_half_amplitude=0.52,
            rhythm_variability=0.12, halt_count=0, hesitation_count=1, freeze_episodes=0,
            fatigue_rate=0.12,
            velocity_first_third=1.9, velocity_mid_third=1.7, velocity_last_third=1.5,
            amplitude_first_third=0.68, amplitude_mid_third=0.60, amplitude_last_third=0.52,
            velocity_slope=-0.05, amplitude_slope=-0.08, rhythm_slope=0.02,
            variability_first_half=0.08, variability_second_half=0.15, variability_change=0.07
        )
    },
    {
        'name': 'Mild',
        'expected': 2,
        'metrics': FingerTappingMetrics(
            tapping_speed=2.5, total_taps=25, duration=10.0,
            amplitude_mean=0.4, amplitude_std=0.15,
            opening_velocity_mean=1.0, closing_velocity_mean=1.0,
            peak_velocity_mean=1.2, velocity_decrement=30.0,
            amplitude_decrement=40.0, decrement_pattern='steep',
            first_half_amplitude=0.50, second_half_amplitude=0.30,
            rhythm_variability=0.22, halt_count=1, hesitation_count=3, freeze_episodes=1,
            fatigue_rate=0.25,
            velocity_first_third=1.4, velocity_mid_third=1.1, velocity_last_third=0.8,
            amplitude_first_third=0.52, amplitude_mid_third=0.40, amplitude_last_third=0.28,
            velocity_slope=-0.12, amplitude_slope=-0.15, rhythm_slope=0.08,
            variability_first_half=0.12, variability_second_half=0.30, variability_change=0.18
        )
    },
    {
        'name': 'Moderate',
        'expected': 3,
        'metrics': FingerTappingMetrics(
            tapping_speed=1.5, total_taps=15, duration=10.0,
            amplitude_mean=0.25, amplitude_std=0.18,
            opening_velocity_mean=0.5, closing_velocity_mean=0.5,
            peak_velocity_mean=0.7, velocity_decrement=50.0,
            amplitude_decrement=60.0, decrement_pattern='arrest',
            first_half_amplitude=0.35, second_half_amplitude=0.14,
            rhythm_variability=0.38, halt_count=3, hesitation_count=6, freeze_episodes=3,
            fatigue_rate=0.45,
            velocity_first_third=0.9, velocity_mid_third=0.5, velocity_last_third=0.3,
            amplitude_first_third=0.38, amplitude_mid_third=0.22, amplitude_last_third=0.12,
            velocity_slope=-0.20, amplitude_slope=-0.22, rhythm_slope=0.15,
            variability_first_half=0.20, variability_second_half=0.50, variability_change=0.30
        )
    },
]

gait_cases = [
    {
        'name': 'Normal',
        'expected': 0,
        'metrics': GaitMetrics(
            walking_speed=1.3, cadence=120, step_count=20, duration=10.0,
            stride_length=1.3, stride_variability=0.05,
            swing_time_mean=0.4, stance_time_mean=0.6, swing_stance_ratio=0.67,
            double_support_time=0.15, double_support_percent=15,
            step_length_asymmetry=0.03, swing_time_asymmetry=0.02, arm_swing_asymmetry=0.05,
            left_step_count=10, right_step_count=10,
            left_stride_length=1.32, right_stride_length=1.28,
            festination_index=0.0, gait_regularity=0.95,
            arm_swing_amplitude_left=25, arm_swing_amplitude_right=24, arm_swing_amplitude_mean=24.5,
            step_height_left=0.15, step_height_right=0.14, step_height_mean=0.145,
            trunk_flexion_mean=5.0, trunk_flexion_rom=8.0,
            hip_flexion_rom_left=35, hip_flexion_rom_right=34, hip_flexion_rom_mean=34.5,
            knee_flexion_rom_left=60, knee_flexion_rom_right=58, knee_flexion_rom_mean=59,
            ankle_dorsiflexion_rom_left=20, ankle_dorsiflexion_rom_right=19, ankle_dorsiflexion_rom_mean=19.5,
            step_length_first_half=1.32, step_length_second_half=1.28, step_length_trend=-0.02,
            cadence_first_half=122, cadence_second_half=118, cadence_trend=-0.02,
            arm_swing_first_half=25, arm_swing_second_half=24, arm_swing_trend=-0.02,
            stride_variability_first_half=0.04, stride_variability_second_half=0.05, variability_trend=0.01,
            step_height_first_half=0.15, step_height_second_half=0.14, step_height_trend=-0.01
        )
    },
    {
        'name': 'Slight',
        'expected': 1,
        'metrics': GaitMetrics(
            walking_speed=1.0, cadence=105, step_count=18, duration=10.0,
            stride_length=1.0, stride_variability=0.12,
            swing_time_mean=0.35, stance_time_mean=0.65, swing_stance_ratio=0.54,
            double_support_time=0.22, double_support_percent=22,
            step_length_asymmetry=0.08, swing_time_asymmetry=0.06, arm_swing_asymmetry=0.15,
            left_step_count=9, right_step_count=9,
            left_stride_length=1.04, right_stride_length=0.96,
            festination_index=0.05, gait_regularity=0.82,
            arm_swing_amplitude_left=20, arm_swing_amplitude_right=17, arm_swing_amplitude_mean=18.5,
            step_height_left=0.12, step_height_right=0.11, step_height_mean=0.115,
            trunk_flexion_mean=10.0, trunk_flexion_rom=6.0,
            hip_flexion_rom_left=30, hip_flexion_rom_right=28, hip_flexion_rom_mean=29,
            knee_flexion_rom_left=52, knee_flexion_rom_right=48, knee_flexion_rom_mean=50,
            ankle_dorsiflexion_rom_left=15, ankle_dorsiflexion_rom_right=13, ankle_dorsiflexion_rom_mean=14,
            step_length_first_half=1.05, step_length_second_half=0.95, step_length_trend=-0.08,
            cadence_first_half=108, cadence_second_half=102, cadence_trend=-0.05,
            arm_swing_first_half=20, arm_swing_second_half=17, arm_swing_trend=-0.10,
            stride_variability_first_half=0.08, stride_variability_second_half=0.15, variability_trend=0.08,
            step_height_first_half=0.13, step_height_second_half=0.10, step_height_trend=-0.08
        )
    },
    {
        'name': 'Mild',
        'expected': 2,
        'metrics': GaitMetrics(
            walking_speed=0.7, cadence=90, step_count=15, duration=10.0,
            stride_length=0.7, stride_variability=0.22,
            swing_time_mean=0.30, stance_time_mean=0.70, swing_stance_ratio=0.43,
            double_support_time=0.30, double_support_percent=30,
            step_length_asymmetry=0.15, swing_time_asymmetry=0.12, arm_swing_asymmetry=0.30,
            left_step_count=8, right_step_count=7,
            left_stride_length=0.78, right_stride_length=0.62,
            festination_index=0.15, gait_regularity=0.65,
            arm_swing_amplitude_left=12, arm_swing_amplitude_right=8, arm_swing_amplitude_mean=10,
            step_height_left=0.08, step_height_right=0.06, step_height_mean=0.07,
            trunk_flexion_mean=18.0, trunk_flexion_rom=4.0,
            hip_flexion_rom_left=22, hip_flexion_rom_right=18, hip_flexion_rom_mean=20,
            knee_flexion_rom_left=42, knee_flexion_rom_right=35, knee_flexion_rom_mean=38.5,
            ankle_dorsiflexion_rom_left=10, ankle_dorsiflexion_rom_right=7, ankle_dorsiflexion_rom_mean=8.5,
            step_length_first_half=0.78, step_length_second_half=0.62, step_length_trend=-0.15,
            cadence_first_half=95, cadence_second_half=85, cadence_trend=-0.10,
            arm_swing_first_half=12, arm_swing_second_half=8, arm_swing_trend=-0.25,
            stride_variability_first_half=0.15, stride_variability_second_half=0.28, variability_trend=0.15,
            step_height_first_half=0.09, step_height_second_half=0.05, step_height_trend=-0.20
        )
    },
]

results = []

for method in ['rule', 'ml', 'ensemble']:
    scorer = UPDRSScorer(method=method)

    for tc in finger_cases:
        result = scorer.score_finger_tapping(tc['metrics'])
        score = result['total_score'] if isinstance(result, dict) else result.total_score
        results.append({
            'task': 'finger', 'name': tc['name'], 'method': method,
            'expected': tc['expected'], 'score': score
        })

    for tc in gait_cases:
        result = scorer.score_gait(tc['metrics'])
        score = result['total_score'] if isinstance(result, dict) else result.total_score
        results.append({
            'task': 'gait', 'name': tc['name'], 'method': method,
            'expected': tc['expected'], 'score': score
        })

# Print results
print()
print('=== Finger Tapping ===')
print(f'{"Case":<12} {"Expected":>8} {"Rule":>8} {"ML":>8} {"Ensemble":>10}')
print('-' * 50)

for name in ['Normal', 'Slight', 'Mild', 'Moderate']:
    row = [r for r in results if r['name'] == name and r['task'] == 'finger']
    if row:
        exp = row[0]['expected']
        rule = next((r['score'] for r in row if r['method'] == 'rule'), 0)
        ml = next((r['score'] for r in row if r['method'] == 'ml'), 0)
        ens = next((r['score'] for r in row if r['method'] == 'ensemble'), 0)
        print(f'{name:<12} {exp:>8} {rule:>8.1f} {ml:>8.1f} {ens:>10.1f}')

print()
print('=== Gait ===')
print(f'{"Case":<12} {"Expected":>8} {"Rule":>8} {"ML":>8} {"Ensemble":>10}')
print('-' * 50)

for name in ['Normal', 'Slight', 'Mild']:
    row = [r for r in results if r['name'] == name and r['task'] == 'gait']
    if row:
        exp = row[0]['expected']
        rule = next((r['score'] for r in row if r['method'] == 'rule'), 0)
        ml = next((r['score'] for r in row if r['method'] == 'ml'), 0)
        ens = next((r['score'] for r in row if r['method'] == 'ensemble'), 0)
        print(f'{name:<12} {exp:>8} {rule:>8.1f} {ml:>8.1f} {ens:>10.1f}')

# Accuracy
print()
print('=' * 60)
print('Accuracy Summary')
print('=' * 60)

for method in ['rule', 'ml', 'ensemble']:
    mr = [r for r in results if r['method'] == method]
    errors = [abs(r['score'] - r['expected']) for r in mr]
    mae = np.mean(errors)
    w1 = sum(1 for e in errors if e <= 1.0) / len(errors) * 100
    ex = sum(1 for e in errors if e < 0.5) / len(errors) * 100
    print(f'{method.upper():>10}: MAE={mae:.2f}, Within 1pt={w1:.0f}%, Exact={ex:.0f}%')
