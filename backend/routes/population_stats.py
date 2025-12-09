"""
Population Statistics API
Pre-computed statistics for UPDRS score comparison
"""

from flask import Blueprint, jsonify
import pandas as pd
import numpy as np
import os

bp = Blueprint('population_stats', __name__, url_prefix='/api')

# Cache for population statistics (loaded once at startup)
_POPULATION_STATS_CACHE = {}

def get_data_path():
    """Get the path to processed features directory"""
    # Try multiple possible paths
    paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'features'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features'),
        'C:/Users/YK/tulip/Hawkeye/data/processed/features'
    ]
    for path in paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return paths[-1]  # Default fallback


def load_population_stats():
    """Load and compute population statistics from stratified CSV files"""
    global _POPULATION_STATS_CACHE

    if _POPULATION_STATS_CACHE:
        return _POPULATION_STATS_CACHE

    data_path = get_data_path()
    print(f"[PopulationStats] Loading from: {data_path}")

    stats = {}

    # Finger Tapping Statistics
    try:
        ft_files = [
            os.path.join(data_path, 'finger_tapping_train_features_stratified.csv'),
            os.path.join(data_path, 'finger_tapping_test_features_stratified.csv'),
            os.path.join(data_path, 'finger_tapping_valid_features_stratified.csv')
        ]
        ft_dfs = [pd.read_csv(f) for f in ft_files if os.path.exists(f)]

        if ft_dfs:
            ft_df = pd.concat(ft_dfs, ignore_index=True)
            stats['finger_tapping'] = compute_task_stats(ft_df, 'finger_tapping')
            print(f"[PopulationStats] Finger Tapping: {len(ft_df)} samples loaded")
    except Exception as e:
        print(f"[PopulationStats] Error loading finger tapping: {e}")
        stats['finger_tapping'] = get_default_finger_tapping_stats()

    # Gait Statistics
    try:
        gait_files = [
            os.path.join(data_path, 'gait_train_features_stratified.csv'),
            os.path.join(data_path, 'gait_test_features_stratified.csv'),
            os.path.join(data_path, 'gait_valid_features_stratified.csv')
        ]
        gait_dfs = [pd.read_csv(f) for f in gait_files if os.path.exists(f)]

        if gait_dfs:
            gait_df = pd.concat(gait_dfs, ignore_index=True)
            stats['gait'] = compute_task_stats(gait_df, 'gait')
            print(f"[PopulationStats] Gait: {len(gait_df)} samples loaded")
    except Exception as e:
        print(f"[PopulationStats] Error loading gait: {e}")
        stats['gait'] = get_default_gait_stats()

    _POPULATION_STATS_CACHE = stats
    return stats


def compute_task_stats(df: pd.DataFrame, task_type: str) -> dict:
    """Compute statistics for a specific task type"""

    # Define metrics based on task type
    if task_type == 'finger_tapping':
        metrics = {
            'tapping_freq': 'tapping_speed',
            'amplitude_ratio': 'amplitude_mean',
            'amplitude_variability': 'amplitude_std',
            'hesitation_rate': 'hesitation_count',
            'fatigue_index': 'fatigue_rate',
            'velocity_variability': 'rhythm_variability'
        }
        display_names = {
            'tapping_freq': '태핑 빈도 (Hz)',
            'amplitude_ratio': '진폭 비율',
            'amplitude_variability': '진폭 변동성 (%)',
            'hesitation_rate': '멈춤율 (%)',
            'fatigue_index': '속도저하 지수',
            'velocity_variability': '속도 변동성 (%)'
        }
    else:  # gait
        metrics = {
            'walking_speed': 'velocity_mean' if 'velocity_mean' in df.columns else 'speed',
            'stride_length': 'stride_length' if 'stride_length' in df.columns else 'step_length',
            'cadence': 'cadence' if 'cadence' in df.columns else 'step_frequency',
            'stride_variability': 'stride_variability' if 'stride_variability' in df.columns else 'cv_stride',
            'arm_swing_asymmetry': 'arm_swing_asymmetry' if 'arm_swing_asymmetry' in df.columns else 'asymmetry'
        }
        display_names = {
            'walking_speed': '보행 속도 (m/s)',
            'stride_length': '보폭 길이 (m)',
            'cadence': '보행률 (steps/min)',
            'stride_variability': '보폭 변동성 (%)',
            'arm_swing_asymmetry': '팔 흔들기 비대칭 (%)'
        }

    # Group by score
    score_groups = {
        'score_0': df[df['score'] == 0],
        'score_1': df[df['score'] == 1],
        'score_2': df[df['score'] == 2],
        'score_3_4': df[df['score'] >= 3]
    }

    result = {
        'task_type': task_type,
        'total_samples': len(df),
        'metrics': {},
        'display_names': display_names,
        'score_distribution': {
            'score_0': len(score_groups['score_0']),
            'score_1': len(score_groups['score_1']),
            'score_2': len(score_groups['score_2']),
            'score_3_4': len(score_groups['score_3_4'])
        }
    }

    # Compute statistics for each metric and score group
    for metric_key, col_name in metrics.items():
        if col_name not in df.columns:
            continue

        metric_stats = {}
        for group_name, group_df in score_groups.items():
            if len(group_df) > 0:
                values = group_df[col_name].dropna()
                if len(values) > 0:
                    metric_stats[group_name] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                        'count': len(values)
                    }

        if metric_stats:
            result['metrics'][metric_key] = metric_stats

    return result


def get_default_finger_tapping_stats():
    """Default statistics for finger tapping (fallback)"""
    return {
        'task_type': 'finger_tapping',
        'total_samples': 0,
        'metrics': {
            'tapping_freq': {
                'score_0': {'mean': 2.39, 'std': 0.58},
                'score_3_4': {'mean': 1.69, 'std': 0.52}
            },
            'amplitude_ratio': {
                'score_0': {'mean': 1.53, 'std': 0.31},
                'score_3_4': {'mean': 1.12, 'std': 0.24}
            },
            'amplitude_variability': {
                'score_0': {'mean': 16.72, 'std': 11.56},
                'score_3_4': {'mean': 20.85, 'std': 12.89}
            },
            'hesitation_rate': {
                'score_0': {'mean': 0.96, 'std': 1.05},
                'score_3_4': {'mean': 1.86, 'std': 1.63}
            },
            'fatigue_index': {
                'score_0': {'mean': 0.33, 'std': 0.49},
                'score_3_4': {'mean': 0.35, 'std': 0.44}
            },
            'velocity_variability': {
                'score_0': {'mean': 3.92, 'std': 4.19},
                'score_3_4': {'mean': 11.14, 'std': 11.33}
            }
        },
        'display_names': {
            'tapping_freq': '태핑 빈도 (Hz)',
            'amplitude_ratio': '진폭 비율',
            'amplitude_variability': '진폭 변동성 (%)',
            'hesitation_rate': '멈춤율 (%)',
            'fatigue_index': '속도저하 지수',
            'velocity_variability': '속도 변동성 (%)'
        }
    }


def get_default_gait_stats():
    """Default statistics for gait (fallback)"""
    return {
        'task_type': 'gait',
        'total_samples': 0,
        'metrics': {
            'walking_speed': {
                'score_0': {'mean': 1.1, 'std': 0.2},
                'score_3_4': {'mean': 0.7, 'std': 0.25}
            },
            'stride_length': {
                'score_0': {'mean': 1.3, 'std': 0.15},
                'score_3_4': {'mean': 0.9, 'std': 0.2}
            },
            'cadence': {
                'score_0': {'mean': 110, 'std': 10},
                'score_3_4': {'mean': 85, 'std': 15}
            }
        },
        'display_names': {
            'walking_speed': '보행 속도 (m/s)',
            'stride_length': '보폭 길이 (m)',
            'cadence': '보행률 (steps/min)'
        }
    }


# Load stats on module import
try:
    load_population_stats()
except Exception as e:
    print(f"[PopulationStats] Deferred loading due to: {e}")


@bp.route('/population-stats', methods=['GET'])
def get_all_stats():
    """Get all population statistics"""
    stats = load_population_stats()
    return jsonify({
        'success': True,
        'data': stats
    })


@bp.route('/population-stats/<task_type>', methods=['GET'])
def get_task_stats(task_type: str):
    """Get population statistics for a specific task type"""
    stats = load_population_stats()

    # Normalize task type
    task_type_normalized = task_type.lower().replace('-', '_')
    if task_type_normalized in ['finger', 'hand']:
        task_type_normalized = 'finger_tapping'

    if task_type_normalized not in stats:
        return jsonify({
            'success': False,
            'error': f'Unknown task type: {task_type}'
        }), 404

    return jsonify({
        'success': True,
        'data': stats[task_type_normalized]
    })
