"""
Fix Numerical Overflow in Feature Files

Recalculates percentage-based features using safe_divide and clip_percentage
to eliminate overflow values like 9.36e+16.
"""
import pandas as pd
import numpy as np
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

# Constants (same as metrics_calculator.py)
EPSILON = 1e-8
PERCENTAGE_CLIP_MIN = -500.0
PERCENTAGE_CLIP_MAX = 500.0


def safe_divide(numerator, denominator, default=0.0):
    """Safe division with epsilon to prevent overflow"""
    if isinstance(denominator, (pd.Series, np.ndarray)):
        result = np.where(
            np.abs(denominator) < EPSILON,
            default,
            numerator / denominator
        )
        return result
    else:
        if abs(denominator) < EPSILON:
            return default
        return numerator / denominator


def clip_percentage(value):
    """Clip percentage values to reasonable range"""
    return np.clip(value, PERCENTAGE_CLIP_MIN, PERCENTAGE_CLIP_MAX)


def fix_finger_tapping_features(df):
    """Fix overflow in finger tapping features"""
    print("\n  Fixing Finger Tapping features...")

    # 1. variability_change = (second - first) / first * 100
    if 'variability_first_half' in df.columns and 'variability_second_half' in df.columns:
        old_max = df['variability_change'].max()

        diff = df['variability_second_half'] - df['variability_first_half']
        df['variability_change'] = clip_percentage(
            safe_divide(diff, df['variability_first_half'], 0.0) * 100
        )

        new_max = df['variability_change'].max()
        print(f"    variability_change: max {old_max:.2e} → {new_max:.2f}")

    # 2. velocity_decrement = (first_third - last_third) / first_third * 100
    if 'velocity_first_third' in df.columns and 'velocity_last_third' in df.columns:
        old_max = df['velocity_decrement'].max() if 'velocity_decrement' in df.columns else 0

        diff = df['velocity_first_third'] - df['velocity_last_third']
        df['velocity_decrement'] = clip_percentage(
            safe_divide(diff, df['velocity_first_third'], 0.0) * 100
        )

        new_max = df['velocity_decrement'].max()
        print(f"    velocity_decrement: max {old_max:.2e} → {new_max:.2f}")

    # 3. amplitude_decrement = (first_third - last_third) / first_third * 100
    if 'amplitude_first_third' in df.columns and 'amplitude_last_third' in df.columns:
        old_max = df['amplitude_decrement'].max() if 'amplitude_decrement' in df.columns else 0

        diff = df['amplitude_first_third'] - df['amplitude_last_third']
        df['amplitude_decrement'] = clip_percentage(
            safe_divide(diff, df['amplitude_first_third'], 0.0) * 100
        )

        new_max = df['amplitude_decrement'].max()
        print(f"    amplitude_decrement: max {old_max:.2e} → {new_max:.2f}")

    # 4. velocity_slope and amplitude_slope - these need tap-by-tap data, just clip them
    for col in ['velocity_slope', 'amplitude_slope', 'rhythm_slope', 'fatigue_rate']:
        if col in df.columns:
            old_max = df[col].max()
            df[col] = clip_percentage(df[col])
            new_max = df[col].max()
            if old_max != new_max:
                print(f"    {col}: max {old_max:.2e} → {new_max:.2f}")

    return df


def fix_gait_features(df):
    """Fix overflow in gait features"""
    print("\n  Fixing Gait features...")

    # Asymmetry features (already use safe pattern but may have old values)
    asymmetry_cols = ['step_length_asymmetry', 'swing_time_asymmetry', 'arm_swing_asymmetry']
    for col in asymmetry_cols:
        if col in df.columns:
            old_max = df[col].max()
            df[col] = clip_percentage(df[col])
            new_max = df[col].max()
            if old_max != new_max:
                print(f"    {col}: max {old_max:.2e} → {new_max:.2f}")

    # Trend features
    trend_cols = ['step_length_trend', 'cadence_trend', 'arm_swing_trend',
                  'variability_trend', 'step_height_trend']
    for col in trend_cols:
        if col in df.columns:
            old_max = df[col].max()
            df[col] = clip_percentage(df[col])
            new_max = df[col].max()
            if old_max != new_max:
                print(f"    {col}: max {old_max:.2e} → {new_max:.2f}")

    return df


def process_feature_files(features_dir, task_type):
    """Process all feature files for a task type"""
    print(f"\n{'='*60}")
    print(f"Processing {task_type} features")
    print(f"{'='*60}")

    for split in ['train', 'valid', 'test']:
        # Process stratified files
        filename = f"{task_type}_{split}_features_stratified.csv"
        filepath = os.path.join(features_dir, filename)

        if not os.path.exists(filepath):
            print(f"\n  {filename}: NOT FOUND")
            continue

        print(f"\n  Processing: {filename}")
        df = pd.read_csv(filepath)

        # Check for outliers before
        outlier_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if df[c].abs().max() > 500 and c not in ['video_id', 'subject', 'score', 'frames', 'duration', 'total_taps']]
        if outlier_cols:
            print(f"    Outlier columns: {outlier_cols}")

        # Apply fixes based on task type
        if task_type == 'finger_tapping':
            df = fix_finger_tapping_features(df)
        elif task_type == 'gait':
            df = fix_gait_features(df)

        # Save fixed file
        df.to_csv(filepath, index=False)
        print(f"    Saved: {filepath}")

        # Verify no more overflow
        remaining_outliers = [c for c in df.select_dtypes(include=[np.number]).columns
                             if df[c].abs().max() > 500 and c not in ['video_id', 'subject', 'score', 'frames', 'duration', 'total_taps']]
        if remaining_outliers:
            print(f"    WARNING: Remaining outliers in {remaining_outliers}")
        else:
            print(f"    [OK] No outliers remaining")


def main():
    print("="*60)
    print("Feature Overflow Fix Script")
    print("="*60)

    # Get features directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    features_dir = os.path.join(project_root, 'data', 'processed', 'features')

    print(f"\nFeatures directory: {features_dir}")

    # Process both task types
    process_feature_files(features_dir, 'finger_tapping')
    process_feature_files(features_dir, 'gait')

    # Final verification
    print("\n" + "="*60)
    print("FINAL VERIFICATION")
    print("="*60)

    for task in ['finger_tapping', 'gait']:
        for split in ['train', 'valid', 'test']:
            filepath = os.path.join(features_dir, f"{task}_{split}_features_stratified.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                max_vals = df[numeric_cols].abs().max()
                extreme = max_vals[max_vals > 500]
                extreme = extreme[~extreme.index.isin(['duration', 'total_taps', 'step_count'])]

                if len(extreme) > 0:
                    print(f"  [WARN] {task}_{split}: {len(extreme)} columns with values > 500")
                else:
                    print(f"  [OK] {task}_{split}: OK")


if __name__ == "__main__":
    main()
