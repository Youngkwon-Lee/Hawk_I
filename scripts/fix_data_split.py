"""
Fix Data Leakage in PD4T Dataset
Creates proper subject-level train/valid/test splits

Usage:
    python scripts/fix_data_split.py
    python scripts/fix_data_split.py --task "Finger Tapping"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from config import PD4T_ROOT, FEATURES_DIR, ensure_dirs


def extract_patient_id(video_id: str) -> str:
    """Extract patient ID from video_id"""
    match = re.match(r'\d+-(\d+)', str(video_id))
    if match:
        return match.group(1)
    return str(video_id)


def fix_annotation_split(task: str, train_ratio: float = 0.7,
                        valid_ratio: float = 0.15, test_ratio: float = 0.15,
                        random_state: int = 42):
    """Create proper subject-level split for annotations"""

    task_dir = PD4T_ROOT / "Annotations" / task

    # Load all data
    all_data = []
    for split in ['train', 'test', 'valid']:
        filepath = task_dir / f"{split}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath, header=None,
                           names=['video_id', 'frames', 'score'])
            all_data.append(df)

    if not all_data:
        print(f"[ERROR] No data found for {task}")
        return None

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined['patient_id'] = combined['video_id'].apply(extract_patient_id)

    print(f"\n{'='*60}")
    print(f"Fixing split for: {task}")
    print(f"{'='*60}")
    print(f"Total videos: {len(combined)}")
    print(f"Unique patients: {combined['patient_id'].nunique()}")

    # Get unique patients with their score distribution
    patient_scores = combined.groupby('patient_id')['score'].agg(['mean', 'count'])

    # Split patients (not videos) into train/valid/test
    patients = patient_scores.index.tolist()

    # First split: train vs (valid+test)
    train_patients, temp_patients = train_test_split(
        patients,
        test_size=(valid_ratio + test_ratio),
        random_state=random_state
    )

    # Second split: valid vs test
    valid_patients, test_patients = train_test_split(
        temp_patients,
        test_size=test_ratio / (valid_ratio + test_ratio),
        random_state=random_state
    )

    # Assign videos to splits based on patient
    combined['split'] = combined['patient_id'].apply(
        lambda x: 'train' if x in train_patients
                  else ('valid' if x in valid_patients else 'test')
    )

    # Create new split DataFrames
    train_df = combined[combined['split'] == 'train'][['video_id', 'frames', 'score']]
    valid_df = combined[combined['split'] == 'valid'][['video_id', 'frames', 'score']]
    test_df = combined[combined['split'] == 'test'][['video_id', 'frames', 'score']]

    # Print statistics
    print(f"\nNew split (subject-level, no leakage):")
    print(f"  Train: {len(train_df)} videos, {len(train_patients)} patients")
    print(f"  Valid: {len(valid_df)} videos, {len(valid_patients)} patients")
    print(f"  Test: {len(test_df)} videos, {len(test_patients)} patients")

    # Verify no overlap
    train_set = set(train_df['video_id'].apply(extract_patient_id))
    valid_set = set(valid_df['video_id'].apply(extract_patient_id))
    test_set = set(test_df['video_id'].apply(extract_patient_id))

    assert len(train_set & valid_set) == 0, "Train-Valid overlap!"
    assert len(train_set & test_set) == 0, "Train-Test overlap!"
    assert len(valid_set & test_set) == 0, "Valid-Test overlap!"
    print("[OK] No patient overlap between splits")

    # Save corrected splits
    output_dir = task_dir / "corrected"
    output_dir.mkdir(exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False, header=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False, header=False)
    test_df.to_csv(output_dir / "test.csv", index=False, header=False)

    print(f"\nSaved to: {output_dir}/")

    return {
        'train': train_df,
        'valid': valid_df,
        'test': test_df,
        'train_patients': train_patients,
        'valid_patients': valid_patients,
        'test_patients': test_patients
    }


def fix_feature_split(task: str, patient_splits: dict):
    """Apply patient-level split to feature files"""

    task_name = task.lower().replace(' ', '_')

    # Load all feature data
    all_features = []
    for split in ['train', 'valid', 'test']:
        filepath = FEATURES_DIR / f"{task_name}_{split}_features.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            all_features.append(df)

    if not all_features:
        print(f"[INFO] No features found for {task}")
        return None

    combined = pd.concat(all_features, ignore_index=True)
    combined['patient_id'] = combined['video_id'].apply(extract_patient_id)

    print(f"\nFixing features for: {task}")
    print(f"Total samples: {len(combined)}")

    train_patients = set(patient_splits['train_patients'])
    valid_patients = set(patient_splits['valid_patients'])
    test_patients = set(patient_splits['test_patients'])

    # Assign to splits
    combined['new_split'] = combined['patient_id'].apply(
        lambda x: 'train' if x in train_patients
                  else ('valid' if x in valid_patients
                        else ('test' if x in test_patients else 'unknown'))
    )

    # Handle unknown patients (not in annotation splits)
    unknown = combined[combined['new_split'] == 'unknown']
    if len(unknown) > 0:
        print(f"[WARNING] {len(unknown)} samples from unknown patients")
        # Assign to train by default
        combined.loc[combined['new_split'] == 'unknown', 'new_split'] = 'train'

    # Create new splits
    cols_to_save = [c for c in combined.columns if c not in ['patient_id', 'new_split']]

    for split in ['train', 'valid', 'test']:
        split_df = combined[combined['new_split'] == split][cols_to_save]
        output_path = FEATURES_DIR / f"{task_name}_{split}_features_corrected.csv"
        split_df.to_csv(output_path, index=False)
        patients = split_df['video_id'].apply(extract_patient_id).nunique()
        print(f"  {split}: {len(split_df)} samples, {patients} patients -> {output_path.name}")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix data leakage in PD4T")
    parser.add_argument('--task', type=str, help="Specific task to fix")
    args = parser.parse_args()

    tasks = ['Finger Tapping', 'Gait', 'Hand Movement', 'Leg Agility']

    if args.task:
        tasks = [args.task]

    for task in tasks:
        splits = fix_annotation_split(task)
        if splits:
            fix_feature_split(task, splits)

    print("\n" + "="*60)
    print("DONE - Use *_corrected.csv files for proper evaluation")
    print("="*60)


if __name__ == "__main__":
    main()
