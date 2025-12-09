"""
Fix Data Leakage with Stratified Subject-Level Split
Ensures score distribution is preserved across splits

Usage:
    python scripts/fix_data_split_stratified.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import PD4T_ROOT, FEATURES_DIR, ensure_dirs


def extract_patient_id(video_id: str) -> str:
    """Extract patient ID from video_id"""
    match = re.match(r'\d+-(\d+)', str(video_id))
    if match:
        return match.group(1)
    return str(video_id)


def stratified_patient_split(df: pd.DataFrame,
                            train_ratio: float = 0.7,
                            valid_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            random_state: int = 42) -> dict:
    """
    Split patients while trying to maintain score distribution.

    Strategy:
    1. Group patients by their dominant score
    2. Split each score group proportionally
    3. Ensure rare scores are distributed across all splits
    """
    np.random.seed(random_state)

    df = df.copy()
    df['patient_id'] = df['video_id'].apply(extract_patient_id)

    # Get dominant score for each patient (mode or mean rounded)
    patient_scores = df.groupby('patient_id')['score'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else round(x.mean())
    )

    # Group patients by their dominant score
    score_to_patients = defaultdict(list)
    for patient, score in patient_scores.items():
        score_to_patients[score].append(patient)

    train_patients = []
    valid_patients = []
    test_patients = []

    # Split each score group
    for score in sorted(score_to_patients.keys()):
        patients = score_to_patients[score]
        np.random.shuffle(patients)

        n = len(patients)

        if n >= 3:
            # Normal split
            n_train = max(1, int(n * train_ratio))
            n_valid = max(1, int(n * valid_ratio))
            n_test = n - n_train - n_valid

            # Ensure at least 1 in test for rare classes
            if n_test == 0 and n >= 3:
                n_test = 1
                n_train -= 1
        elif n == 2:
            # Put 1 in train, 1 in test
            n_train, n_valid, n_test = 1, 0, 1
        else:
            # Only 1 patient - put in train
            n_train, n_valid, n_test = 1, 0, 0

        train_patients.extend(patients[:n_train])
        valid_patients.extend(patients[n_train:n_train+n_valid])
        test_patients.extend(patients[n_train+n_valid:])

        print(f"  Score {score}: {n} patients -> train:{n_train}, valid:{n_valid}, test:{n_test}")

    return {
        'train': set(train_patients),
        'valid': set(valid_patients),
        'test': set(test_patients)
    }


def fix_split_stratified(task: str, random_state: int = 42):
    """Create stratified subject-level split"""

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

    combined = pd.concat(all_data, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"Stratified Split for: {task}")
    print(f"{'='*60}")
    print(f"Total videos: {len(combined)}")
    print(f"Score distribution: {combined['score'].value_counts().sort_index().to_dict()}")

    # Get stratified patient split
    print("\nSplitting by score groups:")
    patient_splits = stratified_patient_split(combined, random_state=random_state)

    # Assign videos to splits
    combined['patient_id'] = combined['video_id'].apply(extract_patient_id)
    combined['split'] = combined['patient_id'].apply(
        lambda x: 'train' if x in patient_splits['train']
                  else ('valid' if x in patient_splits['valid'] else 'test')
    )

    # Create split DataFrames
    train_df = combined[combined['split'] == 'train'][['video_id', 'frames', 'score']]
    valid_df = combined[combined['split'] == 'valid'][['video_id', 'frames', 'score']]
    test_df = combined[combined['split'] == 'test'][['video_id', 'frames', 'score']]

    # Print results
    print(f"\nFinal split:")
    for name, df in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
        patients = df['video_id'].apply(extract_patient_id).nunique()
        dist = df['score'].value_counts().sort_index().to_dict()
        print(f"  {name}: {len(df)} videos, {patients} patients")
        print(f"         Scores: {dist}")

    # Verify no overlap
    train_set = set(train_df['video_id'].apply(extract_patient_id))
    valid_set = set(valid_df['video_id'].apply(extract_patient_id))
    test_set = set(test_df['video_id'].apply(extract_patient_id))

    assert len(train_set & valid_set) == 0, "Train-Valid overlap!"
    assert len(train_set & test_set) == 0, "Train-Test overlap!"
    assert len(valid_set & test_set) == 0, "Valid-Test overlap!"
    print("\n[OK] No patient overlap")

    # Save
    output_dir = task_dir / "stratified"
    output_dir.mkdir(exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False, header=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False, header=False)
    test_df.to_csv(output_dir / "test.csv", index=False, header=False)

    print(f"Saved to: {output_dir}/")

    return patient_splits


def fix_features_stratified(task: str, patient_splits: dict):
    """Apply stratified split to features"""

    task_name = task.lower().replace(' ', '_')

    all_features = []
    for split in ['train', 'valid', 'test']:
        # Try both original and corrected
        for suffix in ['_features.csv', '_features_corrected.csv']:
            filepath = FEATURES_DIR / f"{task_name}_{split}{suffix}"
            if filepath.exists():
                df = pd.read_csv(filepath)
                all_features.append(df)
                break

    if not all_features:
        print(f"[INFO] No features for {task}")
        return

    combined = pd.concat(all_features, ignore_index=True).drop_duplicates(subset=['video_id'])
    combined['patient_id'] = combined['video_id'].apply(extract_patient_id)

    # Assign splits
    combined['new_split'] = combined['patient_id'].apply(
        lambda x: 'train' if x in patient_splits['train']
                  else ('valid' if x in patient_splits['valid']
                        else ('test' if x in patient_splits['test'] else 'train'))
    )

    cols = [c for c in combined.columns if c not in ['patient_id', 'new_split']]

    print(f"\nFeatures for {task}:")
    for split in ['train', 'valid', 'test']:
        split_df = combined[combined['new_split'] == split][cols]
        output_path = FEATURES_DIR / f"{task_name}_{split}_features_stratified.csv"
        split_df.to_csv(output_path, index=False)

        dist = split_df['score'].value_counts().sort_index().to_dict()
        print(f"  {split}: {len(split_df)} -> {output_path.name}")
        print(f"         Scores: {dist}")


def main():
    tasks = ['Finger Tapping', 'Gait', 'Hand Movement', 'Leg Agility']

    for task in tasks:
        splits = fix_split_stratified(task)
        if splits:
            fix_features_stratified(task, splits)

    print("\n" + "="*60)
    print("DONE - Use *_stratified.csv for proper evaluation")
    print("="*60)


if __name__ == "__main__":
    main()
