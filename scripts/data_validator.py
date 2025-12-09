"""
Data Validation Script for PD4T/TULIP Datasets
Automatically checks for data leakage and split integrity

Usage:
    python scripts/data_validator.py
    python scripts/data_validator.py --task "Finger Tapping"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys
import argparse

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from config import PD4T_ROOT, FEATURES_DIR


class DataValidator:
    """Validate PD4T/TULIP dataset splits for data leakage"""

    def __init__(self):
        self.pd4t_root = PD4T_ROOT
        self.features_dir = FEATURES_DIR
        self.errors = []
        self.warnings = []

    def extract_patient_id(self, video_id: str) -> str:
        """Extract patient ID from video_id

        Formats:
        - PD4T: '13-002356_r_003' -> patient_id = '002356'
        - TULIP: varies by task
        """
        # PD4T format: XX-NNNNNN_[lr]_TTT
        match = re.match(r'\d+-(\d+)_[lr]', str(video_id))
        if match:
            return match.group(1)

        # Alternative format: XX-NNNNNN_[lr]
        match = re.match(r'\d+-(\d+)', str(video_id))
        if match:
            return match.group(1)

        return str(video_id)

    def load_annotations(self, task: str) -> dict:
        """Load train/valid/test annotations for a task"""
        task_dir = self.pd4t_root / "Annotations" / task

        splits = {}
        for split in ['train', 'valid', 'test']:
            filepath = task_dir / f"{split}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath, header=None,
                               names=['video_id', 'frames', 'score'])
                splits[split] = df

        return splits

    def load_features(self, task: str) -> dict:
        """Load feature files for a task"""
        task_name = task.lower().replace(' ', '_')

        splits = {}
        for split in ['train', 'valid', 'test']:
            filepath = self.features_dir / f"{task_name}_{split}_features.csv"
            if filepath.exists():
                splits[split] = pd.read_csv(filepath)

        return splits

    def check_patient_overlap(self, splits: dict, source: str) -> bool:
        """Check for patient-level overlap between splits"""
        patient_sets = {}

        for split_name, df in splits.items():
            if 'video_id' in df.columns:
                patients = set(df['video_id'].apply(self.extract_patient_id))
                patient_sets[split_name] = patients

        has_leakage = False
        split_names = list(patient_sets.keys())

        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                overlap = patient_sets[split1] & patient_sets[split2]
                if overlap:
                    has_leakage = True
                    if 'test' in [split1, split2]:
                        self.errors.append(
                            f"[CRITICAL] {source}: {split1}-{split2} overlap: "
                            f"{len(overlap)} patients ({list(overlap)[:3]}...)"
                        )
                    else:
                        self.warnings.append(
                            f"[WARNING] {source}: {split1}-{split2} overlap: "
                            f"{len(overlap)} patients"
                        )

        return not has_leakage

    def check_score_distribution(self, splits: dict, source: str) -> bool:
        """Check for reasonable score distribution across splits"""
        for split_name, df in splits.items():
            if 'score' in df.columns:
                dist = df['score'].value_counts(normalize=True)

                # Check if any class is missing
                expected_classes = set(range(5))  # 0-4 for UPDRS
                actual_classes = set(df['score'].unique())
                missing = expected_classes - actual_classes

                if missing and split_name == 'test':
                    self.warnings.append(
                        f"[INFO] {source} {split_name}: Missing scores {missing}"
                    )

        return True

    def check_sample_counts(self, splits: dict, source: str) -> bool:
        """Check for reasonable sample counts"""
        total = sum(len(df) for df in splits.values())

        for split_name, df in splits.items():
            ratio = len(df) / total if total > 0 else 0

            if split_name == 'train' and ratio < 0.5:
                self.warnings.append(
                    f"[WARNING] {source}: Train set is only {ratio:.1%} of total"
                )

            if split_name == 'test' and ratio < 0.1:
                self.warnings.append(
                    f"[WARNING] {source}: Test set is only {ratio:.1%} of total"
                )

        return True

    def validate_task(self, task: str) -> dict:
        """Run all validations for a task"""
        print(f"\n{'='*60}")
        print(f"Validating: {task}")
        print('='*60)

        results = {
            'task': task,
            'annotations_ok': False,
            'features_ok': False,
            'no_leakage': False
        }

        # Check annotations
        ann_splits = self.load_annotations(task)
        if ann_splits:
            print(f"\nAnnotations: {list(ann_splits.keys())}")
            for name, df in ann_splits.items():
                patients = df['video_id'].apply(self.extract_patient_id).nunique()
                print(f"  {name}: {len(df)} videos, {patients} patients")

            results['annotations_ok'] = self.check_patient_overlap(
                ann_splits, f"Annotations/{task}"
            )
        else:
            self.warnings.append(f"[INFO] No annotations found for {task}")

        # Check features
        feat_splits = self.load_features(task)
        if feat_splits:
            print(f"\nFeatures: {list(feat_splits.keys())}")
            for name, df in feat_splits.items():
                patients = df['video_id'].apply(self.extract_patient_id).nunique()
                print(f"  {name}: {len(df)} samples, {patients} patients")

            results['features_ok'] = self.check_patient_overlap(
                feat_splits, f"Features/{task}"
            )
            self.check_score_distribution(feat_splits, f"Features/{task}")
            self.check_sample_counts(feat_splits, f"Features/{task}")
        else:
            self.warnings.append(f"[INFO] No features found for {task}")

        results['no_leakage'] = results['annotations_ok'] and results['features_ok']

        return results

    def validate_all(self) -> dict:
        """Validate all tasks"""
        tasks = ['Finger Tapping', 'Gait', 'Hand Movement', 'Leg Agility']

        all_results = {}
        for task in tasks:
            all_results[task] = self.validate_task(task)

        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        if self.errors:
            print("\n[ERRORS]")
            for err in self.errors:
                print(f"  {err}")

        if self.warnings:
            print("\n[WARNINGS]")
            for warn in self.warnings:
                print(f"  {warn}")

        if not self.errors and not self.warnings:
            print("\n[OK] All validations passed!")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Validate PD4T/TULIP data splits")
    parser.add_argument('--task', type=str, help="Specific task to validate")
    args = parser.parse_args()

    validator = DataValidator()

    if args.task:
        validator.validate_task(args.task)
    else:
        validator.validate_all()

    # Return exit code based on errors
    return 1 if validator.errors else 0


if __name__ == "__main__":
    sys.exit(main())
