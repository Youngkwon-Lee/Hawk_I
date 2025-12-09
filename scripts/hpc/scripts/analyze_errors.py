"""
Error Analysis for UPDRS Scoring Models
Analyze misclassified cases from results pkl files

Usage:
    python scripts/analyze_errors.py --task gait
    python scripts/analyze_errors.py --task finger
"""
import pickle
import numpy as np
import argparse
from collections import defaultdict

def analyze_errors(results_file, data_file, task):
    print("=" * 60)
    print(f"ERROR ANALYSIS - {task.upper()}")
    print("=" * 60)

    # Load results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Load data for video IDs
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    # Get best model (SpatialTemporalLSTM)
    best_model = 'SpatialTemporalLSTM'
    if best_model not in results:
        best_model = list(results.keys())[0]

    print(f"\nAnalyzing: {best_model}")
    model_results = results[best_model]

    # Aggregate all folds
    all_preds = []
    all_labels = []

    for fold_idx, fold in enumerate(model_results['fold_results']):
        preds = fold['preds']
        labels = fold['labels']
        all_preds.extend(preds)
        all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds_rounded = np.clip(np.round(all_preds), 0, 4).astype(int)

    # Error analysis
    errors = all_preds_rounded != all_labels
    error_indices = np.where(errors)[0]

    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Correct: {len(all_labels) - len(error_indices)} ({(1-len(error_indices)/len(all_labels))*100:.1f}%)")
    print(f"Errors: {len(error_indices)} ({len(error_indices)/len(all_labels)*100:.1f}%)")

    # Confusion Matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"\n{'':>10} | Predicted")
    print(f"{'Actual':>10} |   0    1    2    3    4")
    print("-" * 45)

    confusion = np.zeros((5, 5), dtype=int)
    for true, pred in zip(all_labels, all_preds_rounded):
        confusion[int(true), int(pred)] += 1

    for i in range(5):
        row = confusion[i]
        total = row.sum()
        if total > 0:
            print(f"{i:>10} | {row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>4} {row[4]:>4}  (n={total})")

    # Per-class accuracy
    print("\n" + "=" * 60)
    print("PER-CLASS ACCURACY")
    print("=" * 60)

    for score in range(5):
        mask = all_labels == score
        if mask.sum() > 0:
            correct = (all_preds_rounded[mask] == score).sum()
            total = mask.sum()
            acc = correct / total * 100
            print(f"Score {score}: {correct}/{total} = {acc:.1f}%")

    # Error distribution
    print("\n" + "=" * 60)
    print("ERROR DISTRIBUTION")
    print("=" * 60)

    error_dist = defaultdict(int)
    for true, pred in zip(all_labels[errors], all_preds_rounded[errors]):
        error_dist[f"{int(true)}->{int(pred)}"] += 1

    sorted_errors = sorted(error_dist.items(), key=lambda x: -x[1])
    print("\nMost common errors:")
    for err_type, count in sorted_errors[:10]:
        print(f"  {err_type}: {count} cases")

    # Error magnitude
    print("\n" + "=" * 60)
    print("ERROR MAGNITUDE")
    print("=" * 60)

    error_magnitudes = np.abs(all_preds_rounded - all_labels)
    for mag in range(5):
        count = (error_magnitudes == mag).sum()
        pct = count / len(all_labels) * 100
        if count > 0:
            print(f"Off by {mag}: {count} ({pct:.1f}%)")

    # Sample errors with video IDs (if available)
    if 'ids' in data:
        print("\n" + "=" * 60)
        print("SAMPLE ERROR CASES")
        print("=" * 60)

        ids = data['ids']
        # Note: IDs might not match exactly due to CV splits
        # This shows a sample of error patterns

        print("\nLarge errors (off by 2+):")
        large_errors = np.where(error_magnitudes >= 2)[0]
        for idx in large_errors[:10]:
            true = int(all_labels[idx])
            pred = int(all_preds_rounded[idx])
            raw = all_preds[idx]
            print(f"  True={true}, Pred={pred} (raw={raw:.2f})")

    # Save detailed results
    output_file = f'error_analysis_{task}.txt'
    with open(output_file, 'w') as f:
        f.write(f"Error Analysis - {task.upper()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"{'':>10} |   0    1    2    3    4\n")
        for i in range(5):
            row = confusion[i]
            f.write(f"{i:>10} | {row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>4} {row[4]:>4}\n")

        f.write("\nPer-class accuracy:\n")
        for score in range(5):
            mask = all_labels == score
            if mask.sum() > 0:
                correct = (all_preds_rounded[mask] == score).sum()
                total = mask.sum()
                acc = correct / total * 100
                f.write(f"Score {score}: {correct}/{total} = {acc:.1f}%\n")

        f.write("\nAll predictions:\n")
        f.write("idx,true,pred,raw_pred,error\n")
        for i in range(len(all_labels)):
            true = int(all_labels[i])
            pred = int(all_preds_rounded[i])
            raw = all_preds[i]
            err = abs(true - pred)
            f.write(f"{i},{true},{pred},{raw:.3f},{err}\n")

    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='gait', choices=['finger', 'gait'])
    args = parser.parse_args()

    if args.task == 'gait':
        results_file = 'results_gait_v2.pkl'
        data_file = 'data/gait_train_v2.pkl'
    else:
        results_file = 'results_finger_v2.pkl'
        data_file = 'data/finger_train_v2.pkl'

    analyze_errors(results_file, data_file, args.task)


if __name__ == "__main__":
    main()
