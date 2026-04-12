"""
GroupKFold Cross-Validation for Gait and Finger Tapping
- Subject-level split: 같은 환자가 train/test에 중복되지 않음
- 전체 데이터(Train+Valid+Test) 사용
- 모든 Score가 포함된 신뢰할 수 있는 결과

Usage:
    python scripts/evaluate_groupkfold.py --task all
    python scripts/evaluate_groupkfold.py --task gait
    python scripts/evaluate_groupkfold.py --task finger
"""
import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 20
N_FOLDS = 5  # 30명 subject로 5-fold 가능


def extract_subject_from_id(video_id, task='finger'):
    """Extract subject ID from video filename

    Finger: '15-001908_l_011' -> '011'
    Gait: '15-001760_009' -> '009'
    """
    # Remove .mp4 extension if present
    video_id = video_id.replace('.mp4', '')
    # Get last segment after underscore
    subject = video_id.rsplit('_', 1)[-1]
    return subject


# ============================================================
# Dataset
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment and np.random.random() < 0.5:
            x = x + torch.randn_like(x) * 0.02
        return x, y


# ============================================================
# GAIT Models
# ============================================================
class SpatialTemporalLSTM(nn.Module):
    def __init__(self, input_size=129, hidden_size=128, raw_size=99, clinical_size=30, dropout=0.3):
        super().__init__()
        self.raw_size = raw_size
        self.clinical_size = clinical_size

        self.spatial_conv = nn.Sequential(
            nn.Conv1d(raw_size, hidden_size, 3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(hidden_size + clinical_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        raw = x[:, :, :self.raw_size].permute(0, 2, 1)
        clinical = x[:, :, self.raw_size:]

        spatial = self.spatial_conv(raw).permute(0, 2, 1)
        combined = torch.cat([spatial, clinical], dim=2)

        lstm_out, _ = self.lstm(combined)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.classifier(context).squeeze(-1)


# ============================================================
# FINGER Models
# ============================================================
class OrdinalLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        batch_size = targets.size(0)
        binary_targets = torch.zeros(batch_size, self.num_classes - 1, device=targets.device)
        for k in range(self.num_classes - 1):
            binary_targets[:, k] = (targets > k).float()
        return F.binary_cross_entropy_with_logits(logits, binary_targets)


class OrdinalModel(nn.Module):
    def __init__(self, input_size=98, hidden_size=128, dropout=0.4, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes - 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)

    def predict(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).sum(dim=1).float()
        return preds


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, is_ordinal=False):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, is_ordinal=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            if is_ordinal:
                pred = model.predict(X)
            else:
                pred = model(X)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    preds_rounded = np.clip(np.round(preds), 0, 4).astype(int)

    mae = mean_absolute_error(labels, preds)
    exact_acc = accuracy_score(labels, preds_rounded)
    within1 = np.mean(np.abs(preds_rounded - labels) <= 1)

    if len(np.unique(labels)) > 1:
        r, _ = pearsonr(preds, labels)
    else:
        r = 0

    return {
        'mae': mae,
        'exact_acc': exact_acc,
        'within1_acc': within1,
        'pearson_r': r,
        'preds': preds,
        'preds_rounded': preds_rounded,
        'labels': labels
    }


def run_groupkfold(X, y, groups, model_class, model_kwargs, criterion_class,
                   criterion_kwargs, device, task_name, is_ordinal=False):
    """Run GroupKFold CV with subject-level split"""
    print(f"\n{'='*60}")
    print(f"GroupKFold CV - {task_name}")
    print(f"Total samples: {len(X)}, Unique subjects: {len(np.unique(groups))}")
    print('='*60)

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    all_preds = []
    all_labels = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train = groups[train_idx]
        groups_val = groups[val_idx]

        print(f"    Train subjects: {len(np.unique(groups_train))}, Val subjects: {len(np.unique(groups_val))}")
        print(f"    Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"    Val labels: {dict(zip(*np.unique(y_val, return_counts=True)))}")

        # Check no overlap
        overlap = set(groups_train) & set(groups_val)
        if overlap:
            print(f"    WARNING: Subject overlap detected: {overlap}")

        # Weighted sampling for imbalanced classes
        class_counts = np.bincount(y_train.astype(int), minlength=5)
        sample_weights = 1.0 / (class_counts[y_train.astype(int)] + 1)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataset = TimeSeriesDataset(X_train, y_train, augment=True)
        val_dataset = TimeSeriesDataset(X_val, y_val, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Create model
        model = model_class(**model_kwargs).to(device)
        criterion = criterion_class(**criterion_kwargs)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_val_mae = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, is_ordinal)
            val_metrics = evaluate(model, val_loader, device, is_ordinal)
            scheduler.step()

            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_state:
            model.load_state_dict(best_state)

        # Final validation
        val_metrics = evaluate(model, val_loader, device, is_ordinal)
        fold_results.append(val_metrics)
        all_preds.extend(val_metrics['preds'])
        all_labels.extend(val_metrics['labels'])

        print(f"    Fold {fold+1}: MAE={val_metrics['mae']:.3f}, "
              f"Exact={val_metrics['exact_acc']*100:.1f}%, "
              f"Pearson={val_metrics['pearson_r']:.3f}")

    # Overall results
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds_rounded = np.clip(np.round(all_preds), 0, 4).astype(int)

    overall_mae = mean_absolute_error(all_labels, all_preds)
    overall_exact = accuracy_score(all_labels, all_preds_rounded)
    overall_within1 = np.mean(np.abs(all_preds_rounded - all_labels) <= 1)
    overall_pearson, _ = pearsonr(all_preds, all_labels)

    cv_results = {
        'cv_mae': overall_mae,
        'cv_exact_acc': overall_exact,
        'cv_within1_acc': overall_within1,
        'cv_pearson_r': overall_pearson,
        'fold_results': fold_results,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

    print(f"\n{task_name} GroupKFold CV Results:")
    print(f"  MAE: {overall_mae:.3f}")
    print(f"  Exact Accuracy: {overall_exact*100:.1f}%")
    print(f"  Within-1: {overall_within1*100:.1f}%")
    print(f"  Pearson r: {overall_pearson:.3f}")

    # Confusion Matrix
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds_rounded, labels=list(range(5)))
    print(f"  {'':>8} |   0    1    2    3    4")
    print(f"  {'-'*40}")
    for i in range(5):
        row = cm[i]
        if row.sum() > 0:
            print(f"  {i:>8} | {row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>4} {row[4]:>4}  (n={row.sum()})")

    # Per-class accuracy
    print(f"\n  Per-Class Accuracy:")
    for i in range(5):
        mask = all_labels == i
        if mask.sum() > 0:
            correct = (all_preds_rounded[mask] == i).sum()
            total = mask.sum()
            acc = correct / total * 100
            print(f"    Score {i}: {correct}/{total} = {acc:.1f}%")

    return cv_results


def evaluate_gait(device):
    print("\n" + "=" * 70)
    print("GAIT - GroupKFold CV with Subject-Level Split")
    print("=" * 70)

    # Load all data
    with open('data/gait_train_v2.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/gait_valid_v2.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open('data/gait_test_v2.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Combine all data
    X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
    y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

    # Extract subject IDs from video IDs
    # Format: '15-001760_009' -> subject '009'
    train_ids = train_data.get('ids', [f'train_{i}' for i in range(len(train_data['y']))])
    valid_ids = valid_data.get('ids', [f'valid_{i}' for i in range(len(valid_data['y']))])
    test_ids = test_data.get('ids', [f'test_{i}' for i in range(len(test_data['y']))])

    all_ids = list(train_ids) + list(valid_ids) + list(test_ids)
    subjects = np.array([extract_subject_from_id(vid, task='gait') for vid in all_ids])

    print(f"\nTotal: {X.shape}")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Unique subjects: {len(np.unique(subjects))}")

    results = run_groupkfold(
        X, y, subjects,
        SpatialTemporalLSTM,
        {'input_size': 129, 'hidden_size': 128, 'raw_size': 99, 'clinical_size': 30},
        nn.MSELoss,
        {},
        device,
        'Gait-SpatialTemporalLSTM',
        is_ordinal=False
    )

    return results


def evaluate_finger(device):
    print("\n" + "=" * 70)
    print("FINGER - GroupKFold CV with Subject-Level Split")
    print("=" * 70)

    # Load all data
    with open('data/finger_train_v3.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/finger_valid_v3.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open('data/finger_test_v3.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Combine all data
    X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
    y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

    # Extract subject IDs from video IDs
    # Format: '15-001908_l_011' -> subject '011'
    train_ids = train_data.get('ids', [f'train_{i}' for i in range(len(train_data['y']))])
    valid_ids = valid_data.get('ids', [f'valid_{i}' for i in range(len(valid_data['y']))])
    test_ids = test_data.get('ids', [f'test_{i}' for i in range(len(test_data['y']))])

    all_ids = list(train_ids) + list(valid_ids) + list(test_ids)
    subjects = np.array([extract_subject_from_id(vid, task='finger') for vid in all_ids])

    print(f"\nTotal: {X.shape}")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Unique subjects: {len(np.unique(subjects))}")

    results = run_groupkfold(
        X, y, subjects,
        OrdinalModel,
        {'input_size': 98, 'hidden_size': 128, 'dropout': 0.4},
        OrdinalLoss,
        {'num_classes': 5},
        device,
        'Finger-OrdinalModel',
        is_ordinal=True
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all', choices=['all', 'gait', 'finger'])
    args = parser.parse_args()

    print("=" * 70)
    print("GroupKFold CV - Subject-Level Split")
    print("No subject overlap between train and validation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    np.random.seed(42)
    torch.manual_seed(42)

    all_results = {}

    if args.task in ['all', 'gait']:
        gait_results = evaluate_gait(device)
        all_results['gait'] = gait_results

    if args.task in ['all', 'finger']:
        finger_results = evaluate_finger(device)
        all_results['finger'] = finger_results

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - GroupKFold CV Results")
    print("=" * 70)

    for task, results in all_results.items():
        print(f"\n{task.upper()}:")
        print(f"  MAE: {results['cv_mae']:.3f}")
        print(f"  Exact Accuracy: {results['cv_exact_acc']*100:.1f}%")
        print(f"  Within-1: {results['cv_within1_acc']*100:.1f}%")
        print(f"  Pearson r: {results['cv_pearson_r']:.3f}")

    # Save results
    with open('results_groupkfold_cv.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: results_groupkfold_cv.pkl")


if __name__ == "__main__":
    main()
