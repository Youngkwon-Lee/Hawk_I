"""
GroupKFold Cross-Validation - Multiple Models
Subject-level split으로 다양한 모델 비교

Usage:
    python scripts/evaluate_groupkfold_multi.py --task gait
    python scripts/evaluate_groupkfold_multi.py --task finger
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
N_FOLDS = 5


def extract_subject_from_id(video_id, task='finger'):
    """Extract subject ID from video filename"""
    video_id = video_id.replace('.mp4', '')
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


class DilatedCNN(nn.Module):
    def __init__(self, input_size=129, hidden_size=128, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, padding=4, dilation=4)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)
        return self.fc(x).squeeze(-1)


class TCN(nn.Module):
    def __init__(self, input_size=129, hidden_size=128, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, 3, padding=8, dilation=8)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.mean(dim=2)
        return self.fc(x).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, input_size=129, hidden_size=128, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


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


class FingerDilatedCNN(nn.Module):
    """DilatedCNN for Finger with Ordinal output"""
    def __init__(self, input_size=98, hidden_size=128, dropout=0.4, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, padding=4, dilation=4)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=2)
        return self.fc(x)

    def predict(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).sum(dim=1).float()
        return preds


class FingerTransformer(nn.Module):
    """Transformer for Finger with Ordinal output"""
    def __init__(self, input_size=98, hidden_size=128, nhead=4, num_layers=2, dropout=0.4, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

    def predict(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).sum(dim=1).float()
        return preds


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
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
        'labels': labels
    }


def run_groupkfold(X, y, groups, model_class, model_kwargs, criterion_class,
                   criterion_kwargs, device, task_name, is_ordinal=False):
    print(f"\n{'='*60}")
    print(f"GroupKFold CV - {task_name}")
    print(f"Total samples: {len(X)}, Unique subjects: {len(np.unique(groups))}")
    print('='*60)

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_preds = []
    all_labels = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Weighted sampling
        class_counts = np.bincount(y_train.astype(int), minlength=5)
        sample_weights = 1.0 / (class_counts[y_train.astype(int)] + 1)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataset = TimeSeriesDataset(X_train, y_train, augment=True)
        val_dataset = TimeSeriesDataset(X_val, y_val, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = model_class(**model_kwargs).to(device)
        criterion = criterion_class(**criterion_kwargs)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_val_mae = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, device, is_ordinal)
            scheduler.step()

            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

        if best_state:
            model.load_state_dict(best_state)

        val_metrics = evaluate(model, val_loader, device, is_ordinal)
        all_preds.extend(val_metrics['preds'])
        all_labels.extend(val_metrics['labels'])
        print(f"  Fold {fold+1}: MAE={val_metrics['mae']:.3f}, Acc={val_metrics['exact_acc']*100:.1f}%")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_preds_rounded = np.clip(np.round(all_preds), 0, 4).astype(int)

    overall_mae = mean_absolute_error(all_labels, all_preds)
    overall_exact = accuracy_score(all_labels, all_preds_rounded)
    overall_within1 = np.mean(np.abs(all_preds_rounded - all_labels) <= 1)
    overall_pearson, _ = pearsonr(all_preds, all_labels)

    print(f"\n{task_name} Results:")
    print(f"  MAE: {overall_mae:.3f}")
    print(f"  Exact Accuracy: {overall_exact*100:.1f}%")
    print(f"  Within-1: {overall_within1*100:.1f}%")
    print(f"  Pearson r: {overall_pearson:.3f}")

    return {
        'mae': overall_mae,
        'exact_acc': overall_exact,
        'within1_acc': overall_within1,
        'pearson_r': overall_pearson
    }


def evaluate_gait_models(device):
    print("\n" + "=" * 70)
    print("GAIT - Multi-Model GroupKFold CV")
    print("=" * 70)

    # Load data
    with open('data/gait_train_v2.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/gait_valid_v2.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open('data/gait_test_v2.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
    y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

    train_ids = train_data.get('ids', [f'train_{i}' for i in range(len(train_data['y']))])
    valid_ids = valid_data.get('ids', [f'valid_{i}' for i in range(len(valid_data['y']))])
    test_ids = test_data.get('ids', [f'test_{i}' for i in range(len(test_data['y']))])
    all_ids = list(train_ids) + list(valid_ids) + list(test_ids)
    subjects = np.array([extract_subject_from_id(vid, task='gait') for vid in all_ids])

    print(f"\nTotal: {X.shape}, Unique subjects: {len(np.unique(subjects))}")

    models = [
        ('SpatialTemporalLSTM', SpatialTemporalLSTM,
         {'input_size': 129, 'hidden_size': 128, 'raw_size': 99, 'clinical_size': 30}),
        ('DilatedCNN', DilatedCNN, {'input_size': 129, 'hidden_size': 128}),
        ('TCN', TCN, {'input_size': 129, 'hidden_size': 128}),
        ('Transformer', TransformerModel, {'input_size': 129, 'hidden_size': 128}),
    ]

    results = {}
    for name, model_class, model_kwargs in models:
        result = run_groupkfold(
            X, y, subjects, model_class, model_kwargs,
            nn.MSELoss, {}, device, f'Gait-{name}', is_ordinal=False
        )
        results[name] = result

    return results


def evaluate_finger_models(device):
    print("\n" + "=" * 70)
    print("FINGER - Multi-Model GroupKFold CV")
    print("=" * 70)

    # Load data
    with open('data/finger_train_v3.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/finger_valid_v3.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open('data/finger_test_v3.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
    y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

    train_ids = train_data.get('ids', [f'train_{i}' for i in range(len(train_data['y']))])
    valid_ids = valid_data.get('ids', [f'valid_{i}' for i in range(len(valid_data['y']))])
    test_ids = test_data.get('ids', [f'test_{i}' for i in range(len(test_data['y']))])
    all_ids = list(train_ids) + list(valid_ids) + list(test_ids)
    subjects = np.array([extract_subject_from_id(vid, task='finger') for vid in all_ids])

    print(f"\nTotal: {X.shape}, Unique subjects: {len(np.unique(subjects))}")

    models = [
        ('OrdinalModel', OrdinalModel, {'input_size': 98, 'hidden_size': 128}),
        ('DilatedCNN', FingerDilatedCNN, {'input_size': 98, 'hidden_size': 128}),
        ('Transformer', FingerTransformer, {'input_size': 98, 'hidden_size': 128}),
    ]

    results = {}
    for name, model_class, model_kwargs in models:
        result = run_groupkfold(
            X, y, subjects, model_class, model_kwargs,
            OrdinalLoss, {'num_classes': 5}, device, f'Finger-{name}', is_ordinal=True
        )
        results[name] = result

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all', choices=['all', 'gait', 'finger'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    np.random.seed(42)
    torch.manual_seed(42)

    all_results = {}

    if args.task in ['all', 'gait']:
        all_results['gait'] = evaluate_gait_models(device)

    if args.task in ['all', 'finger']:
        all_results['finger'] = evaluate_finger_models(device)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for task, results in all_results.items():
        print(f"\n{task.upper()}:")
        for model, metrics in results.items():
            print(f"  {model}: Acc={metrics['exact_acc']*100:.1f}%, "
                  f"MAE={metrics['mae']:.3f}, r={metrics['pearson_r']:.3f}")

    # Save
    with open('results_groupkfold_multi.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: results_groupkfold_multi.pkl")


if __name__ == "__main__":
    main()
