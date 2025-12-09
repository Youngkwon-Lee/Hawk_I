"""
Finger Tapping - Ordinal Regression Final Test Evaluation
Best model from v4: OrdinalRegression (51.9% CV, Pearson 0.450)

Usage:
    python scripts/evaluate_finger_ordinal.py
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 150
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 25


# ============================================================
# Dataset
# ============================================================
class FingerDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            if np.random.random() < 0.5:
                x = x + torch.randn_like(x) * 0.02
            if np.random.random() < 0.3:
                scale = 0.9 + np.random.random() * 0.2
                x = x * scale

        return x, y


# ============================================================
# Ordinal Loss
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

        loss = F.binary_cross_entropy_with_logits(logits, binary_targets)
        return loss


# ============================================================
# Ordinal Model
# ============================================================
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
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                pred = model(X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model.predict(X)
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


def main():
    print("=" * 70)
    print("FINGER TAPPING - ORDINAL REGRESSION FINAL TEST")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading data...")
    with open('data/finger_train_v3.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/finger_valid_v3.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open('data/finger_test_v3.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Combine train + valid for final training
    X_train = np.vstack([train_data['X'], valid_data['X']])
    y_train = np.concatenate([train_data['y'], valid_data['y']])
    X_test, y_test = test_data['X'], test_data['y']

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Split for validation during training (90/10)
    n_valid = int(len(X_train) * 0.1)
    np.random.seed(42)
    indices = np.random.permutation(len(X_train))
    train_idx, valid_idx = indices[n_valid:], indices[:n_valid]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[valid_idx], y_train[valid_idx]

    # Weighted sampling
    class_counts = np.bincount(y_tr.astype(int), minlength=5)
    sample_weights = 1.0 / (class_counts[y_tr.astype(int)] + 1)
    sample_weights = sample_weights ** 1.5
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights) * 2)

    train_dataset = FingerDataset(X_tr, y_tr, augment=True)
    valid_dataset = FingerDataset(X_val, y_val, augment=False)
    test_dataset = FingerDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Create model
    model = OrdinalModel(input_size=98, hidden_size=128, dropout=0.4).to(device)
    criterion = OrdinalLoss(num_classes=5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    print(f"\nTraining OrdinalModel...")

    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate(model, valid_loader, device)
        scheduler.step()

        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, "
                  f"Val MAE={val_metrics['mae']:.3f}, "
                  f"Exact={val_metrics['exact_acc']*100:.1f}%")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Final Test Evaluation
    print(f"\n{'='*70}")
    print("FINAL TEST SET RESULTS")
    print('='*70)

    test_metrics = evaluate(model, test_loader, device)

    print(f"\nOrdinalRegression Test Results:")
    print(f"  MAE: {test_metrics['mae']:.3f}")
    print(f"  Exact Accuracy: {test_metrics['exact_acc']*100:.1f}%")
    print(f"  Within-1: {test_metrics['within1_acc']*100:.1f}%")
    print(f"  Pearson r: {test_metrics['pearson_r']:.3f}")

    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(test_metrics['labels'], test_metrics['preds_rounded'], labels=list(range(5)))
    print(f"{'':>10} |   0    1    2    3    4")
    print("-" * 45)
    for i in range(5):
        row = cm[i]
        if row.sum() > 0:
            print(f"{i:>10} | {row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>4} {row[4]:>4}  (n={row.sum()})")

    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for i in range(5):
        mask = test_metrics['labels'] == i
        if mask.sum() > 0:
            correct = (test_metrics['preds_rounded'][mask] == i).sum()
            total = mask.sum()
            acc = correct / total * 100
            print(f"  Score {i}: {correct}/{total} = {acc:.1f}%")

    # Prediction distribution
    print(f"\nPrediction Distribution:")
    pred_counts = np.bincount(test_metrics['preds_rounded'].astype(int), minlength=5)
    print(f"  Predicted: {dict(enumerate(pred_counts))}")
    label_counts = np.bincount(test_metrics['labels'].astype(int), minlength=5)
    print(f"  Actual:    {dict(enumerate(label_counts))}")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/finger_ordinal_final.pth')
    print(f"\nModel saved to: models/finger_ordinal_final.pth")

    # Save results
    results = {
        'test_metrics': test_metrics,
        'confusion_matrix': cm
    }
    with open('results_finger_ordinal_test.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: results_finger_ordinal_test.pkl")


if __name__ == "__main__":
    main()
