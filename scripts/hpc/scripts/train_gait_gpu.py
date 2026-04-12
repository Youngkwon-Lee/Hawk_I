"""
Train LSTM models on PD4T Gait data (v1 - Clinical features only)
Run on HPC with GPU

Features: 30 per frame (15 position + 15 velocity)

Usage:
    python scripts/train_gait_gpu.py
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
TRAIN_FILE = 'data/gait_train_data.pkl'
TEST_FILE = 'data/gait_test_data.pkl'
SEQ_LEN = 300
INPUT_SIZE = 30  # 15 position + 15 velocity

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
N_FOLDS = 5


# ============================================================
# Dataset
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Models
# ============================================================
class AttentionLSTM(nn.Module):
    """Bidirectional LSTM with Attention"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(context).squeeze(-1)


class TransformerModel(nn.Module):
    """Transformer Encoder for sequence classification"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, nhead=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x).squeeze(-1)


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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    preds_rounded = np.clip(np.round(preds), 0, 4)

    mae = mean_absolute_error(labels, preds)
    exact_acc = accuracy_score(labels, preds_rounded)
    within1 = np.mean(np.abs(preds_rounded - labels) <= 1)
    r, _ = pearsonr(preds, labels) if len(np.unique(labels)) > 1 else (0, 0)

    return {'mae': mae, 'exact_acc': exact_acc, 'within1_acc': within1, 'pearson_r': r}


def train_model(model, train_loader, valid_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate(model, valid_loader, device)
        scheduler.step(val_metrics['mae'])

        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Val MAE={val_metrics['mae']:.3f}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_mae


def run_cross_validation(X, y, model_class, device):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model = model_class(INPUT_SIZE, hidden_size=128, num_layers=2).to(device)
        model, _ = train_model(model, train_loader, val_loader, device)

        val_metrics = evaluate(model, val_loader, device)
        fold_results.append(val_metrics)
        print(f"  Fold {fold+1}: MAE={val_metrics['mae']:.3f}, Exact={val_metrics['exact_acc']*100:.1f}%")

    return {
        'cv_mae': np.mean([r['mae'] for r in fold_results]),
        'cv_exact_acc': np.mean([r['exact_acc'] for r in fold_results]),
        'cv_within1_acc': np.mean([r['within1_acc'] for r in fold_results]),
        'cv_pearson_r': np.mean([r['pearson_r'] for r in fold_results]),
    }


def main():
    print("=" * 60)
    print("PD4T GAIT - LSTM Training (Clinical Features)")
    print(f"Features: {INPUT_SIZE} per frame")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading data from {TRAIN_FILE}...")
    with open(TRAIN_FILE, 'rb') as f:
        train_data = pickle.load(f)

    X = train_data['X']
    y = train_data['y']
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.bincount(y.astype(int))}")

    # Models
    models = [
        ('AttentionLSTM', AttentionLSTM),
        ('Transformer', TransformerModel),
    ]

    results = {}
    for name, model_class in models:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print('='*60)
        try:
            cv_results = run_cross_validation(X, y, model_class, device)
            results[name] = cv_results
            print(f"\n{name} CV Results:")
            print(f"  MAE: {cv_results['cv_mae']:.3f}")
            print(f"  Exact: {cv_results['cv_exact_acc']*100:.1f}%")
            print(f"  Within-1: {cv_results['cv_within1_acc']*100:.1f}%")
        except Exception as e:
            print(f"Error: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - GAIT")
    print('='*60)
    for name, res in results.items():
        print(f"{name:<20} MAE={res['cv_mae']:.3f}, Exact={res['cv_exact_acc']*100:.1f}%, Within-1={res['cv_within1_acc']*100:.1f}%")


if __name__ == "__main__":
    main()
