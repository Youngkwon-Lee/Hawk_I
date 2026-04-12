"""
TCN Hyperparameter Tuning
Grid search over key hyperparameters

Usage:
    python scripts/train_tcn_tuning.py
"""
import os
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from scipy.interpolate import CubicSpline
from datetime import datetime
from itertools import product


class Config:
    DATA_DIR = "./data"
    RESULT_DIR = "./results"
    EPOCHS = 150  # Shorter for tuning
    PATIENCE = 20
    N_FOLDS = 3  # 3-fold for faster tuning
    USE_AMP = True


# Hyperparameter search space
PARAM_GRID = {
    'hidden_size': [128, 256, 512],
    'num_layers': [3, 4, 5, 6],
    'dropout': [0.3, 0.4, 0.5],
    'kernel_size': [3, 5, 7],
    'learning_rate': [0.001, 0.0005, 0.0003],
}

# Top configurations to try (reduced for efficiency)
TOP_CONFIGS = [
    {'hidden_size': 256, 'num_layers': 4, 'dropout': 0.4, 'kernel_size': 3, 'learning_rate': 0.0005},  # baseline
    {'hidden_size': 512, 'num_layers': 4, 'dropout': 0.4, 'kernel_size': 3, 'learning_rate': 0.0003},  # larger
    {'hidden_size': 256, 'num_layers': 6, 'dropout': 0.4, 'kernel_size': 3, 'learning_rate': 0.0005},  # deeper
    {'hidden_size': 256, 'num_layers': 4, 'dropout': 0.3, 'kernel_size': 5, 'learning_rate': 0.0005},  # wider kernel
    {'hidden_size': 512, 'num_layers': 5, 'dropout': 0.4, 'kernel_size': 5, 'learning_rate': 0.0003},  # large + deep
    {'hidden_size': 256, 'num_layers': 5, 'dropout': 0.5, 'kernel_size': 3, 'learning_rate': 0.0005},  # more regularization
    {'hidden_size': 384, 'num_layers': 4, 'dropout': 0.4, 'kernel_size': 3, 'learning_rate': 0.0004},  # medium
    {'hidden_size': 256, 'num_layers': 4, 'dropout': 0.4, 'kernel_size': 7, 'learning_rate': 0.0005},  # large kernel
]


# ============================================================
# Data Augmentation
# ============================================================
class TimeSeriesAugmentation:
    @staticmethod
    def augment(x, p=0.5):
        if np.random.random() < p:
            x = x + np.random.normal(0, 0.03, x.shape)
        if np.random.random() < p:
            x = x * np.random.normal(1, 0.1, (1, x.shape[1]))
        return x


class AugmentedDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = TimeSeriesAugmentation.augment(x)
        return torch.FloatTensor(x), torch.FloatTensor([self.y[idx]])


# ============================================================
# TCN Model
# ============================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        if out.size(2) != residual.size(2):
            diff = residual.size(2) - out.size(2)
            out = F.pad(out, (diff // 2, diff - diff // 2))
        return F.gelu(out + residual)


class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, hidden_size, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        return self.classifier(x).squeeze(-1)


# ============================================================
# Training
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.squeeze().to(device)
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model(batch_X), batch_y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X), 64):
            pred = model(torch.FloatTensor(X[i:i+64]).to(device)).cpu().numpy()
            preds.append(pred)
        preds = np.clip(np.concatenate(preds), 0, 4)
    return np.mean(np.abs(y - preds)), np.mean(np.round(preds) == y) * 100


def evaluate_config(X, y, config_params, device, n_folds=3, epochs=150):
    """Evaluate a single configuration with k-fold CV"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_binned = np.clip(y, 0, 3).astype(int)

    maes, exacts = [], []

    for train_idx, val_idx in skf.split(X, y_binned):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std

        model = TCN(
            X.shape[2],
            config_params['hidden_size'],
            config_params['num_layers'],
            config_params['dropout'],
            config_params['kernel_size']
        ).to(device)

        train_loader = DataLoader(
            AugmentedDataset(X_train_norm, y_train, augment=True),
            batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=config_params['learning_rate'], weight_decay=0.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        scaler = GradScaler()
        criterion = nn.MSELoss()

        best_mae = float('inf')
        patience = 0

        for epoch in range(epochs):
            train_epoch(model, train_loader, criterion, optimizer, scaler, device)
            scheduler.step()

            mae, _ = evaluate(model, X_val_norm, y_val, device)
            if mae < best_mae:
                best_mae = mae
                patience = 0
            else:
                patience += 1
                if patience >= 20:
                    break

        mae, exact = evaluate(model, X_val_norm, y_val, device)
        maes.append(mae)
        exacts.append(exact)

        del model
        torch.cuda.empty_cache()

    return np.mean(maes), np.mean(exacts)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Testing {len(TOP_CONFIGS)} configurations")

    config = Config()
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Load data
    for train_path, test_path in [
        (os.path.join(config.DATA_DIR, "train_data.pkl"), os.path.join(config.DATA_DIR, "test_data.pkl")),
        ("./finger_train.pkl", "./finger_test.pkl"),
    ]:
        if os.path.exists(train_path):
            break

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    X, y = train_data['X'], train_data['y']
    print(f"Data: {X.shape}")

    results = []

    for i, params in enumerate(TOP_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Config {i+1}/{len(TOP_CONFIGS)}")
        print(f"Params: {params}")
        print('='*60)

        mae, exact = evaluate_config(X, y, params, device, n_folds=config.N_FOLDS, epochs=config.EPOCHS)

        results.append({
            'params': params,
            'mae': mae,
            'exact': exact
        })

        print(f"Result: MAE={mae:.3f}, Exact={exact:.1f}%")

    # Sort by MAE
    results.sort(key=lambda x: x['mae'])

    print(f"\n{'='*60}")
    print("TUNING RESULTS (sorted by MAE)")
    print('='*60)

    for i, r in enumerate(results):
        print(f"\n{i+1}. MAE: {r['mae']:.3f}, Exact: {r['exact']:.1f}%")
        print(f"   {r['params']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"tcn_tuning_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("TCN Hyperparameter Tuning Results\n")
        f.write(f"Date: {datetime.now()}\n\n")

        for i, r in enumerate(results):
            f.write(f"{i+1}. MAE: {r['mae']:.3f}, Exact: {r['exact']:.1f}%\n")
            f.write(f"   {r['params']}\n\n")

        f.write(f"\nBest config:\n{results[0]['params']}\n")

    print(f"\nSaved: {result_path}")
    print(f"\nBest: {results[0]['params']}")


if __name__ == "__main__":
    main()
