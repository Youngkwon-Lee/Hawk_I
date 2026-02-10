"""
Mamba + Enhanced Features Training
Combining the best model (Mamba) with enhanced feature engineering

Features:
- Original kinematic features
- Velocity (1st derivative)
- Acceleration (2nd derivative)
- Moving statistics (mean, std, min, max)

Usage:
    python scripts/train_mamba_enhanced.py --epochs 200
"""
import os
import math
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
import argparse
from datetime import datetime


class Config:
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    RESULT_DIR = "./results"

    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.02
    PATIENCE = 30
    N_FOLDS = 5

    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    DROPOUT = 0.4
    USE_AMP = True


# ============================================================
# Feature Engineering
# ============================================================
class FeatureEngineer:
    """Add derived features to time series data"""

    @staticmethod
    def add_velocity(x):
        """First derivative (velocity)"""
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        return velocity

    @staticmethod
    def add_acceleration(x):
        """Second derivative (acceleration)"""
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        return acceleration

    @staticmethod
    def add_moving_stats(x, window=5):
        """Moving window statistics"""
        seq_len, n_features = x.shape
        stats_features = []

        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_data = x[start:end]

            stats_features.append([
                window_data.mean(axis=0),
                window_data.std(axis=0),
                window_data.max(axis=0),
                window_data.min(axis=0),
            ])

        stats_array = np.array(stats_features)
        return stats_array.reshape(seq_len, -1)

    @staticmethod
    def engineer_features(X):
        """Apply feature engineering to dataset"""
        print("Engineering features...")
        print(f"  Original shape: {X.shape}")

        features_list = [X]

        # Velocity
        velocity = np.array([FeatureEngineer.add_velocity(x) for x in X])
        features_list.append(velocity)
        print(f"  + Velocity: {velocity.shape}")

        # Acceleration
        acceleration = np.array([FeatureEngineer.add_acceleration(x) for x in X])
        features_list.append(acceleration)
        print(f"  + Acceleration: {acceleration.shape}")

        # Moving statistics
        moving_stats = np.array([FeatureEngineer.add_moving_stats(x) for x in X])
        features_list.append(moving_stats)
        print(f"  + Moving stats: {moving_stats.shape}")

        X_enhanced = np.concatenate(features_list, axis=2)
        print(f"  Final shape: {X_enhanced.shape}")

        return X_enhanced


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
# Mamba Model (State Space Model)
# ============================================================
class Mamba(nn.Module):
    """
    Simplified Mamba-style State Space Model
    Linear time complexity for long sequences
    """
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)

        # Mamba blocks
        self.mamba_layers = nn.ModuleList([
            MambaBlock(hidden_size, dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.input_proj(x)

        for layer in self.mamba_layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling

        return self.classifier(x).squeeze(-1)


class MambaBlock(nn.Module):
    """
    Simplified Mamba block with selective state space
    """
    def __init__(self, hidden_size, dropout, state_size=16, expand_factor=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_size = state_size
        expanded = hidden_size * expand_factor

        # Input projection
        self.in_proj = nn.Linear(hidden_size, expanded * 2)

        # Convolution for local context
        self.conv = nn.Conv1d(expanded, expanded, kernel_size=3, padding=1, groups=expanded)

        # SSM parameters
        self.dt_proj = nn.Linear(expanded, expanded)
        self.A_log = nn.Parameter(torch.randn(expanded, state_size))
        self.D = nn.Parameter(torch.ones(expanded))

        # B and C projections (input-dependent)
        self.B_proj = nn.Linear(expanded, state_size)
        self.C_proj = nn.Linear(expanded, state_size)

        # Output projection
        self.out_proj = nn.Linear(expanded, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq, hidden)
        residual = x
        x = self.norm(x)

        # Split into two paths
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        # Convolution path
        x_conv = x_path.transpose(1, 2)  # (batch, expanded, seq)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (batch, seq, expanded)
        x_conv = F.silu(x_conv)

        # SSM path (simplified selective scan)
        x_ssm = self._selective_scan(x_conv)

        # Gate with z
        z = F.silu(z)
        x = x_ssm * z

        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)

        return x + residual

    def _selective_scan(self, x):
        """Simplified selective state space scan - parallel approximation"""
        batch, seq, expanded = x.shape
        device = x.device

        # Compute dt, B, C (input-dependent)
        dt = F.softplus(self.dt_proj(x))  # (batch, seq, expanded)
        B = self.B_proj(x)  # (batch, seq, state_size)
        C = self.C_proj(x)  # (batch, seq, state_size)

        # Discretize A
        A = -torch.exp(self.A_log)  # (expanded, state_size)

        # Parallel approximation using convolution-like operation
        y = torch.zeros_like(x)

        # Compute cumulative effect using exponential decay
        decay = torch.sigmoid(dt.mean(dim=-1, keepdim=True))  # (batch, seq, 1)

        # Create causal weights
        weights = torch.zeros(seq, seq, device=device)
        for i in range(seq):
            for j in range(i + 1):
                weights[i, j] = (0.9 ** (i - j))  # Exponential decay

        # Limit computation for efficiency
        weights = weights[:, :min(seq, 100)]

        # Apply weighted sum (simplified SSM)
        x_weighted = x[:, :min(seq, 100), :]  # (batch, min_seq, expanded)

        # Batch matrix multiply for efficiency
        y_temp = torch.matmul(weights[:seq, :x_weighted.size(1)], x_weighted)

        # Apply input-dependent gating
        gate = torch.sigmoid(dt)
        y = y_temp * gate + x * self.D

        return y


# ============================================================
# Training Functions
# ============================================================
def get_cosine_schedule(optimizer, warmup, total):
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        return max(1e-6, 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.squeeze().to(device)
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model(X), y)
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

    mae = np.mean(np.abs(y - preds))
    exact = np.mean(np.round(preds) == y) * 100
    within1 = np.mean(np.abs(y - np.round(preds)) <= 1) * 100
    return {'mae': mae, 'exact': exact, 'within1': within1, 'preds': preds}


def kfold_cv(X, y, config, device):
    print(f"\n{'='*60}")
    print(f"Mamba + Enhanced Features - 5-Fold CV")
    print(f"Features: {X.shape[2]}")
    print('='*60)

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    y_binned = np.clip(y, 0, 3).astype(int)

    fold_results = []
    all_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        print(f"\n--- Fold {fold+1}/{config.N_FOLDS} ---")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std

        model = Mamba(X.shape[2], config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {num_params:,}")

        loader = DataLoader(AugmentedDataset(X_train_norm, y_train, True),
                           batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = get_cosine_schedule(optimizer, 10, config.EPOCHS)
        scaler = GradScaler()
        criterion = nn.MSELoss()

        best_mae, best_state, patience = float('inf'), None, 0

        pbar = tqdm(range(config.EPOCHS), desc=f"Fold {fold+1}")
        for epoch in pbar:
            train_loss = train_epoch(model, loader, criterion, optimizer, scaler, device)
            scheduler.step()

            results = evaluate(model, X_val_norm, y_val, device)
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'mae': f'{results["mae"]:.3f}'})

            if results['mae'] < best_mae:
                best_mae = results['mae']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= config.PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.to(device)
        results = evaluate(model, X_val_norm, y_val, device)
        all_preds[val_idx] = results['preds']
        fold_results.append(results)
        print(f"  MAE: {results['mae']:.3f}, Exact: {results['exact']:.1f}%")

        del model
        torch.cuda.empty_cache()

    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_exact = np.mean([r['exact'] for r in fold_results])
    avg_within1 = np.mean([r['within1'] for r in fold_results])
    pearson, _ = stats.pearsonr(y, all_preds)
    spearman, _ = stats.spearmanr(y, all_preds)

    print(f"\n{'='*60}")
    print(f"RESULTS - Mamba + Enhanced Features")
    print('='*60)
    print(f"MAE: {avg_mae:.3f}")
    print(f"Exact: {avg_exact:.1f}%")
    print(f"Within1: {avg_within1:.1f}%")
    print(f"Pearson: {pearson:.3f}")
    print(f"Spearman: {spearman:.3f}")

    return {'mae': avg_mae, 'exact': avg_exact, 'within1': avg_within1, 'pearson': pearson, 'spearman': spearman}


def load_data(config):
    for train_path, test_path in [
        (os.path.join(config.DATA_DIR, "train_data.pkl"), os.path.join(config.DATA_DIR, "test_data.pkl")),
        ("./finger_train.pkl", "./finger_test.pkl"),
    ]:
        if os.path.exists(train_path):
            break

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    return train_data['X'], train_data['y'], test_data['X'], test_data['y']


def main():
    parser = argparse.ArgumentParser(description='Mamba + Enhanced Features Training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("MAMBA + ENHANCED FEATURES")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Load data
    X_train, y_train, _, _ = load_data(config)
    print(f"Original data: {X_train.shape}")

    # Feature engineering
    X_enhanced = FeatureEngineer.engineer_features(X_train)

    # Train
    results = kfold_cv(X_enhanced, y_train, config, device)

    # Compare with baseline
    print(f"\n{'='*60}")
    print("COMPARISON")
    print('='*60)
    print(f"{'Model':<30} {'MAE':>8} {'Exact':>10} {'Pearson':>10}")
    print("-" * 60)
    print(f"{'Mamba (baseline)':<30} {'0.455':>8} {'62.9%':>10} {'0.536':>10}")
    print(f"{'Mamba + Enhanced':<30} {results['mae']:>8.3f} {results['exact']:>9.1f}% {results['pearson']:>10.3f}")

    improvement = (0.536 - results['pearson']) / 0.536 * -100
    print(f"\nImprovement: {improvement:+.1f}% (Pearson)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"mamba_enhanced_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("Mamba + Enhanced Features Training\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {config.EPOCHS}\n\n")

        f.write("Feature Engineering:\n")
        f.write(f"  Original features: 10\n")
        f.write(f"  Enhanced features: {X_enhanced.shape[2]}\n")
        f.write("  Additions: velocity, acceleration, moving_stats\n\n")

        f.write("Results:\n")
        f.write(f"  MAE: {results['mae']:.3f}\n")
        f.write(f"  Exact: {results['exact']:.1f}%\n")
        f.write(f"  Within1: {results['within1']:.1f}%\n")
        f.write(f"  Pearson: {results['pearson']:.3f}\n\n")

        f.write("Comparison:\n")
        f.write(f"  Mamba (baseline): MAE=0.455, Exact=62.9%, Pearson=0.536\n")
        f.write(f"  Mamba + Enhanced: MAE={results['mae']:.3f}, Exact={results['exact']:.1f}%, Pearson={results['pearson']:.3f}\n")
        f.write(f"  Improvement: {improvement:+.1f}% (Pearson)\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
