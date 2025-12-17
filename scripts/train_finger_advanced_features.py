"""
Mamba + Advanced Features (IQR, Aperiodicity, Entropy)
Based on ParkTest paper features that achieved strong correlation

New Features:
- IQR (Inter-quartile range) of speed - strongest predictor (r=-0.56)
- Aperiodicity (cycle irregularity)
- Period entropy
- Amplitude decrement ratio
- Freezing detection

Usage:
    python scripts/train_finger_advanced_features.py --epochs 200
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
from scipy.signal import find_peaks
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
# Advanced Feature Engineering (ParkTest-inspired)
# ============================================================
class AdvancedFeatureEngineer:
    """Advanced features based on ParkTest paper"""

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
    def compute_iqr_features(x, window=10):
        """
        Inter-quartile range of speed - strongest predictor in ParkTest (r=-0.56)
        Captures variability in movement speed
        """
        seq_len, n_features = x.shape
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        speed = np.abs(velocity)

        iqr_features = []
        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_speed = speed[start:end]

            q75, q25 = np.percentile(window_speed, [75, 25], axis=0)
            iqr = q75 - q25
            iqr_features.append(iqr)

        return np.array(iqr_features)

    @staticmethod
    def compute_aperiodicity(x, window=20):
        """
        Aperiodicity - measures cycle irregularity
        PD patients show more irregular movement patterns
        """
        seq_len, n_features = x.shape
        aperiodicity_features = []

        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_data = x[start:end]

            # Compute autocorrelation-based aperiodicity
            aperiodicities = []
            for f in range(n_features):
                signal = window_data[:, f]
                if len(signal) > 3:
                    # Find peaks to estimate periodicity
                    peaks, _ = find_peaks(signal, distance=2)
                    if len(peaks) > 1:
                        # Compute period variability
                        periods = np.diff(peaks)
                        if len(periods) > 0:
                            period_cv = np.std(periods) / (np.mean(periods) + 1e-8)
                            aperiodicities.append(period_cv)
                        else:
                            aperiodicities.append(0.5)  # default
                    else:
                        aperiodicities.append(0.5)
                else:
                    aperiodicities.append(0.5)

            aperiodicity_features.append(aperiodicities)

        return np.array(aperiodicity_features)

    @staticmethod
    def compute_entropy(x, window=15):
        """
        Signal entropy - measures complexity/randomness
        Higher entropy indicates more irregular movements
        """
        seq_len, n_features = x.shape
        entropy_features = []

        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_data = x[start:end]

            entropies = []
            for f in range(n_features):
                signal = window_data[:, f]
                # Discretize and compute Shannon entropy
                hist, _ = np.histogram(signal, bins=10, density=True)
                hist = hist + 1e-10  # avoid log(0)
                entropy = -np.sum(hist * np.log2(hist))
                entropies.append(entropy)

            entropy_features.append(entropies)

        return np.array(entropy_features)

    @staticmethod
    def compute_amplitude_decrement(x, window=30):
        """
        Amplitude decrement ratio - fatigue indicator
        PD patients show faster amplitude decline
        """
        seq_len, n_features = x.shape
        decrement_features = []

        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_data = x[start:end]

            decrements = []
            for f in range(n_features):
                signal = window_data[:, f]
                if len(signal) > 5:
                    # Compute amplitude over time
                    first_half = np.abs(signal[:len(signal)//2]).mean()
                    second_half = np.abs(signal[len(signal)//2:]).mean()
                    decrement = (first_half - second_half) / (first_half + 1e-8)
                    decrements.append(decrement)
                else:
                    decrements.append(0)

            decrement_features.append(decrements)

        return np.array(decrement_features)

    @staticmethod
    def detect_freezing(x, threshold=0.1, window=10):
        """
        Freezing detection - periods of very low movement
        Common in PD patients
        """
        seq_len, n_features = x.shape
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        speed = np.sqrt((velocity ** 2).sum(axis=1, keepdims=True))

        freezing_features = []
        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_speed = speed[start:end]

            # Fraction of time below threshold
            freeze_fraction = (window_speed < threshold).mean()
            freezing_features.append([freeze_fraction] * n_features)

        return np.array(freezing_features)

    @staticmethod
    def engineer_features(X):
        """Apply all feature engineering to dataset"""
        print("Engineering advanced features...")
        print(f"  Original shape: {X.shape}")

        features_list = [X]

        # Basic features
        velocity = np.array([AdvancedFeatureEngineer.add_velocity(x) for x in X])
        features_list.append(velocity)
        print(f"  + Velocity: {velocity.shape}")

        acceleration = np.array([AdvancedFeatureEngineer.add_acceleration(x) for x in X])
        features_list.append(acceleration)
        print(f"  + Acceleration: {acceleration.shape}")

        moving_stats = np.array([AdvancedFeatureEngineer.add_moving_stats(x) for x in X])
        features_list.append(moving_stats)
        print(f"  + Moving stats: {moving_stats.shape}")

        # NEW: Advanced features from ParkTest
        iqr = np.array([AdvancedFeatureEngineer.compute_iqr_features(x) for x in X])
        features_list.append(iqr)
        print(f"  + IQR (speed): {iqr.shape}")

        aperiodicity = np.array([AdvancedFeatureEngineer.compute_aperiodicity(x) for x in X])
        features_list.append(aperiodicity)
        print(f"  + Aperiodicity: {aperiodicity.shape}")

        entropy = np.array([AdvancedFeatureEngineer.compute_entropy(x) for x in X])
        features_list.append(entropy)
        print(f"  + Entropy: {entropy.shape}")

        decrement = np.array([AdvancedFeatureEngineer.compute_amplitude_decrement(x) for x in X])
        features_list.append(decrement)
        print(f"  + Amplitude decrement: {decrement.shape}")

        freezing = np.array([AdvancedFeatureEngineer.detect_freezing(x) for x in X])
        features_list.append(freezing)
        print(f"  + Freezing detection: {freezing.shape}")

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
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)

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
        x = self.input_proj(x)

        for layer in self.mamba_layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.classifier(x).squeeze(-1)


class MambaBlock(nn.Module):
    def __init__(self, hidden_size, dropout, state_size=16, expand_factor=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_size = state_size
        expanded = hidden_size * expand_factor

        self.in_proj = nn.Linear(hidden_size, expanded * 2)
        self.conv = nn.Conv1d(expanded, expanded, kernel_size=3, padding=1, groups=expanded)

        self.dt_proj = nn.Linear(expanded, expanded)
        self.A_log = nn.Parameter(torch.randn(expanded, state_size))
        self.D = nn.Parameter(torch.ones(expanded))

        self.B_proj = nn.Linear(expanded, state_size)
        self.C_proj = nn.Linear(expanded, state_size)

        self.out_proj = nn.Linear(expanded, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        x_conv = x_path.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_ssm = self._selective_scan(x_conv)

        z = F.silu(z)
        x = x_ssm * z

        x = self.out_proj(x)
        x = self.dropout(x)

        return x + residual

    def _selective_scan(self, x):
        batch, seq, expanded = x.shape
        device = x.device

        dt = F.softplus(self.dt_proj(x))
        B = self.B_proj(x)
        C = self.C_proj(x)

        A = -torch.exp(self.A_log)

        y = torch.zeros_like(x)
        decay = torch.sigmoid(dt.mean(dim=-1, keepdim=True))

        weights = torch.zeros(seq, seq, device=device)
        for i in range(seq):
            for j in range(i + 1):
                weights[i, j] = (0.9 ** (i - j))

        weights = weights[:, :min(seq, 100)]
        x_weighted = x[:, :min(seq, 100), :]

        y_temp = torch.matmul(weights[:seq, :x_weighted.size(1)], x_weighted)

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
    print(f"Mamba + Advanced Features (ParkTest-inspired)")
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

        # Replace NaN/Inf
        X_train_norm = np.nan_to_num(X_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_norm = np.nan_to_num(X_val_norm, nan=0.0, posinf=0.0, neginf=0.0)

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
    print(f"RESULTS - Mamba + Advanced Features")
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
    parser = argparse.ArgumentParser(description='Mamba + Advanced Features Training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("MAMBA + ADVANCED FEATURES (ParkTest-inspired)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Load data
    X_train, y_train, _, _ = load_data(config)
    print(f"Original data: {X_train.shape}")

    # Feature engineering
    X_enhanced = AdvancedFeatureEngineer.engineer_features(X_train)

    # Train
    results = kfold_cv(X_enhanced, y_train, config, device)

    # Compare with baseline
    print(f"\n{'='*60}")
    print("COMPARISON")
    print('='*60)
    print(f"{'Model':<35} {'MAE':>8} {'Exact':>10} {'Pearson':>10}")
    print("-" * 65)
    print(f"{'Mamba (baseline)':<35} {'0.455':>8} {'62.9%':>10} {'0.536':>10}")
    print(f"{'Mamba + Enhanced':<35} {'0.444':>8} {'63.0%':>10} {'0.609':>10}")
    print(f"{'Mamba + Advanced (this)':<35} {results['mae']:>8.3f} {results['exact']:>9.1f}% {results['pearson']:>10.3f}")

    improvement = (0.609 - results['pearson']) / 0.609 * -100
    print(f"\nImprovement over Enhanced: {improvement:+.1f}% (Pearson)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"finger_advanced_features_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("Mamba + Advanced Features (ParkTest-inspired)\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {config.EPOCHS}\n\n")

        f.write("New Features (based on ParkTest paper):\n")
        f.write("  1. IQR of speed (strongest predictor, r=-0.56)\n")
        f.write("  2. Aperiodicity (cycle irregularity)\n")
        f.write("  3. Signal entropy\n")
        f.write("  4. Amplitude decrement ratio\n")
        f.write("  5. Freezing detection\n\n")

        f.write(f"Total features: {X_enhanced.shape[2]}\n\n")

        f.write("Results:\n")
        f.write(f"  MAE: {results['mae']:.3f}\n")
        f.write(f"  Exact: {results['exact']:.1f}%\n")
        f.write(f"  Within1: {results['within1']:.1f}%\n")
        f.write(f"  Pearson: {results['pearson']:.3f}\n")
        f.write(f"  Spearman: {results['spearman']:.3f}\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
