"""
Mamba + Advanced Features (IQR, Aperiodicity, Entropy) for Gait Task
Based on ParkTest paper features that achieved strong correlation

New Features:
- IQR (Inter-quartile range) of speed - strongest predictor (r=-0.56)
- Aperiodicity (cycle irregularity)
- Period entropy
- Amplitude decrement ratio
- Freezing detection

Usage:
    python scripts/train_gait_advanced_features.py --epochs 200
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
            window_data = speed[start:end]

            q75 = np.percentile(window_data, 75, axis=0)
            q25 = np.percentile(window_data, 25, axis=0)
            iqr = q75 - q25
            iqr_features.append(iqr)

        return np.array(iqr_features)

    @staticmethod
    def compute_aperiodicity(x, window=30):
        """
        Aperiodicity - cycle irregularity
        High values indicate PD-related movement abnormalities
        """
        seq_len, n_features = x.shape
        aperiodicity_features = []

        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_data = x[start:end]

            aperiod = np.zeros(n_features)
            for feat in range(n_features):
                signal = window_data[:, feat]
                if len(signal) > 5:
                    # Auto-correlation based aperiodicity
                    if np.std(signal) > 1e-6:
                        autocorr = np.correlate(signal - signal.mean(), signal - signal.mean(), mode='full')
                        autocorr = autocorr[len(autocorr)//2:]
                        if len(autocorr) > 1 and autocorr[0] > 0:
                            autocorr = autocorr / autocorr[0]
                            # Find first minimum as aperiodicity measure
                            peaks, _ = find_peaks(-autocorr)
                            if len(peaks) > 0:
                                aperiod[feat] = autocorr[peaks[0]]
            aperiodicity_features.append(aperiod)

        return np.array(aperiodicity_features)

    @staticmethod
    def compute_entropy_features(x, window=15):
        """
        Sample entropy approximation for movement regularity
        Higher entropy = more irregular movement (PD indicator)
        """
        seq_len, n_features = x.shape
        entropy_features = []

        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_data = x[start:end]

            entropy = np.zeros(n_features)
            for feat in range(n_features):
                signal = window_data[:, feat]
                if len(signal) > 3 and np.std(signal) > 1e-6:
                    # Approximate entropy using histogram
                    hist, _ = np.histogram(signal, bins=min(10, len(signal)//2), density=True)
                    hist = hist[hist > 0]
                    entropy[feat] = -np.sum(hist * np.log(hist + 1e-10))
            entropy_features.append(entropy)

        return np.array(entropy_features)

    @staticmethod
    def compute_amplitude_decrement(x, window=50):
        """
        Amplitude decrement - fatigue indicator
        Measures how amplitude decreases over time (PD symptom)
        """
        seq_len, n_features = x.shape
        amp_decrement = []

        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_data = x[start:end]

            decrement = np.zeros(n_features)
            if len(window_data) > 3:
                # Compute amplitude trend
                for feat in range(n_features):
                    signal = np.abs(window_data[:, feat])
                    if len(signal) > 1:
                        # Linear regression slope
                        t = np.arange(len(signal))
                        if np.std(signal) > 1e-6:
                            slope = np.polyfit(t, signal, 1)[0]
                            decrement[feat] = slope
            amp_decrement.append(decrement)

        return np.array(amp_decrement)

    @staticmethod
    def detect_freezing(x, threshold=0.01, window=10):
        """
        Freezing detection - periods of abnormally low movement
        Common in advanced PD
        """
        seq_len, n_features = x.shape
        velocity = np.abs(np.diff(x, axis=0, prepend=x[0:1]))

        freezing_features = []
        for i in range(seq_len):
            start = max(0, i - window // 2)
            end = min(seq_len, i + window // 2 + 1)
            window_vel = velocity[start:end]

            # Percentage of time with very low velocity
            freezing_ratio = np.mean(window_vel < threshold, axis=0)
            freezing_features.append(freezing_ratio)

        return np.array(freezing_features)

    @staticmethod
    def engineer_features(X):
        """Apply all advanced feature engineering"""
        print("Engineering advanced features...")
        print(f"  Original shape: {X.shape}")

        features_list = [X]

        # Basic kinematics
        velocity = np.array([AdvancedFeatureEngineer.add_velocity(x) for x in X])
        features_list.append(velocity)

        acceleration = np.array([AdvancedFeatureEngineer.add_acceleration(x) for x in X])
        features_list.append(acceleration)

        # Moving statistics
        moving_stats = np.array([AdvancedFeatureEngineer.add_moving_stats(x) for x in X])
        features_list.append(moving_stats)

        # ParkTest-inspired features
        print("  Computing IQR features...")
        iqr_features = np.array([AdvancedFeatureEngineer.compute_iqr_features(x) for x in X])
        features_list.append(iqr_features)

        print("  Computing aperiodicity features...")
        aperiodicity = np.array([AdvancedFeatureEngineer.compute_aperiodicity(x) for x in X])
        features_list.append(aperiodicity)

        print("  Computing entropy features...")
        entropy = np.array([AdvancedFeatureEngineer.compute_entropy_features(x) for x in X])
        features_list.append(entropy)

        print("  Computing amplitude decrement...")
        amp_dec = np.array([AdvancedFeatureEngineer.compute_amplitude_decrement(x) for x in X])
        features_list.append(amp_dec)

        print("  Computing freezing detection...")
        freezing = np.array([AdvancedFeatureEngineer.detect_freezing(x) for x in X])
        features_list.append(freezing)

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
# Mamba Block (Same as baseline)
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.randn(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :x.size(2)]
        x = x.transpose(1, 2)
        x = F.silu(x)

        B, L, D = x.shape
        x_proj = self.x_proj(x)
        delta, B_param = x_proj.chunk(2, dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        y = x * self.D.unsqueeze(0).unsqueeze(0)
        y = y + x * torch.tanh(B_param).mean(dim=-1, keepdim=True)
        y = y * F.silu(z)
        return self.out_proj(y) + residual


class MambaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = self.dropout(layer(x))
        x = x.mean(dim=1)
        return self.head(x)


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast():
            pred = model(X)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(X)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            with autocast():
                pred = model(X)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(y.numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels):
    """Compute regression and classification metrics"""
    # Clip predictions to valid range
    preds_clipped = np.clip(preds, 0, 4)
    preds_discrete = np.round(preds_clipped).astype(int)

    # MAE
    mae = np.mean(np.abs(preds_clipped - labels))

    # Exact accuracy
    exact = np.mean(preds_discrete == labels)

    # Within-1 accuracy
    within1 = np.mean(np.abs(preds_discrete - labels) <= 1)

    # Pearson correlation
    if np.std(preds) > 0 and np.std(labels) > 0:
        pearson = np.corrcoef(preds, labels)[0, 1]
    else:
        pearson = 0.0

    # Spearman correlation
    if np.std(preds) > 0 and np.std(labels) > 0:
        spearman, _ = stats.spearmanr(preds, labels)
    else:
        spearman = 0.0

    return {
        'mae': mae,
        'exact': exact,
        'within1': within1,
        'pearson': pearson,
        'spearman': spearman
    }


# ============================================================
# Main Training Function
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()

    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load Gait data (same as ordinal script)
    data_paths = [
        (os.path.join(Config.DATA_DIR, "gait_train.pkl"), os.path.join(Config.DATA_DIR, "gait_test.pkl")),
        ("./gait_train.pkl", "./gait_test.pkl"),
    ]

    train_path, test_path = None, None
    for tp, vp in data_paths:
        if os.path.exists(tp):
            train_path, test_path = tp, vp
            break

    if train_path is None:
        raise FileNotFoundError(f"Could not find gait data. Tried: {data_paths}")

    print(f"Loading Gait data from: {train_path}")

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    # Combine train and test for cross-validation
    X = np.concatenate([train_data['X'], test_data['X']], axis=0)
    y = np.concatenate([train_data['y'], test_data['y']], axis=0)

    print(f"Original data: {X.shape}")

    # Apply advanced feature engineering
    X = AdvancedFeatureEngineer.engineer_features(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n{'='*60}")
    print("Mamba + Advanced Features (Gait)")
    print(f"Features: {X.shape[2]}")
    print(f"{'='*60}\n")

    # Cross-validation
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{Config.N_FOLDS} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = AugmentedDataset(X_train, y_train, augment=True)
        val_ds = AugmentedDataset(X_val, y_val, augment=False)

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

        model = MambaModel(
            input_dim=X.shape[2],
            hidden_dim=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT
        ).to(device)

        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
        criterion = nn.MSELoss()
        scaler = GradScaler()

        best_mae = float('inf')
        best_metrics = None
        patience_counter = 0

        for epoch in tqdm(range(Config.EPOCHS), desc=f"Fold {fold+1}"):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == Config.EPOCHS - 1:
                preds, labels = evaluate(model, val_loader, device)
                metrics = compute_metrics(preds, labels)

                if metrics['mae'] < best_mae:
                    best_mae = metrics['mae']
                    best_metrics = metrics
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= Config.PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_metrics is None:
            print("  Warning: No metrics computed, using last epoch")
            preds, labels = evaluate(model, val_loader, device)
            best_metrics = compute_metrics(preds, labels)

        fold_results.append(best_metrics)
        print(f"  Best MAE: {best_metrics['mae']:.3f}, Exact: {best_metrics['exact']*100:.1f}%, Pearson: {best_metrics['pearson']:.3f}")

    # Average results
    avg_results = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0].keys()}
    std_results = {k: np.std([r[k] for r in fold_results]) for k in fold_results[0].keys()}

    print(f"\n{'='*60}")
    print("RESULTS - Mamba + Advanced Features (Gait)")
    print(f"{'='*60}")
    print(f"MAE: {avg_results['mae']:.3f} (+/- {std_results['mae']:.3f})")
    print(f"Exact: {avg_results['exact']*100:.1f}% (+/- {std_results['exact']*100:.1f}%)")
    print(f"Within1: {avg_results['within1']*100:.1f}%")
    print(f"Pearson: {avg_results['pearson']:.3f}")
    print(f"Spearman: {avg_results['spearman']:.3f}")

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'MAE':<8} {'Exact':<8} {'Pearson':<8}")
    print("-" * 65)
    print(f"{'Mamba (baseline)':<40} {'0.320':<8} {'77.9%':<8} {'0.778':<8}")
    print(f"{'Mamba + Enhanced':<40} {'0.282':<8} {'78.8%':<8} {'0.804':<8}")
    print(f"{'Mamba + Advanced (this)':<40} {avg_results['mae']:<8.3f} {avg_results['exact']*100:<7.1f}% {avg_results['pearson']:<8.3f}")

    # Save results
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(Config.RESULT_DIR, f"gait_advanced_features_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("Mamba + Advanced Features (Gait)\n")
        f.write("=" * 60 + "\n")
        f.write(f"MAE: {avg_results['mae']:.3f}\n")
        f.write(f"Exact: {avg_results['exact']*100:.1f}%\n")
        f.write(f"Within1: {avg_results['within1']*100:.1f}%\n")
        f.write(f"Pearson: {avg_results['pearson']:.3f}\n")
        f.write(f"Spearman: {avg_results['spearman']:.3f}\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
