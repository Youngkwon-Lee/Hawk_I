"""
Mamba + Clinical Features V1 (Quick Wins)
Adding 4 critical Parkinson's-specific features

New Features:
1. amplitude_decline_rate - Fatigue indicator
2. frequency_variability - Rhythm consistency
3. smoothness_sparc - Movement quality
4. hesitation_count - Freezing detection

Expected: Pearson 0.609 → 0.64 (+5%)

Usage:
    python scripts/train_mamba_clinical_v1.py --epochs 200
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
from scipy.signal import find_peaks, welch
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
# Clinical Feature Engineering
# ============================================================
class ClinicalFeatureEngineer:
    """Parkinson's Disease-specific clinical features"""

    @staticmethod
    def calculate_sparc(velocity, fs=30):
        """
        Spectral Arc Length (SPARC) - Movement smoothness
        Lower value = smoother movement
        """
        # Velocity magnitude
        if len(velocity.shape) > 1:
            vel_mag = np.linalg.norm(velocity, axis=1)
        else:
            vel_mag = velocity

        if len(vel_mag) < 10:
            return 0.0

        # Welch periodogram
        f, Pxx = welch(vel_mag, fs=fs, nperseg=min(256, len(vel_mag)))

        # Normalize spectrum
        Pxx_norm = Pxx / (np.sum(Pxx) + 1e-8)

        # Calculate arc length
        df = np.diff(f)
        dPxx = np.diff(Pxx_norm)

        arc_length = -np.sum(np.sqrt(df**2 + dPxx**2))

        return arc_length

    @staticmethod
    def calculate_amplitude_decline(positions, thumb_idx=4, index_idx=8):
        """
        Amplitude decline rate - Fatigue indicator
        Measures decrease in movement amplitude over time
        """
        # Calculate thumb-index distance per frame
        if len(positions.shape) == 2:
            # Assuming flattened features, reconstruct
            return 0.0

        distances = np.linalg.norm(
            positions[:, thumb_idx] - positions[:, index_idx],
            axis=-1
        )

        # Find peaks (maximum openings)
        peaks, _ = find_peaks(distances, distance=5)

        if len(peaks) < 3:
            return 0.0

        # Linear regression on peak amplitudes
        peak_amps = distances[peaks]
        x = np.arange(len(peak_amps))

        # Slope = decline rate
        if len(peak_amps) > 1:
            slope = np.polyfit(x, peak_amps, 1)[0]
            return slope
        else:
            return 0.0

    @staticmethod
    def calculate_frequency_variability(velocity, fs=30):
        """
        Frequency variability - Rhythm consistency
        Lower = more consistent rhythm
        """
        # Velocity magnitude
        if len(velocity.shape) > 1:
            vel_mag = np.linalg.norm(velocity, axis=1)
        else:
            vel_mag = velocity

        if len(vel_mag) < 10:
            return 0.0

        # FFT
        fft = np.fft.rfft(vel_mag)
        freqs = np.fft.rfftfreq(len(vel_mag), 1/fs)

        # Power spectrum
        power = np.abs(fft)**2

        # Find dominant frequency
        dominant_idx = np.argmax(power[1:]) + 1  # Skip DC
        dominant_freq = freqs[dominant_idx]

        # Coefficient of variation around dominant frequency
        window = 5
        start = max(0, dominant_idx - window)
        end = min(len(power), dominant_idx + window + 1)

        freq_window = power[start:end]
        cv = np.std(freq_window) / (np.mean(freq_window) + 1e-8)

        return cv

    @staticmethod
    def detect_hesitations(velocity, threshold=0.05):
        """
        Hesitation detection - Freezing episodes
        Returns: (number of episodes, fraction of time hesitating)
        """
        # Velocity magnitude
        if len(velocity.shape) > 1:
            vel_mag = np.linalg.norm(velocity, axis=1)
        else:
            vel_mag = velocity

        # Below threshold = hesitation
        hesitations = vel_mag < threshold

        # Count episodes using connected components
        from scipy.ndimage import label
        labeled, num_episodes = label(hesitations)

        # Fraction of time
        hesitation_fraction = np.sum(hesitations) / len(hesitations)

        return num_episodes, hesitation_fraction

    @staticmethod
    def add_clinical_features(X):
        """
        Add all clinical features to dataset
        Input: (N, seq_len, features)
        Output: (N, seq_len, features + 4 clinical features)
        """
        print("Adding clinical features...")
        print(f"  Original shape: {X.shape}")

        N, seq_len, n_feat = X.shape

        # Calculate velocity (for all samples)
        velocity = np.diff(X, axis=1, prepend=X[:, 0:1, :])

        clinical_features = []

        for i in range(N):
            # 1. SPARC (smoothness)
            sparc = ClinicalFeatureEngineer.calculate_sparc(velocity[i])

            # 2. Amplitude decline
            # Note: This requires 3D positions, approximation using variance
            amp_decline = -np.std(np.linalg.norm(velocity[i], axis=1))

            # 3. Frequency variability
            freq_var = ClinicalFeatureEngineer.calculate_frequency_variability(velocity[i])

            # 4. Hesitations
            num_hesitations, hesitation_frac = ClinicalFeatureEngineer.detect_hesitations(velocity[i])

            # Broadcast to sequence length
            clinical_feat = np.array([
                sparc,
                amp_decline,
                freq_var,
                hesitation_frac
            ])

            # Repeat for each frame
            clinical_feat_seq = np.tile(clinical_feat, (seq_len, 1))

            clinical_features.append(clinical_feat_seq)

        clinical_features = np.array(clinical_features)
        print(f"  + Clinical features: {clinical_features.shape}")

        # Concatenate
        X_clinical = np.concatenate([X, clinical_features], axis=2)
        print(f"  Final shape: {X_clinical.shape}")

        return X_clinical


# ============================================================
# Enhanced Feature Engineering (from previous)
# ============================================================
class FeatureEngineer:
    """Standard enhanced features"""

    @staticmethod
    def add_velocity(x):
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        return velocity

    @staticmethod
    def add_acceleration(x):
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        return acceleration

    @staticmethod
    def add_moving_stats(x, window=5):
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
        print("Engineering standard features...")
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
        print(f"  Enhanced shape: {X_enhanced.shape}")

        return X_enhanced


# ============================================================
# Data Augmentation & Dataset
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
# Mamba Model (same as before)
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
    print(f"Mamba + Clinical Features V1 - 5-Fold CV")
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

    print(f"\n{'='*60}")
    print(f"RESULTS - Mamba + Clinical V1")
    print('='*60)
    print(f"MAE: {avg_mae:.3f}")
    print(f"Exact: {avg_exact:.1f}%")
    print(f"Within1: {avg_within1:.1f}%")
    print(f"Pearson: {pearson:.3f}")

    return {'mae': avg_mae, 'exact': avg_exact, 'within1': avg_within1, 'pearson': pearson}


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
    parser = argparse.ArgumentParser(description='Mamba + Clinical Features V1')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("MAMBA + CLINICAL FEATURES V1 (Quick Wins)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Load data
    X_train, y_train, _, _ = load_data(config)
    print(f"Original data: {X_train.shape}")

    # Standard feature engineering
    X_enhanced = FeatureEngineer.engineer_features(X_train)

    # Add clinical features
    X_clinical = ClinicalFeatureEngineer.add_clinical_features(X_enhanced)

    # Train
    results = kfold_cv(X_clinical, y_train, config, device)

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print('='*60)
    print(f"{'Model':<30} {'MAE':>8} {'Exact':>10} {'Pearson':>10}")
    print("-" * 60)
    print(f"{'Mamba + Enhanced':<30} {'0.444':>8} {'63.0%':>10} {'0.609':>10}")
    print(f"{'Mamba + Clinical V1':<30} {results['mae']:>8.3f} {results['exact']:>9.1f}% {results['pearson']:>10.3f}")

    improvement = (results['pearson'] - 0.609) / 0.609 * 100
    print(f"\nImprovement: {improvement:+.1f}% (Pearson)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"mamba_clinical_v1_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("Mamba + Clinical Features V1\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {config.EPOCHS}\n\n")

        f.write("Clinical Features Added:\n")
        f.write("  1. SPARC smoothness\n")
        f.write("  2. Amplitude decline rate\n")
        f.write("  3. Frequency variability\n")
        f.write("  4. Hesitation fraction\n\n")

        f.write(f"Total features: {X_clinical.shape[2]}\n\n")

        f.write("Results:\n")
        f.write(f"  MAE: {results['mae']:.3f}\n")
        f.write(f"  Exact: {results['exact']:.1f}%\n")
        f.write(f"  Within1: {results['within1']:.1f}%\n")
        f.write(f"  Pearson: {results['pearson']:.3f}\n\n")

        f.write("Comparison:\n")
        f.write(f"  Mamba + Enhanced: MAE=0.444, Exact=63.0%, Pearson=0.609\n")
        f.write(f"  Mamba + Clinical V1: MAE={results['mae']:.3f}, Exact={results['exact']:.1f}%, Pearson={results['pearson']:.3f}\n")
        f.write(f"  Improvement: {improvement:+.1f}% (Pearson)\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
