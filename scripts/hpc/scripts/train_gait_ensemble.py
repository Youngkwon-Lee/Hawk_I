"""
Gait Ensemble: Mamba + Enhanced + Clinical
Combines all features for maximum performance

Usage:
    python train_gait_ensemble.py --epochs 200
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
from scipy.signal import welch, find_peaks
from scipy.ndimage import label
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
# Feature Engineering (All features combined)
# ============================================================
class FeatureEngineer:
    @staticmethod
    def add_velocity(x):
        return np.diff(x, axis=0, prepend=x[0:1])

    @staticmethod
    def add_acceleration(x):
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        return np.diff(velocity, axis=0, prepend=velocity[0:1])

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
        print("Engineering enhanced features...")
        print(f"  Original shape: {X.shape}")
        features_list = [X]

        velocity = np.array([FeatureEngineer.add_velocity(x) for x in X])
        features_list.append(velocity)

        acceleration = np.array([FeatureEngineer.add_acceleration(x) for x in X])
        features_list.append(acceleration)

        moving_stats = np.array([FeatureEngineer.add_moving_stats(x) for x in X])
        features_list.append(moving_stats)

        X_enhanced = np.concatenate(features_list, axis=2)
        print(f"  Enhanced shape: {X_enhanced.shape}")
        return X_enhanced


class ClinicalFeatureEngineer:
    """Combined clinical features for Gait"""

    @staticmethod
    def calculate_sparc(velocity, fs=30):
        if len(velocity.shape) > 1:
            vel_mag = np.linalg.norm(velocity, axis=1)
        else:
            vel_mag = velocity
        if len(vel_mag) < 10:
            return 0.0
        f, Pxx = welch(vel_mag, fs=fs, nperseg=min(256, len(vel_mag)))
        Pxx_norm = Pxx / (np.sum(Pxx) + 1e-8)
        df = np.diff(f)
        dPxx = np.diff(Pxx_norm)
        arc_length = -np.sum(np.sqrt(df**2 + dPxx**2))
        return arc_length

    @staticmethod
    def calculate_gait_symmetry(velocity):
        if len(velocity.shape) > 1:
            vel_mag = np.linalg.norm(velocity, axis=1)
        else:
            vel_mag = velocity
        mid = len(vel_mag) // 2
        left = vel_mag[:mid]
        right = vel_mag[mid:mid*2]
        if len(left) == 0 or len(right) == 0:
            return 1.0
        symmetry = min(np.mean(left), np.mean(right)) / (max(np.mean(left), np.mean(right)) + 1e-8)
        return symmetry

    @staticmethod
    def calculate_stride_variability(velocity, fs=30):
        if len(velocity.shape) > 1:
            vel_mag = np.linalg.norm(velocity, axis=1)
        else:
            vel_mag = velocity
        if len(vel_mag) < 20:
            return 0.0
        peaks, _ = find_peaks(vel_mag, distance=int(fs * 0.3))
        if len(peaks) < 3:
            return 0.0
        intervals = np.diff(peaks)
        cv = np.std(intervals) / (np.mean(intervals) + 1e-8)
        return cv

    @staticmethod
    def detect_freezing(velocity, threshold=0.03):
        if len(velocity.shape) > 1:
            vel_mag = np.linalg.norm(velocity, axis=1)
        else:
            vel_mag = velocity
        freezing = vel_mag < threshold
        labeled, num_episodes = label(freezing)
        freezing_fraction = np.sum(freezing) / len(freezing)
        return num_episodes, freezing_fraction

    @staticmethod
    def calculate_frequency_variability(velocity, fs=30):
        if len(velocity.shape) > 1:
            vel_mag = np.linalg.norm(velocity, axis=1)
        else:
            vel_mag = velocity
        if len(vel_mag) < 10:
            return 0.0
        fft = np.fft.rfft(vel_mag)
        freqs = np.fft.rfftfreq(len(vel_mag), 1/fs)
        power = np.abs(fft)**2
        dominant_idx = np.argmax(power[1:]) + 1
        window = 5
        start = max(0, dominant_idx - window)
        end = min(len(power), dominant_idx + window + 1)
        freq_window = power[start:end]
        cv = np.std(freq_window) / (np.mean(freq_window) + 1e-8)
        return cv

    @staticmethod
    def add_clinical_features(X):
        print("Adding gait ensemble clinical features...")
        N, seq_len, n_feat = X.shape
        velocity = np.diff(X, axis=1, prepend=X[:, 0:1, :])
        clinical_features = []

        for i in range(N):
            # 1. SPARC (smoothness)
            sparc = ClinicalFeatureEngineer.calculate_sparc(velocity[i])

            # 2. Gait symmetry
            symmetry = ClinicalFeatureEngineer.calculate_gait_symmetry(velocity[i])

            # 3. Stride variability
            stride_var = ClinicalFeatureEngineer.calculate_stride_variability(velocity[i])

            # 4. Freezing of gait
            num_freezing, freezing_frac = ClinicalFeatureEngineer.detect_freezing(velocity[i])

            # 5. Frequency variability
            freq_var = ClinicalFeatureEngineer.calculate_frequency_variability(velocity[i])

            # 6. Amplitude decline (from velocity std)
            amp_decline = -np.std(np.linalg.norm(velocity[i], axis=1))

            clinical_feat = np.array([sparc, symmetry, stride_var, freezing_frac, freq_var, amp_decline])
            clinical_feat_seq = np.tile(clinical_feat, (seq_len, 1))
            clinical_features.append(clinical_feat_seq)

        clinical_features = np.array(clinical_features)
        print(f"  + Clinical features (6): {clinical_features.shape}")
        X_clinical = np.concatenate([X, clinical_features], axis=2)
        print(f"  Final shape: {X_clinical.shape}")
        return X_clinical


# ============================================================
# Dataset & Augmentation
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
# Mamba Model
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
    print(f"Gait Ensemble (Enhanced + Clinical) - 5-Fold CV")
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
    print(f"RESULTS - Gait Ensemble")
    print('='*60)
    print(f"MAE: {avg_mae:.3f}")
    print(f"Exact: {avg_exact:.1f}%")
    print(f"Within1: {avg_within1:.1f}%")
    print(f"Pearson: {pearson:.3f}")
    print(f"Spearman: {spearman:.3f}")
    return {'mae': avg_mae, 'exact': avg_exact, 'within1': avg_within1, 'pearson': pearson, 'spearman': spearman}


def load_gait_data():
    possible_paths = [
        ("./gait_train_v4.pkl", "./gait_test_v4.pkl"),
        ("./gait_train.pkl", "./gait_test.pkl"),
    ]
    for train_path, test_path in possible_paths:
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"Loading: {train_path}")
            with open(train_path, 'rb') as f:
                train_data = pickle.load(f)
            with open(test_path, 'rb') as f:
                test_data = pickle.load(f)
            return train_data['X'], train_data['y'], test_data['X'], test_data['y']
    raise FileNotFoundError("No gait data files found!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print("GAIT ENSEMBLE (Enhanced + Clinical)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    X_train, y_train, _, _ = load_gait_data()
    print(f"Gait data: {X_train.shape}")

    # Enhanced features
    X_enhanced = FeatureEngineer.engineer_features(X_train)

    # Clinical features (6 features)
    X_ensemble = ClinicalFeatureEngineer.add_clinical_features(X_enhanced)

    results = kfold_cv(X_ensemble, y_train, config, device)

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON (Gait Task)")
    print('='*60)
    print(f"{'Model':<30} {'MAE':>8} {'Exact':>10} {'Pearson':>10}")
    print("-" * 60)
    print(f"{'Gait + Enhanced':<30} {'0.335':>8} {'71.9%':>10} {'0.804':>10}")
    print(f"{'Gait Ensemble':<30} {results['mae']:>8.3f} {results['exact']:>9.1f}% {results['pearson']:>10.3f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"gait_ensemble_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("Gait Ensemble (Enhanced + Clinical)\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {config.EPOCHS}\n\n")
        f.write("Features:\n")
        f.write("  Enhanced: velocity, acceleration, moving_stats\n")
        f.write("  Clinical (6): SPARC, symmetry, stride_var, freezing, freq_var, amp_decline\n")
        f.write(f"  Total: {X_ensemble.shape[2]}\n\n")
        f.write("Results:\n")
        f.write(f"  MAE: {results['mae']:.3f}\n")
        f.write(f"  Exact: {results['exact']:.1f}%\n")
        f.write(f"  Within1: {results['within1']:.1f}%\n")
        f.write(f"  Pearson: {results['pearson']:.3f}\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
