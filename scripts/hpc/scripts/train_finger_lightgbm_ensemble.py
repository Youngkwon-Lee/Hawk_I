"""
LightGBM + Mamba Ensemble (ParkTest-style)
Combines:
1. LightGBM on extracted features (ParkTest approach)
2. Mamba on time series
3. Ensemble averaging

ParkTest used LightGBM with 22 handcrafted features and achieved:
- MAE: 0.58
- Pearson: 0.66

Usage:
    python scripts/train_finger_lightgbm_ensemble.py --epochs 200
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
import lightgbm as lgb
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
# ParkTest-style Feature Extraction (Global Features per Video)
# ============================================================
class ParkTestFeatureExtractor:
    """
    Extract global features from entire sequence (ParkTest paper style).
    These are aggregated statistics, not time-series features.
    """

    @staticmethod
    def extract_features(X):
        """
        X: (n_samples, seq_len, n_features)
        Returns: (n_samples, n_extracted_features) - global features
        """
        print("Extracting ParkTest-style global features...")
        all_features = []

        for sample in tqdm(X, desc="Extracting features"):
            features = ParkTestFeatureExtractor._extract_single(sample)
            all_features.append(features)

        return np.array(all_features)

    @staticmethod
    def _extract_single(x):
        """Extract features from a single sample"""
        seq_len, n_features = x.shape
        features = []

        # Compute velocity and acceleration
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        speed = np.sqrt((velocity ** 2).sum(axis=1))
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

        # 1. Speed statistics (most important in ParkTest)
        features.extend([
            np.mean(speed),                    # mean speed
            np.std(speed),                     # std speed
            np.median(speed),                  # median speed
            np.max(speed),                     # max speed
            np.min(speed),                     # min speed
            np.percentile(speed, 75) - np.percentile(speed, 25),  # IQR of speed (strongest predictor!)
            stats.skew(speed),                 # skewness
            stats.kurtosis(speed),             # kurtosis
        ])

        # 2. Amplitude features
        # Compute finger distance (first feature typically represents this)
        amplitude = x[:, 0] if n_features > 0 else speed
        peaks, _ = find_peaks(amplitude, distance=5)

        if len(peaks) > 1:
            peak_amplitudes = amplitude[peaks]
            features.extend([
                np.mean(peak_amplitudes),      # mean peak amplitude
                np.std(peak_amplitudes),       # std peak amplitude
                peak_amplitudes[0] - peak_amplitudes[-1] if len(peak_amplitudes) > 1 else 0,  # amplitude decrement
            ])

            # Decrement ratio (first half vs second half)
            first_half = peak_amplitudes[:len(peak_amplitudes)//2]
            second_half = peak_amplitudes[len(peak_amplitudes)//2:]
            if len(first_half) > 0 and len(second_half) > 0:
                decrement_ratio = (np.mean(first_half) - np.mean(second_half)) / (np.mean(first_half) + 1e-8)
                features.append(decrement_ratio)
            else:
                features.append(0)
        else:
            features.extend([np.mean(np.abs(amplitude)), np.std(np.abs(amplitude)), 0, 0])

        # 3. Rhythm/Period features
        if len(peaks) > 2:
            periods = np.diff(peaks)
            features.extend([
                np.mean(periods),              # mean period
                np.std(periods),               # period std (rhythm variability)
                np.std(periods) / (np.mean(periods) + 1e-8),  # coefficient of variation
                stats.entropy(np.histogram(periods, bins=10)[0] + 1),  # period entropy
            ])
        else:
            features.extend([seq_len / 10, 5, 0.5, 2])  # defaults

        # 4. Frequency features
        taps_per_second = len(peaks) / (seq_len / 30)  # assuming 30 fps
        features.append(taps_per_second)

        # 5. Aperiodicity
        if len(peaks) > 2:
            periods = np.diff(peaks)
            # Linearity of period (how well does linear fit period?)
            x_idx = np.arange(len(periods))
            if len(periods) > 1:
                slope, _, r_value, _, _ = stats.linregress(x_idx, periods)
                features.extend([
                    slope,                     # period slope (fatigue)
                    1 - r_value ** 2,          # aperiodicity (1 - R^2)
                ])
            else:
                features.extend([0, 0.5])
        else:
            features.extend([0, 0.5])

        # 6. Freezing detection
        freeze_threshold = np.percentile(speed, 10)
        freeze_fraction = (speed < freeze_threshold).mean()
        features.append(freeze_fraction)

        # 7. Acceleration features
        acc_magnitude = np.sqrt((acceleration ** 2).sum(axis=1))
        features.extend([
            np.mean(acc_magnitude),
            np.std(acc_magnitude),
            np.max(acc_magnitude),
        ])

        # 8. Per-feature statistics (for each keypoint dimension)
        for f in range(min(n_features, 5)):  # first 5 features
            features.extend([
                np.mean(x[:, f]),
                np.std(x[:, f]),
                np.max(x[:, f]) - np.min(x[:, f]),  # range
            ])

        return np.array(features)


# ============================================================
# Time Series Feature Engineering (for Mamba)
# ============================================================
class FeatureEngineer:
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
        print("Engineering time-series features for Mamba...")
        print(f"  Original shape: {X.shape}")
        features_list = [X]

        velocity = np.array([FeatureEngineer.add_velocity(x) for x in X])
        features_list.append(velocity)

        acceleration = np.array([FeatureEngineer.add_acceleration(x) for x in X])
        features_list.append(acceleration)

        moving_stats = np.array([FeatureEngineer.add_moving_stats(x) for x in X])
        features_list.append(moving_stats)

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


def train_mamba_epoch(model, loader, criterion, optimizer, scaler, device):
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


def evaluate_mamba(model, X, y, device):
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X), 64):
            pred = model(torch.FloatTensor(X[i:i+64]).to(device)).cpu().numpy()
            preds.append(pred)
        preds = np.clip(np.concatenate(preds), 0, 4)
    return preds


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM on global features"""
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    return model


def kfold_cv(X_ts, X_global, y, config, device):
    """
    X_ts: time-series features for Mamba
    X_global: global features for LightGBM
    """
    print(f"\n{'='*60}")
    print(f"LightGBM + Mamba Ensemble")
    print(f"Time-series features: {X_ts.shape[2]}")
    print(f"Global features: {X_global.shape[1]}")
    print('='*60)

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    y_binned = np.clip(y, 0, 3).astype(int)

    fold_results = []
    all_preds_mamba = np.zeros(len(y))
    all_preds_lgb = np.zeros(len(y))
    all_preds_ensemble = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_ts, y_binned)):
        print(f"\n--- Fold {fold+1}/{config.N_FOLDS} ---")

        # Split data
        X_ts_train, y_train = X_ts[train_idx], y[train_idx]
        X_ts_val, y_val = X_ts[val_idx], y[val_idx]

        X_global_train = X_global[train_idx]
        X_global_val = X_global[val_idx]

        # Normalize time-series
        mean = X_ts_train.mean(axis=(0, 1), keepdims=True)
        std = X_ts_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_ts_train_norm = (X_ts_train - mean) / std
        X_ts_val_norm = (X_ts_val - mean) / std

        # Normalize global features
        mean_g = X_global_train.mean(axis=0, keepdims=True)
        std_g = X_global_train.std(axis=0, keepdims=True) + 1e-8
        X_global_train_norm = (X_global_train - mean_g) / std_g
        X_global_val_norm = (X_global_val - mean_g) / std_g

        # Handle NaN/Inf
        X_global_train_norm = np.nan_to_num(X_global_train_norm, nan=0.0, posinf=0.0, neginf=0.0)
        X_global_val_norm = np.nan_to_num(X_global_val_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Train LightGBM ---
        print("  Training LightGBM...")
        lgb_model = train_lightgbm(X_global_train_norm, y_train, X_global_val_norm, y_val)
        preds_lgb = np.clip(lgb_model.predict(X_global_val_norm), 0, 4)

        # --- Train Mamba ---
        print("  Training Mamba...")
        model = Mamba(X_ts.shape[2], config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)

        loader = DataLoader(AugmentedDataset(X_ts_train_norm, y_train, True),
                           batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = get_cosine_schedule(optimizer, 10, config.EPOCHS)
        scaler = GradScaler()
        criterion = nn.MSELoss()

        best_mae, best_state, patience = float('inf'), None, 0

        for epoch in range(config.EPOCHS):
            train_loss = train_mamba_epoch(model, loader, criterion, optimizer, scaler, device)
            scheduler.step()

            preds_mamba_temp = evaluate_mamba(model, X_ts_val_norm, y_val, device)
            mae = np.mean(np.abs(y_val - preds_mamba_temp))

            if mae < best_mae:
                best_mae = mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= config.PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.to(device)
        preds_mamba = evaluate_mamba(model, X_ts_val_norm, y_val, device)

        # --- Ensemble ---
        preds_ensemble = 0.5 * preds_mamba + 0.5 * preds_lgb

        # Store predictions
        all_preds_mamba[val_idx] = preds_mamba
        all_preds_lgb[val_idx] = preds_lgb
        all_preds_ensemble[val_idx] = preds_ensemble

        # Compute metrics
        mae_mamba = np.mean(np.abs(y_val - preds_mamba))
        mae_lgb = np.mean(np.abs(y_val - preds_lgb))
        mae_ensemble = np.mean(np.abs(y_val - preds_ensemble))

        exact_mamba = np.mean(np.round(preds_mamba) == y_val) * 100
        exact_lgb = np.mean(np.round(preds_lgb) == y_val) * 100
        exact_ensemble = np.mean(np.round(preds_ensemble) == y_val) * 100

        print(f"  Mamba:    MAE={mae_mamba:.3f}, Exact={exact_mamba:.1f}%")
        print(f"  LightGBM: MAE={mae_lgb:.3f}, Exact={exact_lgb:.1f}%")
        print(f"  Ensemble: MAE={mae_ensemble:.3f}, Exact={exact_ensemble:.1f}%")

        fold_results.append({
            'mae_mamba': mae_mamba, 'mae_lgb': mae_lgb, 'mae_ensemble': mae_ensemble,
            'exact_mamba': exact_mamba, 'exact_lgb': exact_lgb, 'exact_ensemble': exact_ensemble
        })

        del model
        torch.cuda.empty_cache()

    # Final metrics
    pearson_mamba, _ = stats.pearsonr(y, all_preds_mamba)
    pearson_lgb, _ = stats.pearsonr(y, all_preds_lgb)
    pearson_ensemble, _ = stats.pearsonr(y, all_preds_ensemble)

    spearman_mamba, _ = stats.spearmanr(y, all_preds_mamba)
    spearman_lgb, _ = stats.spearmanr(y, all_preds_lgb)
    spearman_ensemble, _ = stats.spearmanr(y, all_preds_ensemble)

    avg_mae_mamba = np.mean([r['mae_mamba'] for r in fold_results])
    avg_mae_lgb = np.mean([r['mae_lgb'] for r in fold_results])
    avg_mae_ensemble = np.mean([r['mae_ensemble'] for r in fold_results])

    avg_exact_mamba = np.mean([r['exact_mamba'] for r in fold_results])
    avg_exact_lgb = np.mean([r['exact_lgb'] for r in fold_results])
    avg_exact_ensemble = np.mean([r['exact_ensemble'] for r in fold_results])

    within1_mamba = np.mean(np.abs(y - np.round(all_preds_mamba)) <= 1) * 100
    within1_lgb = np.mean(np.abs(y - np.round(all_preds_lgb)) <= 1) * 100
    within1_ensemble = np.mean(np.abs(y - np.round(all_preds_ensemble)) <= 1) * 100

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print('='*60)
    print(f"{'Model':<15} {'MAE':>8} {'Exact':>10} {'Within1':>10} {'Pearson':>10} {'Spearman':>10}")
    print("-" * 65)
    print(f"{'Mamba':<15} {avg_mae_mamba:>8.3f} {avg_exact_mamba:>9.1f}% {within1_mamba:>9.1f}% {pearson_mamba:>10.3f} {spearman_mamba:>10.3f}")
    print(f"{'LightGBM':<15} {avg_mae_lgb:>8.3f} {avg_exact_lgb:>9.1f}% {within1_lgb:>9.1f}% {pearson_lgb:>10.3f} {spearman_lgb:>10.3f}")
    print(f"{'Ensemble':<15} {avg_mae_ensemble:>8.3f} {avg_exact_ensemble:>9.1f}% {within1_ensemble:>9.1f}% {pearson_ensemble:>10.3f} {spearman_ensemble:>10.3f}")

    return {
        'mamba': {'mae': avg_mae_mamba, 'exact': avg_exact_mamba, 'within1': within1_mamba,
                  'pearson': pearson_mamba, 'spearman': spearman_mamba},
        'lgb': {'mae': avg_mae_lgb, 'exact': avg_exact_lgb, 'within1': within1_lgb,
                'pearson': pearson_lgb, 'spearman': spearman_lgb},
        'ensemble': {'mae': avg_mae_ensemble, 'exact': avg_exact_ensemble, 'within1': within1_ensemble,
                     'pearson': pearson_ensemble, 'spearman': spearman_ensemble}
    }


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
    parser = argparse.ArgumentParser(description='LightGBM + Mamba Ensemble')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("LIGHTGBM + MAMBA ENSEMBLE (ParkTest-style)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Load data
    X_train, y_train, _, _ = load_data(config)
    print(f"Original data: {X_train.shape}")

    # Extract global features (ParkTest-style)
    X_global = ParkTestFeatureExtractor.extract_features(X_train)
    print(f"Global features: {X_global.shape}")

    # Extract time-series features (for Mamba)
    X_ts = FeatureEngineer.engineer_features(X_train)

    # Train ensemble
    results = kfold_cv(X_ts, X_global, y_train, config, device)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"finger_lgb_ensemble_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("LightGBM + Mamba Ensemble (ParkTest-style)\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {config.EPOCHS}\n\n")

        f.write("Features:\n")
        f.write(f"  Time-series (Mamba): {X_ts.shape[2]}\n")
        f.write(f"  Global (LightGBM): {X_global.shape[1]}\n\n")

        f.write("Results:\n")
        for name, r in results.items():
            f.write(f"\n{name.upper()}:\n")
            f.write(f"  MAE: {r['mae']:.3f}\n")
            f.write(f"  Exact: {r['exact']:.1f}%\n")
            f.write(f"  Within1: {r['within1']:.1f}%\n")
            f.write(f"  Pearson: {r['pearson']:.3f}\n")
            f.write(f"  Spearman: {r['spearman']:.3f}\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
