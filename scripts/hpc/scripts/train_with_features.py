"""
Training with Enhanced Features
- Velocity (1st derivative)
- Acceleration (2nd derivative)
- Jerk (3rd derivative)
- Moving statistics (mean, std, min, max)
- Frequency features (FFT)

Usage:
    python scripts/train_with_features.py --model tcn --epochs 200
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
from scipy.fft import rfft
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
    def add_jerk(x):
        """Third derivative (jerk)"""
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        jerk = np.diff(acceleration, axis=0, prepend=acceleration[0:1])
        return jerk

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

        # Stack: (seq_len, 4, n_features) -> (seq_len, 4*n_features)
        stats_array = np.array(stats_features)
        return stats_array.reshape(seq_len, -1)

    @staticmethod
    def add_fft_features(x, n_coeffs=5):
        """FFT magnitude features (top n coefficients)"""
        fft_features = []
        for i in range(x.shape[1]):
            fft = np.abs(rfft(x[:, i]))
            # Take top n coefficients
            top_coeffs = fft[:n_coeffs]
            if len(top_coeffs) < n_coeffs:
                top_coeffs = np.pad(top_coeffs, (0, n_coeffs - len(top_coeffs)))
            fft_features.append(top_coeffs)

        # Broadcast to sequence length
        fft_array = np.array(fft_features).T  # (n_coeffs, n_features)
        fft_broadcast = np.tile(fft_array, (x.shape[0], 1, 1)).reshape(x.shape[0], -1)
        return fft_broadcast

    @staticmethod
    def engineer_features(X, include_velocity=True, include_acceleration=True,
                         include_jerk=False, include_moving_stats=True,
                         include_fft=False):
        """Apply feature engineering to dataset"""
        print("Engineering features...")
        print(f"  Original shape: {X.shape}")

        features_list = [X]  # Start with original features

        if include_velocity:
            velocity = np.array([FeatureEngineer.add_velocity(x) for x in X])
            features_list.append(velocity)
            print(f"  + Velocity: {velocity.shape}")

        if include_acceleration:
            acceleration = np.array([FeatureEngineer.add_acceleration(x) for x in X])
            features_list.append(acceleration)
            print(f"  + Acceleration: {acceleration.shape}")

        if include_jerk:
            jerk = np.array([FeatureEngineer.add_jerk(x) for x in X])
            features_list.append(jerk)
            print(f"  + Jerk: {jerk.shape}")

        if include_moving_stats:
            moving_stats = np.array([FeatureEngineer.add_moving_stats(x) for x in X])
            features_list.append(moving_stats)
            print(f"  + Moving stats: {moving_stats.shape}")

        if include_fft:
            fft_features = np.array([FeatureEngineer.add_fft_features(x) for x in X])
            features_list.append(fft_features)
            print(f"  + FFT: {fft_features.shape}")

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
# Models
# ============================================================
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        out = self.dropout(F.gelu(self.bn2(self.conv2(out))))
        if out.size(2) != res.size(2):
            diff = res.size(2) - out.size(2)
            out = F.pad(out, (diff // 2, diff - diff // 2))
        return F.gelu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()
        # Input projection for larger feature space
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(hidden_size, hidden_size, 3, dilation, dropout))
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
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.network(x)
        return self.classifier(x).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=6, dropout=0.4, nhead=8):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size))
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x).mean(dim=1)
        return self.classifier(x).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(hidden_size * 2, 8, dropout, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.classifier(attn_out.mean(dim=1)).squeeze(-1)


# ============================================================
# Training
# ============================================================
def get_cosine_schedule(optimizer, warmup, total):
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        return max(1e-6, 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (total - warmup))))
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


def kfold_cv(X, y, config, device, model_class, model_name):
    print(f"\n{'='*60}")
    print(f"{model_name} with Enhanced Features - 5-Fold CV")
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

        model = model_class(X.shape[2], config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        loader = DataLoader(AugmentedDataset(X_train_norm, y_train, True),
                           batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = get_cosine_schedule(optimizer, 10, config.EPOCHS)
        scaler = GradScaler()
        criterion = nn.MSELoss()

        best_mae, best_state, patience = float('inf'), None, 0

        pbar = tqdm(range(config.EPOCHS), desc=f"Fold {fold+1}")
        for epoch in pbar:
            train_epoch(model, loader, criterion, optimizer, scaler, device)
            scheduler.step()

            results = evaluate(model, X_val_norm, y_val, device)
            pbar.set_postfix({'mae': f'{results["mae"]:.3f}'})

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
    print(f"RESULTS - {model_name}")
    print('='*60)
    print(f"MAE: {avg_mae:.3f}, Exact: {avg_exact:.1f}%, Within1: {avg_within1:.1f}%, Pearson: {pearson:.3f}")

    return {'model': model_name, 'mae': avg_mae, 'exact': avg_exact, 'within1': avg_within1, 'pearson': pearson}


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='tcn', choices=['tcn', 'transformer', 'lstm', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Load and enhance data
    X_train, y_train, X_test, y_test = load_data(config)
    print(f"Original data: {X_train.shape}")

    # Feature engineering
    X_train_enhanced = FeatureEngineer.engineer_features(
        X_train,
        include_velocity=True,
        include_acceleration=True,
        include_jerk=False,
        include_moving_stats=True,
        include_fft=False
    )

    models = {
        'tcn': (TCN, 'TCN'),
        'transformer': (TransformerModel, 'Transformer'),
        'lstm': (LSTM, 'LSTM'),
    }

    if args.model == 'all':
        model_list = list(models.values())
    else:
        model_list = [models[args.model]]

    all_results = []
    for model_class, name in model_list:
        results = kfold_cv(X_train_enhanced, y_train, config, device, model_class, name)
        all_results.append(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Enhanced Features")
    print('='*60)
    for r in sorted(all_results, key=lambda x: x['pearson'], reverse=True):
        print(f"{r['model']:<15} MAE: {r['mae']:.3f}  Exact: {r['exact']:.1f}%  Pearson: {r['pearson']:.3f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"enhanced_features_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("Training with Enhanced Features\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Original features: 10\n")
        f.write(f"Enhanced features: {X_train_enhanced.shape[2]}\n\n")

        for r in sorted(all_results, key=lambda x: x['pearson'], reverse=True):
            f.write(f"{r['model']}:\n")
            f.write(f"  MAE: {r['mae']:.3f}\n")
            f.write(f"  Exact: {r['exact']:.1f}%\n")
            f.write(f"  Within1: {r['within1']:.1f}%\n")
            f.write(f"  Pearson: {r['pearson']:.3f}\n\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
