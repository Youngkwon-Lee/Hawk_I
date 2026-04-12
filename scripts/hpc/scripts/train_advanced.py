"""
PD4T Advanced Training - All Improvements
- Data Augmentation (Time Warping, Jittering, Scaling)
- TCN (Temporal Convolutional Network)
- Ensemble Model
- Better Hyperparameters (epochs 200, cosine annealing)
- Label Smoothing

Usage:
    python scripts/train_advanced.py --model all --epochs 200
    python scripts/train_advanced.py --model ensemble --epochs 200
"""
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from scipy.interpolate import CubicSpline
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Configuration
# ============================================================
class Config:
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    RESULT_DIR = "./results"

    SEQUENCE_LENGTH = 150
    NUM_FEATURES = 10

    # Improved hyperparameters
    BATCH_SIZE = 32  # Smaller batch for better generalization
    EPOCHS = 200
    LEARNING_RATE = 0.0005  # Lower LR
    WEIGHT_DECAY = 0.02
    PATIENCE = 25  # More patience
    WARMUP_EPOCHS = 10

    N_FOLDS = 5

    HIDDEN_SIZE = 256  # Larger model
    NUM_LAYERS = 3
    DROPOUT = 0.4

    USE_AMP = True
    USE_AUGMENTATION = True
    LABEL_SMOOTHING = 0.1


# ============================================================
# Data Augmentation
# ============================================================
class TimeSeriesAugmentation:
    """Time series augmentation techniques"""

    @staticmethod
    def jitter(x, sigma=0.03):
        """Add random noise"""
        return x + np.random.normal(0, sigma, x.shape)

    @staticmethod
    def scaling(x, sigma=0.1):
        """Random magnitude scaling"""
        factor = np.random.normal(1, sigma, (1, x.shape[1]))
        return x * factor

    @staticmethod
    def time_warp(x, sigma=0.2, num_knots=4):
        """Time warping using cubic spline"""
        orig_steps = np.arange(x.shape[0])

        random_warps = np.random.normal(1, sigma, num_knots + 2)
        warp_steps = np.linspace(0, x.shape[0] - 1, num_knots + 2)

        time_warp = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)
        time_warp = np.clip(time_warp, 0, x.shape[0] - 1)

        warped = np.zeros_like(x)
        for i, t in enumerate(time_warp):
            t_floor = int(np.floor(t))
            t_ceil = min(t_floor + 1, x.shape[0] - 1)
            weight = t - t_floor
            warped[i] = (1 - weight) * x[t_floor] + weight * x[t_ceil]

        return warped

    @staticmethod
    def window_slice(x, reduce_ratio=0.9):
        """Random window slicing"""
        target_len = int(x.shape[0] * reduce_ratio)
        if target_len >= x.shape[0]:
            return x
        start = np.random.randint(0, x.shape[0] - target_len)
        return x[start:start + target_len]

    @staticmethod
    def augment(x, p=0.5):
        """Apply random augmentations"""
        if np.random.random() < p:
            x = TimeSeriesAugmentation.jitter(x)
        if np.random.random() < p:
            x = TimeSeriesAugmentation.scaling(x)
        if np.random.random() < p * 0.5:  # Less frequent
            x = TimeSeriesAugmentation.time_warp(x)
        return x


class AugmentedDataset(Dataset):
    """Dataset with online augmentation"""
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]

        if self.augment:
            x = TimeSeriesAugmentation.augment(x, p=0.5)

        return torch.FloatTensor(x), torch.FloatTensor([y])


# ============================================================
# Models
# ============================================================
class AttentionLSTM(nn.Module):
    """Improved Bidirectional LSTM with Multi-Head Attention"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        context = attn_out.mean(dim=1)
        return self.classifier(context).squeeze(-1)


class TransformerModel(nn.Module):
    """Improved Transformer with Pre-LN"""
    def __init__(self, input_size, hidden_size=256, num_layers=6, dropout=0.4, nhead=8):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)


class TCN(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()

        channels = [hidden_size] * num_layers
        layers = []

        for i in range(num_layers):
            in_ch = input_size if i == 0 else channels[i-1]
            out_ch = channels[i]
            dilation = 2 ** i

            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=3,
                                        dilation=dilation, dropout=dropout))

        self.network = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.network(x)
        return self.classifier(x).squeeze(-1)


class TemporalBlock(nn.Module):
    """TCN building block with residual connection"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        # Ensure same length for residual
        if out.size(2) != residual.size(2):
            diff = residual.size(2) - out.size(2)
            out = F.pad(out, (diff // 2, diff - diff // 2))

        return F.gelu(out + residual)


class ConvLSTM(nn.Module):
    """Improved 1D CNN + LSTM"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.classifier(x).squeeze(-1)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models with learned weights"""
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.4):
        super().__init__()

        self.lstm = AttentionLSTM(input_size, hidden_size, num_layers, dropout)
        self.transformer = TransformerModel(input_size, hidden_size, num_layers * 2, dropout)
        self.tcn = TCN(input_size, hidden_size, num_layers + 1, dropout)
        self.convlstm = ConvLSTM(input_size, hidden_size, num_layers, dropout)

        # Learned ensemble weights
        self.weights = nn.Parameter(torch.ones(4) / 4)

    def forward(self, x):
        w = F.softmax(self.weights, dim=0)

        pred_lstm = self.lstm(x)
        pred_transformer = self.transformer(x)
        pred_tcn = self.tcn(x)
        pred_convlstm = self.convlstm(x)

        return (w[0] * pred_lstm + w[1] * pred_transformer +
                w[2] * pred_tcn + w[3] * pred_convlstm)


# ============================================================
# Loss Functions
# ============================================================
class LabelSmoothingMSE(nn.Module):
    """MSE with label smoothing for ordinal regression"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        # Add small noise to targets
        noise = torch.randn_like(target) * self.smoothing
        smoothed_target = target + noise
        smoothed_target = torch.clamp(smoothed_target, 0, 4)
        return F.mse_loss(pred, smoothed_target)


class OrdinalLoss(nn.Module):
    """Ordinal regression loss (CORN)"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        # Add penalty for predictions far from correct ordinal class
        rounded_pred = torch.round(pred)
        ordinal_penalty = F.mse_loss(rounded_pred, target) * 0.1
        return mse + ordinal_penalty


# ============================================================
# Training Functions
# ============================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Cosine annealing with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.squeeze().to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)

        # Batch inference for large datasets
        batch_size = 64
        preds = []
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i+batch_size]
            pred = model(batch).cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds)
        preds = np.clip(preds, 0, 4)

    mae = np.mean(np.abs(y - preds))
    preds_rounded = np.round(preds)
    exact = np.mean(preds_rounded == y) * 100
    within1 = np.mean(np.abs(y - preds_rounded) <= 1) * 100

    return {'mae': mae, 'exact': exact, 'within1': within1, 'preds': preds}


def kfold_cv(X, y, config, device, model_class):
    """K-Fold CV with augmentation"""
    print(f"\n{'='*60}")
    print(f"K-Fold CV (K={config.N_FOLDS}) - {model_class.__name__}")
    print(f"Augmentation: {config.USE_AUGMENTATION}")
    print('='*60)

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    y_binned = np.clip(y, 0, 3).astype(int)

    fold_results = []
    all_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        print(f"\n--- Fold {fold+1}/{config.N_FOLDS} ---")

        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Normalize
        mean = X_train_fold.mean(axis=(0, 1), keepdims=True)
        std = X_train_fold.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_norm = (X_train_fold - mean) / std
        X_val_norm = (X_val_fold - mean) / std

        # Model
        input_size = X.shape[2]
        model = model_class(
            input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ).to(device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        # Dataset with augmentation
        train_dataset = AugmentedDataset(X_train_norm, y_train_fold,
                                         augment=config.USE_AUGMENTATION)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Loss with label smoothing
        criterion = LabelSmoothingMSE(config.LABEL_SMOOTHING) if config.LABEL_SMOOTHING > 0 else nn.MSELoss()

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Cosine annealing with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, config.WARMUP_EPOCHS, config.EPOCHS
        )

        scaler = GradScaler() if config.USE_AMP else None

        best_val_mae = float('inf')
        best_state = None
        patience_counter = 0

        pbar = tqdm(range(config.EPOCHS), desc=f"Fold {fold+1}")
        for epoch in pbar:
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer,
                scaler, device, config.USE_AMP
            )
            scheduler.step()

            # Validate
            val_results = evaluate(model, X_val_norm, y_val_fold, device)
            val_mae = val_results['mae']

            pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'val_mae': f'{val_mae:.3f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        model.to(device)

        results = evaluate(model, X_val_norm, y_val_fold, device)
        all_preds[val_idx] = results['preds']
        fold_results.append(results)

        print(f"  Best MAE: {results['mae']:.3f}, Exact: {results['exact']:.1f}%")

        del model
        torch.cuda.empty_cache()

    # Summary
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_exact = np.mean([r['exact'] for r in fold_results])
    avg_within1 = np.mean([r['within1'] for r in fold_results])
    pearson, _ = stats.pearsonr(y, all_preds)

    print(f"\n{'='*60}")
    print(f"CV Results - {model_class.__name__}")
    print('='*60)
    print(f"MAE: {avg_mae:.3f} (+/- {np.std([r['mae'] for r in fold_results]):.3f})")
    print(f"Exact: {avg_exact:.1f}%")
    print(f"Within1: {avg_within1:.1f}%")
    print(f"Pearson r: {pearson:.3f}")

    return {
        'model_name': model_class.__name__,
        'avg_mae': avg_mae,
        'avg_exact': avg_exact,
        'avg_within1': avg_within1,
        'pearson': pearson
    }


def load_data(config):
    """Load data"""
    print("Loading data...")

    # Try different data locations
    possible_paths = [
        (os.path.join(config.DATA_DIR, "train_data.pkl"),
         os.path.join(config.DATA_DIR, "test_data.pkl")),
        ("./finger_train.pkl", "./finger_test.pkl"),
    ]

    for train_path, test_path in possible_paths:
        if os.path.exists(train_path):
            break

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description='PD4T Advanced Training')
    parser.add_argument('--model', type=str, default='all',
                       choices=['lstm', 'transformer', 'tcn', 'convlstm', 'ensemble', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_aug', action='store_true', help='Disable augmentation')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.USE_AMP = not args.no_amp
    config.USE_AUGMENTATION = not args.no_aug

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print("PD4T ADVANCED TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Hidden Size: {config.HIDDEN_SIZE}")
    print(f"Augmentation: {config.USE_AUGMENTATION}")
    print(f"Label Smoothing: {config.LABEL_SMOOTHING}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(config)

    models = {
        'lstm': AttentionLSTM,
        'transformer': TransformerModel,
        'tcn': TCN,
        'convlstm': ConvLSTM,
        'ensemble': EnsembleModel
    }

    if args.model == 'all':
        model_list = [AttentionLSTM, TransformerModel, TCN, ConvLSTM, EnsembleModel]
    else:
        model_list = [models[args.model]]

    all_results = []

    for model_class in model_list:
        cv_results = kfold_cv(X_train, y_train, config, device, model_class)
        all_results.append(cv_results)
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'MAE':>8} {'Exact':>10} {'Within1':>10} {'Pearson':>10}")
    print("-" * 60)

    for r in sorted(all_results, key=lambda x: x['pearson'], reverse=True):
        print(f"{r['model_name']:<20} {r['avg_mae']:>8.3f} {r['avg_exact']:>9.1f}% {r['avg_within1']:>9.1f}% {r['pearson']:>10.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"advanced_results_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write(f"PD4T Advanced Training Results\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {config.EPOCHS}\n")
        f.write(f"Augmentation: {config.USE_AUGMENTATION}\n\n")

        for r in sorted(all_results, key=lambda x: x['pearson'], reverse=True):
            f.write(f"{r['model_name']}:\n")
            f.write(f"  MAE: {r['avg_mae']:.3f}\n")
            f.write(f"  Exact: {r['avg_exact']:.1f}%\n")
            f.write(f"  Within1: {r['avg_within1']:.1f}%\n")
            f.write(f"  Pearson: {r['pearson']:.3f}\n\n")

    print(f"\nResults saved: {result_path}")
    print("\nTraining Complete!")


if __name__ == "__main__":
    main()
