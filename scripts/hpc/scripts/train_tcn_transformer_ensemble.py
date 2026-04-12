"""
TCN + Transformer Ensemble (Top 2 Models Only)
Lighter ensemble for better generalization

Usage:
    python scripts/train_tcn_transformer_ensemble.py --epochs 200
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
    WARMUP_EPOCHS = 10

    N_FOLDS = 5
    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    DROPOUT = 0.4
    USE_AMP = True


# ============================================================
# Data Augmentation
# ============================================================
class TimeSeriesAugmentation:
    @staticmethod
    def jitter(x, sigma=0.03):
        return x + np.random.normal(0, sigma, x.shape)

    @staticmethod
    def scaling(x, sigma=0.1):
        factor = np.random.normal(1, sigma, (1, x.shape[1]))
        return x * factor

    @staticmethod
    def time_warp(x, sigma=0.2, num_knots=4):
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
    def augment(x, p=0.5):
        if np.random.random() < p:
            x = TimeSeriesAugmentation.jitter(x)
        if np.random.random() < p:
            x = TimeSeriesAugmentation.scaling(x)
        if np.random.random() < p * 0.3:
            x = TimeSeriesAugmentation.time_warp(x)
        return x


class AugmentedDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = TimeSeriesAugmentation.augment(x, p=0.5)
        return torch.FloatTensor(x), torch.FloatTensor([self.y[idx]])


# ============================================================
# Models
# ============================================================
class TemporalBlock(nn.Module):
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
        if out.size(2) != residual.size(2):
            diff = residual.size(2) - out.size(2)
            out = F.pad(out, (diff // 2, diff - diff // 2))
        return F.gelu(out + residual)


class TCN(nn.Module):
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
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        return x


class TransformerEncoder(nn.Module):
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
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        return x.mean(dim=1)


class TCNTransformerEnsemble(nn.Module):
    """
    Ensemble: TCN + Transformer with learned weights
    - TCN: Good at local patterns
    - Transformer: Good at global dependencies
    """
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()

        # TCN branch
        self.tcn = TCN(input_size, hidden_size, num_layers, dropout)

        # Transformer branch
        self.transformer = TransformerEncoder(input_size, hidden_size, num_layers + 2, dropout)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learned ensemble weights
        self.ensemble_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

        # Separate heads
        self.tcn_head = nn.Linear(hidden_size, 1)
        self.transformer_head = nn.Linear(hidden_size, 1)

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Extract features
        tcn_feat = self.tcn(x)
        transformer_feat = self.transformer(x)

        # Individual predictions
        tcn_pred = self.tcn_head(tcn_feat).squeeze(-1)
        transformer_pred = self.transformer_head(transformer_feat).squeeze(-1)

        # Weighted ensemble
        w = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = w[0] * tcn_pred + w[1] * transformer_pred

        # Fusion prediction
        fused = self.fusion(torch.cat([tcn_feat, transformer_feat], dim=-1))
        fusion_pred = self.fusion_head(fused).squeeze(-1)

        # Final: average of ensemble and fusion
        return (ensemble_pred + fusion_pred) / 2


# ============================================================
# Training
# ============================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(1e-6, 0.5 * (1 + np.cos(np.pi * progress)))
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
        preds = []
        for i in range(0, len(X), 64):
            batch = torch.FloatTensor(X[i:i+64]).to(device)
            pred = model(batch).cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds)
        preds = np.clip(preds, 0, 4)

    mae = np.mean(np.abs(y - preds))
    exact = np.mean(np.round(preds) == y) * 100
    within1 = np.mean(np.abs(y - np.round(preds)) <= 1) * 100
    return {'mae': mae, 'exact': exact, 'within1': within1, 'preds': preds}


def kfold_cv(X, y, config, device):
    print(f"\n{'='*60}")
    print("TCN + Transformer Ensemble - 5-Fold CV")
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

        mean = X_train_fold.mean(axis=(0, 1), keepdims=True)
        std = X_train_fold.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_norm = (X_train_fold - mean) / std
        X_val_norm = (X_val_fold - mean) / std

        model = TCNTransformerEnsemble(
            X.shape[2], config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        train_dataset = AugmentedDataset(X_train_norm, y_train_fold, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                       weight_decay=config.WEIGHT_DECAY)
        scheduler = get_cosine_schedule_with_warmup(optimizer, config.WARMUP_EPOCHS, config.EPOCHS)
        scaler = GradScaler() if config.USE_AMP else None

        best_val_mae = float('inf')
        best_state = None
        patience_counter = 0

        pbar = tqdm(range(config.EPOCHS), desc=f"Fold {fold+1}")
        for epoch in pbar:
            train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, config.USE_AMP)
            scheduler.step()

            val_results = evaluate(model, X_val_norm, y_val_fold, device)
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'val_mae': f'{val_results["mae"]:.3f}'})

            if val_results['mae'] < best_val_mae:
                best_val_mae = val_results['mae']
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

        # Print ensemble weights
        w = F.softmax(model.ensemble_weights, dim=0)
        print(f"  Ensemble weights - TCN: {w[0]:.3f}, Transformer: {w[1]:.3f}")
        print(f"  MAE: {results['mae']:.3f}, Exact: {results['exact']:.1f}%")

        del model
        torch.cuda.empty_cache()

    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_exact = np.mean([r['exact'] for r in fold_results])
    avg_within1 = np.mean([r['within1'] for r in fold_results])
    pearson, _ = stats.pearsonr(y, all_preds)

    print(f"\n{'='*60}")
    print("RESULTS - TCN + Transformer Ensemble")
    print('='*60)
    print(f"MAE: {avg_mae:.3f}")
    print(f"Exact: {avg_exact:.1f}%")
    print(f"Within1: {avg_within1:.1f}%")
    print(f"Pearson: {pearson:.3f}")

    return {'avg_mae': avg_mae, 'avg_exact': avg_exact, 'avg_within1': avg_within1, 'pearson': pearson}


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
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    X_train, y_train, _, _ = load_data(config)
    print(f"Data: {X_train.shape}")

    results = kfold_cv(X_train, y_train, config, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"tcn_transformer_ensemble_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write(f"TCN + Transformer Ensemble Results\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Epochs: {config.EPOCHS}\n\n")
        f.write(f"MAE: {results['avg_mae']:.3f}\n")
        f.write(f"Exact: {results['avg_exact']:.1f}%\n")
        f.write(f"Within1: {results['avg_within1']:.1f}%\n")
        f.write(f"Pearson: {results['pearson']:.3f}\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
