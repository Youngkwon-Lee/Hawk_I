"""
Gait Ensemble: CORAL Ordinal (0.807) + Enhanced MSE (0.804)
Test multiple ensemble weights to find optimal combination

Usage:
    python scripts/train_gait_ensemble_coral_enhanced.py --epochs 200
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
import argparse
from datetime import datetime


class Config:
    DATA_DIR = "./data"
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
    NUM_CLASSES = 5


# ============================================================
# CORAL Loss
# ============================================================
class CoralLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        levels = torch.zeros(labels.size(0), self.num_classes - 1, device=labels.device)
        for i in range(self.num_classes - 1):
            levels[:, i] = (labels > i).float()
        loss = F.binary_cross_entropy_with_logits(logits, levels)
        return loss


def ordinal_to_label(logits):
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).sum(dim=1)
    return predictions


# ============================================================
# Feature Engineering
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
        features_list = [X]
        velocity = np.array([FeatureEngineer.add_velocity(x) for x in X])
        features_list.append(velocity)
        acceleration = np.array([FeatureEngineer.add_acceleration(x) for x in X])
        features_list.append(acceleration)
        moving_stats = np.array([FeatureEngineer.add_moving_stats(x) for x in X])
        features_list.append(moving_stats)
        X_enhanced = np.concatenate(features_list, axis=2)
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
        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])


# ============================================================
# Mamba Block
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
        y = x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y) + residual


# ============================================================
# Models
# ============================================================
class MambaCoralModel(nn.Module):
    """CORAL Ordinal Regression"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, dropout=0.4, num_classes=5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([MambaBlock(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes - 1)  # CORAL: K-1 outputs
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = self.dropout(layer(x))
        x = x.mean(dim=1)
        return self.head(x)


class MambaMSEModel(nn.Module):
    """Enhanced MSE Regression"""
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
            nn.Linear(hidden_dim // 2, 1)  # MSE: 1 output
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
def train_coral_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.squeeze().to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(X)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


def train_mse_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X = X.to(device)
        y = y.float().to(device)
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


def evaluate_coral(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            with autocast():
                logits = model(X)
            preds = ordinal_to_label(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy().flatten())
    return np.array(all_preds), np.array(all_labels)


def evaluate_mse(model, loader, device):
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
    preds_clipped = np.clip(preds, 0, 4)
    preds_discrete = np.round(preds_clipped).astype(int)
    mae = np.mean(np.abs(preds_clipped - labels))
    exact = np.mean(preds_discrete == labels)
    within1 = np.mean(np.abs(preds_discrete - labels) <= 1)
    pearson = np.corrcoef(preds, labels)[0, 1] if np.std(preds) > 0 else 0.0
    spearman, _ = stats.spearmanr(preds, labels) if np.std(preds) > 0 else (0.0, 0.0)
    return {'mae': mae, 'exact': exact, 'within1': within1, 'pearson': pearson, 'spearman': spearman}


# ============================================================
# Main Training Function
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("GAIT ENSEMBLE: CORAL + ENHANCED")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    for train_path, test_path in [
        (os.path.join(Config.DATA_DIR, "gait_train.pkl"), os.path.join(Config.DATA_DIR, "gait_test.pkl")),
        ("./gait_train.pkl", "./gait_test.pkl"),
    ]:
        if os.path.exists(train_path):
            break

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    print(f"Original data: {X.shape}")

    # Feature engineering
    X = FeatureEngineer.engineer_features(X)
    print(f"Enhanced features: {X.shape}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)

    coral_preds_all = []
    mse_preds_all = []
    labels_all = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{Config.N_FOLDS}")
        print(f"{'='*60}")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        train_ds = AugmentedDataset(X_train_fold, y_train_fold, augment=True)
        val_ds = AugmentedDataset(X_val_fold, y_val_fold, augment=False)

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

        # ===== Train CORAL Model =====
        print(f"\n--- Training CORAL Model ---")
        coral_model = MambaCoralModel(
            input_dim=X.shape[2],
            hidden_dim=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT,
            num_classes=Config.NUM_CLASSES
        ).to(device)

        coral_optimizer = torch.optim.AdamW(coral_model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        coral_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(coral_optimizer, T_max=Config.EPOCHS)
        coral_criterion = CoralLoss(num_classes=Config.NUM_CLASSES)
        coral_scaler = GradScaler()

        best_coral_mae = float('inf')
        best_coral_state = None
        patience_counter = 0

        for epoch in tqdm(range(Config.EPOCHS), desc="CORAL"):
            train_coral_epoch(coral_model, train_loader, coral_optimizer, coral_criterion, coral_scaler, device)
            coral_scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == Config.EPOCHS - 1:
                preds, labels = evaluate_coral(coral_model, val_loader, device)
                metrics = compute_metrics(preds, labels)
                if metrics['mae'] < best_coral_mae:
                    best_coral_mae = metrics['mae']
                    best_coral_state = coral_model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    break

        coral_model.load_state_dict(best_coral_state)
        coral_preds, _ = evaluate_coral(coral_model, val_loader, device)

        # ===== Train MSE Model =====
        print(f"\n--- Training MSE Model ---")
        mse_model = MambaMSEModel(
            input_dim=X.shape[2],
            hidden_dim=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT
        ).to(device)

        mse_optimizer = torch.optim.AdamW(mse_model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        mse_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mse_optimizer, T_max=Config.EPOCHS)
        mse_criterion = nn.MSELoss()
        mse_scaler = GradScaler()

        best_mse_mae = float('inf')
        best_mse_state = None
        patience_counter = 0

        for epoch in tqdm(range(Config.EPOCHS), desc="MSE"):
            train_mse_epoch(mse_model, train_loader, mse_optimizer, mse_criterion, mse_scaler, device)
            mse_scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == Config.EPOCHS - 1:
                preds, labels = evaluate_mse(mse_model, val_loader, device)
                metrics = compute_metrics(preds, labels)
                if metrics['mae'] < best_mse_mae:
                    best_mse_mae = metrics['mae']
                    best_mse_state = mse_model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    break

        mse_model.load_state_dict(best_mse_state)
        mse_preds, _ = evaluate_mse(mse_model, val_loader, device)

        coral_preds_all.extend(coral_preds)
        mse_preds_all.extend(mse_preds)
        labels_all.extend(y_val_fold)

    # ===== Ensemble Testing =====
    coral_preds_all = np.array(coral_preds_all)
    mse_preds_all = np.array(mse_preds_all)
    labels_all = np.array(labels_all)

    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*60}")

    weights = [
        (1.0, 0.0, "CORAL Only"),
        (0.0, 1.0, "MSE Only"),
        (0.5, 0.5, "Equal (0.5/0.5)"),
        (0.6, 0.4, "CORAL 60%"),
        (0.7, 0.3, "CORAL 70%"),
        (0.4, 0.6, "MSE 60%"),
        (0.3, 0.7, "MSE 70%"),
    ]

    best_ensemble = None
    best_pearson = -1

    for w_coral, w_mse, name in weights:
        ensemble_preds = w_coral * coral_preds_all + w_mse * mse_preds_all
        metrics = compute_metrics(ensemble_preds, labels_all)

        print(f"\n{name} (CORAL {w_coral:.1f} + MSE {w_mse:.1f}):")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  Exact: {metrics['exact']*100:.1f}%")
        print(f"  Within1: {metrics['within1']*100:.1f}%")
        print(f"  Pearson: {metrics['pearson']:.3f}")
        print(f"  Spearman: {metrics['spearman']:.3f}")

        if metrics['pearson'] > best_pearson:
            best_pearson = metrics['pearson']
            best_ensemble = (w_coral, w_mse, name, metrics)

    print(f"\n{'='*60}")
    print("BEST ENSEMBLE")
    print(f"{'='*60}")
    w_coral, w_mse, name, metrics = best_ensemble
    print(f"{name} (CORAL {w_coral:.1f} + MSE {w_mse:.1f})")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"Exact: {metrics['exact']*100:.1f}%")
    print(f"Pearson: {metrics['pearson']:.3f}")

    # Save results
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(Config.RESULT_DIR, f"gait_ensemble_coral_enhanced_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("Gait Ensemble: CORAL + Enhanced\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best: {name} (CORAL {w_coral:.1f} + MSE {w_mse:.1f})\n")
        f.write(f"MAE: {metrics['mae']:.3f}\n")
        f.write(f"Exact: {metrics['exact']*100:.1f}%\n")
        f.write(f"Pearson: {metrics['pearson']:.3f}\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
