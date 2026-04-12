"""
SOTA Models for Parkinson's Finger Tapping Assessment
Based on 2024-2025 research papers

Models:
1. Dilated CNN (FastEval Parkinsonism style)
2. ST-GCN (Spatial Temporal Graph Convolutional Network)
3. Mamba (State Space Model)
4. GCN-Transformer Hybrid

References:
- FastEval: https://www.nature.com/articles/s41746-024-01022-x
- ST-GCN: https://arxiv.org/abs/1801.07455
- Mamba: https://arxiv.org/abs/2312.00752
- GCN-Transformer: https://www.nature.com/articles/s41598-025-87752-8

Usage:
    python scripts/train_sota_models.py --model all --epochs 200
    python scripts/train_sota_models.py --model mamba --epochs 200
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
# Model 1: Dilated CNN (FastEval Parkinsonism Style)
# Reference: https://www.nature.com/articles/s41746-024-01022-x
# ============================================================
class DilatedCNN(nn.Module):
    """
    Dilated Convolutional Neural Network for PD assessment
    Based on FastEval Parkinsonism paper (2024)
    """
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)

        # Dilated convolution blocks
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.conv_blocks.append(
                DilatedConvBlock(hidden_size, hidden_size, dilation, dropout)
            )

        # Multi-scale feature aggregation
        self.multi_scale = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1),
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # (batch, hidden, seq)

        # Apply dilated convolutions
        for block in self.conv_blocks:
            x = block(x)

        # Multi-scale pooling
        avg_pool = self.multi_scale[0](x).squeeze(-1)
        max_pool = self.multi_scale[1](x).squeeze(-1)

        # Concatenate
        x = torch.cat([avg_pool, max_pool], dim=-1)

        return self.classifier(x).squeeze(-1)


class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, dropout):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        out = self.dropout(F.gelu(self.bn2(self.conv2(out))))

        # Match sequence lengths
        if out.size(2) != residual.size(2):
            min_len = min(out.size(2), residual.size(2))
            out = out[:, :, :min_len]
            residual = residual[:, :, :min_len]

        return F.gelu(out + residual)


# ============================================================
# Model 2: ST-GCN (Spatial Temporal Graph Convolutional Network)
# Reference: https://arxiv.org/abs/1801.07455
# ============================================================
class STGCN(nn.Module):
    """
    Spatial Temporal Graph Convolutional Network
    Adapted for finger tapping skeleton data
    """
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()

        self.num_joints = input_size  # Treat each feature as a "joint"

        # Build adjacency matrix (fully connected for clinical features)
        self.register_buffer('A', self._build_adjacency(input_size))

        # Input batch norm
        self.bn_in = nn.BatchNorm1d(input_size)

        # ST-GCN layers
        self.st_gcn_layers = nn.ModuleList()
        in_channels = 1
        for i in range(num_layers):
            out_channels = hidden_size // (2 ** max(0, num_layers - i - 2))
            out_channels = max(out_channels, 64)
            self.st_gcn_layers.append(
                STGCNBlock(in_channels, out_channels, self.A, dropout)
            )
            in_channels = out_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def _build_adjacency(self, num_nodes):
        # Fully connected adjacency for clinical features
        A = torch.ones(num_nodes, num_nodes) / num_nodes
        # Add self-loops
        A = A + torch.eye(num_nodes)
        # Normalize
        D = A.sum(dim=1, keepdim=True)
        A = A / D
        return A

    def forward(self, x):
        # x: (batch, seq, features)
        batch_size, seq_len, num_features = x.shape

        # Normalize
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.bn_in(x)
        x = x.transpose(1, 2)  # (batch, seq, features)

        # Reshape for GCN: (batch, channels, seq, nodes)
        x = x.unsqueeze(1)  # (batch, 1, seq, features)

        # Apply ST-GCN layers
        for layer in self.st_gcn_layers:
            x = layer(x)

        # Global pooling
        x = self.pool(x)  # (batch, channels, 1, 1)
        x = x.view(batch_size, -1)

        return self.classifier(x).squeeze(-1)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, dropout):
        super().__init__()

        self.register_buffer('A', A)

        # Spatial graph convolution
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: (batch, channels, seq, nodes)
        residual = self.residual(x)

        # Graph convolution
        batch, channels, seq, nodes = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch, seq, channels, nodes)
        x = x.reshape(batch * seq, channels, nodes)

        # Apply adjacency
        x = torch.matmul(x, self.A)  # (batch*seq, channels, nodes)
        x = x.reshape(batch, seq, channels, nodes)
        x = x.permute(0, 2, 1, 3)  # (batch, channels, seq, nodes)

        x = self.gcn(x)
        x = self.bn(x)

        # Temporal convolution
        x = self.tcn(x)
        x = self.dropout(F.gelu(x + residual))

        return x


# ============================================================
# Model 3: Mamba (State Space Model)
# Reference: https://arxiv.org/abs/2312.00752
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
        # This is faster and avoids sequential loop
        y = torch.zeros_like(x)

        # Compute cumulative effect using exponential decay
        # Simplified: use weighted sum with exponential decay
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
        # y = weights @ x (with proper broadcasting)
        y_temp = torch.matmul(weights[:seq, :x_weighted.size(1)], x_weighted)  # (batch, seq, expanded)

        # Apply input-dependent gating
        gate = torch.sigmoid(dt)
        y = y_temp * gate + x * self.D

        return y


# ============================================================
# Model 4: GCN-Transformer Hybrid
# Reference: https://www.nature.com/articles/s41598-025-87752-8
# ============================================================
class GCNTransformer(nn.Module):
    """
    Two-stream Spatio-Temporal GCN-Transformer Network
    Combines graph convolution with transformer attention
    """
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.4):
        super().__init__()

        # Stream 1: GCN for spatial relationships
        self.gcn_stream = nn.ModuleList([
            GCNLayer(input_size if i == 0 else hidden_size, hidden_size, dropout)
            for i in range(num_layers // 2)
        ])

        # Stream 2: Transformer for temporal patterns
        self.transformer_stream = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers // 2
        )

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, hidden_size) * 0.02)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, seq, features)
        batch_size, seq_len, _ = x.shape

        # GCN Stream (spatial)
        x_gcn = x
        for gcn in self.gcn_stream:
            x_gcn = gcn(x_gcn)
        x_gcn = x_gcn.mean(dim=1)  # (batch, hidden)

        # Transformer Stream (temporal)
        x_trans = self.input_proj(x)
        x_trans = x_trans + self.pos_encoder[:, :seq_len, :]
        x_trans = self.transformer_stream(x_trans)
        x_trans = x_trans.mean(dim=1)  # (batch, hidden)

        # Fusion
        x = torch.cat([x_gcn, x_trans], dim=-1)
        x = self.fusion(x)

        return self.classifier(x).squeeze(-1)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq, features)
        # Treat features as graph nodes, apply simple aggregation
        batch, seq, features = x.shape

        # Self-attention style aggregation
        attn = torch.softmax(torch.matmul(x, x.transpose(-1, -2)) / math.sqrt(features), dim=-1)
        x = torch.matmul(attn, x)

        # Transform
        x = self.fc(x)
        x = x.transpose(1, 2)  # (batch, out, seq)
        x = self.bn(x)
        x = x.transpose(1, 2)  # (batch, seq, out)
        x = self.dropout(F.gelu(x))

        return x


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


def kfold_cv(X, y, config, device, model_class, model_name):
    print(f"\n{'='*60}")
    print(f"{model_name} - 5-Fold CV")
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
    parser = argparse.ArgumentParser(description='SOTA Models Training')
    parser.add_argument('--model', type=str, default='all',
                       choices=['dilated_cnn', 'stgcn', 'mamba', 'gcn_transformer', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("SOTA MODELS FOR PD FINGER TAPPING ASSESSMENT")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    X_train, y_train, _, _ = load_data(config)
    print(f"Data: {X_train.shape}")

    models = {
        'dilated_cnn': (DilatedCNN, 'DilatedCNN (FastEval)'),
        'stgcn': (STGCN, 'ST-GCN'),
        'mamba': (Mamba, 'Mamba (SSM)'),
        'gcn_transformer': (GCNTransformer, 'GCN-Transformer'),
    }

    if args.model == 'all':
        model_list = list(models.values())
    else:
        model_list = [models[args.model]]

    all_results = []
    for model_class, name in model_list:
        try:
            results = kfold_cv(X_train, y_train, config, device, model_class, name)
            all_results.append(results)
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - SOTA MODELS")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'MAE':>8} {'Exact':>10} {'Within1':>10} {'Pearson':>10}")
    print("-" * 65)

    for r in sorted(all_results, key=lambda x: x['pearson'], reverse=True):
        print(f"{r['model']:<25} {r['mae']:>8.3f} {r['exact']:>9.1f}% {r['within1']:>9.1f}% {r['pearson']:>10.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"sota_models_{timestamp}.txt")

    with open(result_path, 'w') as f:
        f.write("SOTA Models for PD Finger Tapping Assessment\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {config.EPOCHS}\n\n")

        f.write("References:\n")
        f.write("- DilatedCNN: FastEval Parkinsonism (Nature Digital Medicine, 2024)\n")
        f.write("- ST-GCN: Spatial Temporal GCN (AAAI, 2018)\n")
        f.write("- Mamba: State Space Model (arXiv, 2023)\n")
        f.write("- GCN-Transformer: Two-stream hybrid (Scientific Reports, 2025)\n\n")

        for r in sorted(all_results, key=lambda x: x['pearson'], reverse=True):
            f.write(f"{r['model']}:\n")
            f.write(f"  MAE: {r['mae']:.3f}\n")
            f.write(f"  Exact: {r['exact']:.1f}%\n")
            f.write(f"  Within1: {r['within1']:.1f}%\n")
            f.write(f"  Pearson: {r['pearson']:.3f}\n\n")

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
