"""
ActionMamba: Mamba + GCN Hybrid for Hand Movement Analysis
Based on: "ActionMamba: Action Spatial–Temporal Aggregation Network" (2025)

Key Innovation:
- Combines Mamba (long-range temporal) + GCN (spatial joint correlation)
- ACE (Action Characteristic Encoder) for feature enhancement
- Hand-specific: 21 MediaPipe Hand landmarks (fingers + palm)
- Expected improvement: +5-8% Pearson over pure Mamba

Architecture:
    Input → ACE → [Spatial GCN ⊕ Temporal Mamba] → Fusion → CORAL Loss
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
    LEARNING_RATE = 0.0001  # Reduced from 0.0005 for numerical stability
    WEIGHT_DECAY = 0.02
    PATIENCE = 30
    N_FOLDS = 5

    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    DROPOUT = 0.4
    USE_AMP = True

    NUM_CLASSES = 5  # UPDRS 0-4


# ============================================================
# CORAL Loss for Ordinal Regression
# ============================================================
class CoralLoss(nn.Module):
    """CORAL (Consistent Rank Logits) loss for ordinal regression."""
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
    """Convert ordinal logits to class predictions."""
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).sum(dim=1)
    return predictions


# ============================================================
# Graph Definition (MediaPipe Hands - 21 landmarks)
# ============================================================
def get_hand_graph(num_nodes=21):
    """
    Define skeleton graph for hand movement analysis

    Args:
        num_nodes: Number of joints (21 for hand landmarks)

    MediaPipe Hand Landmarks:
    - 0: WRIST
    - 1-4: THUMB (CMC, MCP, IP, TIP)
    - 5-8: INDEX (MCP, PIP, DIP, TIP)
    - 9-12: MIDDLE (MCP, PIP, DIP, TIP)
    - 13-16: RING (MCP, PIP, DIP, TIP)
    - 17-20: PINKY (MCP, PIP, DIP, TIP)
    """
    # Self-connections
    edges = [(i, i) for i in range(num_nodes)]

    if num_nodes == 21:
        # MediaPipe Hands topology
        body_edges = [
            # Palm to fingers (wrist connections)
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            # Thumb chain: 1-2-3-4
            (1, 2), (2, 3), (3, 4),
            # Index chain: 5-6-7-8
            (5, 6), (6, 7), (7, 8),
            # Middle chain: 9-10-11-12
            (9, 10), (10, 11), (11, 12),
            # Ring chain: 13-14-15-16
            (13, 14), (14, 15), (15, 16),
            # Pinky chain: 17-18-19-20
            (17, 18), (18, 19), (19, 20),
            # Cross connections (palm structure)
            (5, 9), (9, 13), (13, 17),
        ]
    else:
        # Default: Fully connected graph for unknown configurations
        body_edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]

    edges.extend(body_edges)
    edges.extend([(j, i) for i, j in body_edges])  # Bidirectional

    # Build adjacency matrix
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1

    # Normalize adjacency matrix (D^-0.5 * A * D^-0.5)
    D = np.sum(A, axis=1)
    D_inv = np.power(D, -0.5)
    D_inv[np.isinf(D_inv)] = 0
    D_inv = np.diag(D_inv)
    A_norm = D_inv @ A @ D_inv

    return torch.FloatTensor(A_norm)

# ============================================================
# ActionMamba Components
# ============================================================

class ActionCharacteristicEncoder(nn.Module):
    """
    ACE Module: Enhances temporal-spatial coupling in skeletal features
    Based on ActionMamba paper (2025)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        # Temporal enhancement
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Spatial enhancement
        self.spatial_attn = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, T, J, C) - Batch, Time, Joints, Channels
        returns: (B, T, J, C) - Enhanced features
        """
        B, T, J, C = x.shape

        # Temporal enhancement
        x_flat = x.permute(0, 2, 3, 1).reshape(B * J, C, T)  # (B*J, C, T)
        x_temp = self.temporal_conv(x_flat)  # (B*J, C', T)
        x_temp = x_temp.reshape(B, J, -1, T).permute(0, 3, 1, 2)  # (B, T, J, C')

        # Spatial attention
        x_mean = x_temp.mean(dim=1)  # (B, J, C')
        attn = self.spatial_attn(x_mean)  # (B, J, C')
        attn = attn.unsqueeze(1)  # (B, 1, J, C')

        # Apply attention
        x_enhanced = x_temp * attn

        return x_enhanced


class GraphConvolution(nn.Module):
    """Spatial Graph Convolution Layer"""
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = A
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """x: (N, C, T, V) - batch, channels, time, vertices"""
        N, C, T, V = x.size()

        # Graph convolution
        A = self.A.to(x.device)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        x = x.view(N * T, V, C)
        x = torch.matmul(A, x)  # Apply adjacency matrix
        x = x.view(N, T, V, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)

        # 1x1 convolution
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x)


class SpatialGCN(nn.Module):
    """Spatial GCN pathway for joint correlation"""
    def __init__(self, in_channels, hidden_channels, A, dropout=0.4):
        super().__init__()

        self.gcn1 = GraphConvolution(in_channels, hidden_channels, A)
        self.gcn2 = GraphConvolution(hidden_channels, hidden_channels, A)
        self.dropout = nn.Dropout(dropout)

        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool over time

    def forward(self, x):
        """
        x: (B, T, J, C) - Batch, Time, Joints, Channels
        returns: (B, C') - Spatial features
        """
        B, T, J, C = x.shape

        # Reshape to (B, C, T, J) for GCN
        x = x.permute(0, 3, 1, 2)  # (B, C, T, J)

        x = self.gcn1(x)
        x = self.dropout(x)
        x = self.gcn2(x)
        x = self.dropout(x)

        # Pool over time and joints
        x = self.temporal_pool(x)  # (B, C', 1, J)
        x = x.mean(dim=-1).squeeze(-1)  # (B, C')

        return x


class MambaBlock(nn.Module):
    """Mamba SSM Block for temporal modeling"""
    def __init__(self, hidden_size, dropout, state_size=16, expand_factor=2):
        super().__init__()
        self.hidden_size = hidden_size
        expanded = hidden_size * expand_factor

        self.in_proj = nn.Linear(hidden_size, expanded * 2)
        self.conv = nn.Conv1d(expanded, expanded, kernel_size=3, padding=1, groups=expanded)

        self.dt_proj = nn.Linear(expanded, expanded)
        self.B_proj = nn.Linear(expanded, state_size)
        self.C_proj = nn.Linear(expanded, state_size)
        self.D_proj = nn.Linear(expanded, state_size)  # Project input to state dimension

        self.out_proj = nn.Linear(expanded, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.state_size = state_size
        self.expand_factor = expand_factor

    def forward(self, x):
        B, T, D = x.shape

        x_proj = self.in_proj(x)
        x_main, x_gate = x_proj.chunk(2, dim=-1)

        x_conv = x_main.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))  # (B, T, expanded)
        B_t = self.B_proj(x_conv)  # (B, T, state_size)
        C_t = self.C_proj(x_conv)  # (B, T, state_size)
        D_t = self.D_proj(x_conv)  # (B, T, state_size) - project input to state dim

        state = torch.zeros(B, self.state_size, device=x.device)
        outputs = []

        for t in range(T):
            # SSM state update: state = exp(-dt) * state + B * input
            # Clamp to prevent overflow/underflow (numerical stability)
            decay = torch.exp(torch.clamp(-dt[:, t].mean(dim=-1, keepdim=True), min=-10, max=10))
            state = state * decay + B_t[:, t] * D_t[:, t]  # Both (B, state_size)
            # Output: y = C * state
            y_t = (C_t[:, t] * state).sum(dim=-1, keepdim=True)  # (B, 1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, T, 1)
        y = y.expand(-1, -1, x_main.size(-1))  # Expand to match x_main
        y = y * F.silu(x_gate)
        y = self.out_proj(y)
        y = self.dropout(y)

        return x + y


class TemporalMamba(nn.Module):
    """Temporal Mamba pathway for long-range dependencies"""
    def __init__(self, in_channels, hidden_size, num_layers=4, dropout=0.4):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_size)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(hidden_size, dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        x: (B, T, J, C) - Batch, Time, Joints, Channels
        returns: (B, C') - Temporal features
        """
        B, T, J, C = x.shape

        # Flatten joints and channels: (B, T, J*C)
        x = x.reshape(B, T, J * C)

        # Project to hidden size
        x = self.input_proj(x)  # (B, T, H)

        # Apply Mamba layers
        for layer in self.mamba_layers:
            x = layer(x)

        x = self.norm(x)

        # Temporal pooling
        x = x.mean(dim=1)  # (B, H)

        return x


# ============================================================
# ActionMamba Model
# ============================================================
class ActionMamba(nn.Module):
    """
    ActionMamba: Hybrid Mamba + GCN for skeleton-based action recognition

    Based on: "ActionMamba: Action Spatial–Temporal Aggregation Network
               Based on Mamba and GCN for Skeleton-Based Action Recognition" (2025)
    """
    def __init__(self, num_joints, in_channels, num_classes=5,
                 hidden_size=256, num_mamba_layers=4, dropout=0.4):
        super().__init__()

        # Graph adjacency matrix (auto-detect topology based on num_joints)
        self.A = get_hand_graph(num_joints)

        # 1. Action Characteristic Encoder (ACE)
        self.ace = ActionCharacteristicEncoder(
            in_channels=in_channels,
            out_channels=hidden_size // 4,
            kernel_size=3
        )

        # 2. Spatial GCN pathway
        self.spatial_gcn = SpatialGCN(
            in_channels=hidden_size // 4,
            hidden_channels=hidden_size // 2,
            A=self.A,
            dropout=dropout
        )

        # 3. Temporal Mamba pathway
        self.temporal_mamba = TemporalMamba(
            in_channels=num_joints * (hidden_size // 4),
            hidden_size=hidden_size,
            num_layers=num_mamba_layers,
            dropout=dropout
        )

        # 4. Fusion layer
        fusion_input = (hidden_size // 2) + hidden_size  # GCN + Mamba features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes - 1)  # CORAL: num_classes - 1
        )

    def forward(self, x):
        """
        x: (B, T, J, C) - Batch, Time, Joints, Channels
        returns: (B, num_classes-1) - CORAL logits
        """
        # 1. ACE encoding
        x_ace = self.ace(x)  # (B, T, J, C')

        # 2. Spatial pathway (GCN)
        spatial_feat = self.spatial_gcn(x_ace)  # (B, H/2)

        # 3. Temporal pathway (Mamba)
        temporal_feat = self.temporal_mamba(x_ace)  # (B, H)

        # 4. Fusion
        fused = torch.cat([spatial_feat, temporal_feat], dim=-1)  # (B, H + H/2)
        logits = self.fusion(fused)  # (B, num_classes-1)

        return logits


# ============================================================
# Data Augmentation
# ============================================================
class TimeSeriesAugmentation:
    @staticmethod
    def augment(x, p=0.5):
        if np.random.random() < p:
            x = x + np.random.normal(0, 0.03, x.shape)
        if np.random.random() < p:
            x = x * np.random.normal(1, 0.1, (1, x.shape[1], x.shape[2]))
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
# Training Functions
# ============================================================
def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.squeeze().to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(X)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        # Gradient clipping to prevent explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        logits = model(X_tensor)
        preds = ordinal_to_label(logits).cpu().numpy()

        # Expected value for continuous metric
        probs = torch.sigmoid(logits).cpu().numpy()
        expected_values = probs.sum(axis=1)

    mae = np.mean(np.abs(y - preds))
    exact = np.mean(preds == y) * 100
    within1 = np.mean(np.abs(y - preds) <= 1) * 100

    return {'mae': mae, 'exact': exact, 'within1': within1, 'preds': preds, 'expected': expected_values}


def kfold_cv(X, y, config, device):
    print(f"\n{'='*60}")
    print(f"ActionMamba: Mamba + GCN Hybrid (Hand Movement)")
    print(f"Input: {X.shape}, Classes: {config.NUM_CLASSES}")
    print('='*60)

    # Reshape X to (B, T, J, C) if needed
    if len(X.shape) == 3:
        # Assume X is (B, T, J*C), reshape to (B, T, J, C)
        B, T, feat = X.shape
        # Auto-detect joints from features (always 3 coords: x, y, z)
        C = 3
        J = feat // C
        if feat % C != 0:
            raise ValueError(f"Features ({feat}) not divisible by 3. Expected J*3 format.")
        X = X.reshape(B, T, J, C)
        print(f"Reshaped input: {X.shape} (B, T={T}, J={J}, C={C})")

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    y_binned = np.clip(y, 0, 3).astype(int)

    fold_results = []
    all_preds = np.zeros(len(y))
    all_expected = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        print(f"\n--- Fold {fold+1}/{config.N_FOLDS} ---")

        X_train, y_train = X[train_idx], y[train_idx].astype(int)
        X_val, y_val = X[val_idx], y[val_idx].astype(int)

        # Normalize
        B, T, J, C = X_train.shape
        X_train_flat = X_train.reshape(B, T, J * C)
        X_val_flat = X_val.reshape(len(val_idx), T, J * C)

        mean = X_train_flat.mean(axis=(0, 1), keepdims=True)
        std = X_train_flat.std(axis=(0, 1), keepdims=True) + 1e-8

        X_train_norm = ((X_train_flat - mean) / std).reshape(B, T, J, C)
        X_val_norm = ((X_val_flat - mean) / std).reshape(len(val_idx), T, J, C)

        # Model
        model = ActionMamba(
            num_joints=J,
            in_channels=C,
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            num_mamba_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {num_params:,}")

        loader = DataLoader(
            AugmentedDataset(X_train_norm, y_train, True),
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = get_cosine_schedule(optimizer, 10, config.EPOCHS)
        scaler = GradScaler()
        criterion = CoralLoss(config.NUM_CLASSES)

        best_mae, best_state, patience = float('inf'), None, 0

        pbar = tqdm(range(config.EPOCHS), desc=f"Fold {fold+1}")
        for epoch in pbar:
            train_loss = train_epoch(model, loader, criterion, optimizer, scaler, device)
            scheduler.step()

            results = evaluate(model, X_val_norm, y_val, device)
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'mae': f'{results["mae"]:.3f}',
                            'exact': f'{results["exact"]:.1f}%'})

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
        all_expected[val_idx] = results['expected']
        fold_results.append(results)
        print(f"  MAE: {results['mae']:.3f}, Exact: {results['exact']:.1f}%")

        del model
        torch.cuda.empty_cache()

    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_exact = np.mean([r['exact'] for r in fold_results])
    avg_within1 = np.mean([r['within1'] for r in fold_results])

    pearson, _ = stats.pearsonr(y, all_expected)
    spearman, _ = stats.spearmanr(y, all_expected)

    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print('='*60)
    print(f"MAE: {avg_mae:.3f}")
    print(f"Exact: {avg_exact:.1f}%")
    print(f"Within1: {avg_within1:.1f}%")
    print(f"Pearson: {pearson:.3f}")
    print(f"Spearman: {spearman:.3f}")

    return {
        'mae': avg_mae,
        'exact': avg_exact,
        'within1': avg_within1,
        'pearson': pearson,
        'spearman': spearman
    }


def load_data(config):
    for train_path, test_path in [
        (os.path.join(config.DATA_DIR, "hand_movement_train.pkl"), os.path.join(config.DATA_DIR, "hand_movement_test.pkl")),
        ("./hand_movement_train.pkl", "./hand_movement_test.pkl"),
    ]:
        if os.path.exists(train_path):
            break

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    return train_data['X'], train_data['y'], test_data['X'], test_data['y']


def main():
    parser = argparse.ArgumentParser(description='ActionMamba (Hand Movement)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("ACTIONMAMBA: MAMBA + GCN HYBRID (HAND MOVEMENT)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.RESULT_DIR, exist_ok=True)

    X_train, y_train, _, _ = load_data(config)
    print(f"Original data: {X_train.shape}")

    # Use RAW skeleton data (no feature engineering!)
    results = kfold_cv(X_train, y_train, config, device)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(config.RESULT_DIR, f"action_mamba_hand_{timestamp}.txt")
