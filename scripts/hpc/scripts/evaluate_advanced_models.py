"""
Advanced Models for UPDRS Scoring - GroupKFold CV
최신 모델들: ST-GCN, Mamba, MS-TCN, PatchTST + 고급 증강

Usage:
    python scripts/evaluate_advanced_models.py --task gait
    python scripts/evaluate_advanced_models.py --task finger
    python scripts/evaluate_advanced_models.py --task all
"""
import os
import pickle
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 20
N_FOLDS = 5


def extract_subject_from_id(video_id, task='finger'):
    video_id = video_id.replace('.mp4', '')
    subject = video_id.rsplit('_', 1)[-1]
    return subject


# ============================================================
# Advanced Data Augmentation
# ============================================================
class AdvancedAugmentation:
    @staticmethod
    def time_warp(x, sigma=0.2):
        """시간축 워핑"""
        seq_len = x.shape[0]
        orig_steps = np.arange(seq_len)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(seq_len,))
        warp_steps = np.cumsum(random_warps)
        warp_steps = (warp_steps - warp_steps.min()) / (warp_steps.max() - warp_steps.min()) * (seq_len - 1)
        warped = np.zeros_like(x)
        for i in range(x.shape[1]):
            warped[:, i] = np.interp(orig_steps, warp_steps, x[:, i])
        return warped

    @staticmethod
    def mixup(x1, y1, x2, y2, alpha=0.2):
        """MixUp 증강"""
        lam = np.random.beta(alpha, alpha)
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y

    @staticmethod
    def cutmix_temporal(x1, y1, x2, y2, alpha=0.2):
        """Temporal CutMix"""
        lam = np.random.beta(alpha, alpha)
        seq_len = x1.shape[0]
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len + 1)
        x = x1.copy()
        x[cut_start:cut_start+cut_len] = x2[cut_start:cut_start+cut_len]
        y = lam * y1 + (1 - lam) * y2
        return x, y

    @staticmethod
    def joint_noise(x, sigma=0.02):
        """관절 위치에 노이즈 추가"""
        noise = np.random.normal(0, sigma, x.shape)
        return x + noise

    @staticmethod
    def temporal_mask(x, mask_ratio=0.1):
        """시간 구간 마스킹"""
        seq_len = x.shape[0]
        mask_len = int(seq_len * mask_ratio)
        mask_start = np.random.randint(0, seq_len - mask_len + 1)
        x_masked = x.copy()
        x_masked[mask_start:mask_start+mask_len] = 0
        return x_masked


# ============================================================
# Dataset with Advanced Augmentation
# ============================================================
class AdvancedDataset(Dataset):
    def __init__(self, X, y, augment=False, mixup_prob=0.3):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.augment = augment
        self.mixup_prob = mixup_prob
        self.aug = AdvancedAugmentation()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]

        if self.augment:
            # 기본 노이즈
            if np.random.random() < 0.5:
                x = self.aug.joint_noise(x, sigma=0.02)

            # 시간 워핑
            if np.random.random() < 0.3:
                x = self.aug.time_warp(x, sigma=0.1)

            # 시간 마스킹
            if np.random.random() < 0.2:
                x = self.aug.temporal_mask(x, mask_ratio=0.1)

            # MixUp (다른 샘플과)
            if np.random.random() < self.mixup_prob:
                idx2 = np.random.randint(0, len(self.X))
                x, y = self.aug.mixup(x, y, self.X[idx2], self.y[idx2], alpha=0.2)

        return torch.FloatTensor(x), torch.FloatTensor([y]).squeeze()


# ============================================================
# 1. ST-GCN (Spatial-Temporal Graph Convolutional Network)
# ============================================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.register_buffer('adj', adj_matrix)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: (batch, seq, nodes, features)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(self.adj, support)
        return output


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for skeleton data"""
    def __init__(self, num_joints, in_channels=3, hidden_channels=64,
                 num_classes=1, dropout=0.3, adj_matrix=None):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels

        # 기본 인접 행렬 (자기 자신 + 이웃)
        if adj_matrix is None:
            adj_matrix = torch.eye(num_joints) + torch.randn(num_joints, num_joints) * 0.1
            adj_matrix = (adj_matrix + adj_matrix.T) / 2  # 대칭
            adj_matrix = F.softmax(adj_matrix, dim=-1)

        self.register_buffer('adj', adj_matrix)

        # Spatial GCN layers
        self.gc1 = nn.Linear(in_channels, hidden_channels)
        self.gc2 = nn.Linear(hidden_channels, hidden_channels)

        # Temporal Conv layers
        self.tc1 = nn.Conv1d(hidden_channels * num_joints, hidden_channels, 9, padding=4)
        self.tc2 = nn.Conv1d(hidden_channels, hidden_channels, 9, padding=4)

        self.bn1 = nn.BatchNorm1d(hidden_channels * num_joints)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, features = x.shape

        # Reshape to (batch, seq, joints, channels)
        x = x.view(batch_size, seq_len, self.num_joints, self.in_channels)

        # Spatial GCN
        x = F.relu(self.gc1(x))  # (batch, seq, joints, hidden)
        x = torch.matmul(self.adj, x)  # Graph convolution
        x = F.relu(self.gc2(x))
        x = torch.matmul(self.adj, x)

        # Flatten joints and hidden for temporal conv
        x = x.view(batch_size, seq_len, -1)  # (batch, seq, joints*hidden)
        x = x.permute(0, 2, 1)  # (batch, joints*hidden, seq)

        # Temporal Conv
        x = self.bn1(x)
        x = F.relu(self.tc1(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(self.tc2(x))

        # Global pooling
        x = x.mean(dim=-1)  # (batch, hidden)

        return self.fc(x).squeeze(-1)


# ============================================================
# 2. Mamba (State Space Model) - Simplified Version
# ============================================================
class S4Block(nn.Module):
    """Simplified State Space Model block (S4-inspired)"""
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # State space parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq, d_model)
        batch_size, seq_len, _ = x.shape

        # Initialize state
        state = torch.zeros(batch_size, self.d_state, device=x.device)

        outputs = []
        for t in range(seq_len):
            # State update: s_t = A @ s_{t-1} + B @ x_t
            state = torch.tanh(state @ self.A.T + x[:, t] @ self.B.T)
            # Output: y_t = C @ s_t + D * x_t
            out = state @ self.C.T + self.D * x[:, t]
            outputs.append(out)

        output = torch.stack(outputs, dim=1)
        return self.dropout(self.norm(output + x))


class MambaModel(nn.Module):
    """Mamba-inspired State Space Model for time series"""
    def __init__(self, input_size, hidden_size=128, num_layers=4,
                 d_state=64, dropout=0.3, num_classes=1):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.layers = nn.ModuleList([
            S4Block(hidden_size, d_state, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling

        return self.fc(x).squeeze(-1)


# ============================================================
# 3. MS-TCN (Multi-Stage Temporal Convolutional Network)
# ============================================================
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size,
                                       padding=dilation * (kernel_size - 1) // 2,
                                       dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2**i, hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])

    def forward(self, x):
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MSTCN(nn.Module):
    """Multi-Stage TCN for action segmentation/classification"""
    def __init__(self, input_size, hidden_size=128, num_stages=3,
                 num_layers=10, dropout=0.3, num_classes=1):
        super().__init__()

        self.stages = nn.ModuleList()
        for s in range(num_stages):
            in_ch = input_size if s == 0 else hidden_size
            self.stages.append(SingleStageTCN(in_ch, hidden_size, num_layers))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq)

        for stage in self.stages:
            x = stage(x)

        x = x.mean(dim=-1)  # Global pooling
        x = self.dropout(x)

        return self.fc(x).squeeze(-1)


# ============================================================
# 4. PatchTST (Patch Time Series Transformer)
# ============================================================
class PatchEmbedding(nn.Module):
    def __init__(self, input_size, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * input_size, d_model)

    def forward(self, x):
        # x: (batch, seq, features)
        batch_size, seq_len, features = x.shape

        # Pad if needed
        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        # Reshape into patches
        num_patches = (seq_len + pad_len) // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size * features)

        return self.proj(x), num_patches


class PatchTST(nn.Module):
    """Patch Time Series Transformer"""
    def __init__(self, input_size, patch_size=16, d_model=128, nhead=4,
                 num_layers=3, dropout=0.3, num_classes=1):
        super().__init__()

        self.patch_embed = PatchEmbedding(input_size, patch_size, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 500, d_model))  # Max 500 patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        x, num_patches = self.patch_embed(x)

        # Add positional encoding
        x = x + self.pos_embed[:, :num_patches, :]

        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling

        return self.fc(x).squeeze(-1)


# ============================================================
# 5. Ordinal versions for Finger Tapping
# ============================================================
class OrdinalLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        batch_size = targets.size(0)
        binary_targets = torch.zeros(batch_size, self.num_classes - 1, device=targets.device)
        for k in range(self.num_classes - 1):
            binary_targets[:, k] = (targets > k).float()
        return F.binary_cross_entropy_with_logits(logits, binary_targets)


class OrdinalWrapper(nn.Module):
    """Wrapper to convert any regression model to ordinal classification"""
    def __init__(self, base_model, num_classes=5):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Replace final layer
        if hasattr(base_model, 'fc'):
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        return self.base_model(x)

    def predict(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).sum(dim=1).float()
        return preds


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, is_ordinal=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            if is_ordinal:
                pred = model.predict(X)
            else:
                pred = model(X)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    preds_rounded = np.clip(np.round(preds), 0, 4).astype(int)

    mae = mean_absolute_error(labels, preds)
    exact_acc = accuracy_score(labels, preds_rounded)
    within1 = np.mean(np.abs(preds_rounded - labels) <= 1)
    r = pearsonr(preds, labels)[0] if len(np.unique(labels)) > 1 else 0

    return {'mae': mae, 'exact_acc': exact_acc, 'within1_acc': within1, 'pearson_r': r}


def run_groupkfold(X, y, groups, model_fn, device, task_name, is_ordinal=False):
    print(f"\n{'='*60}")
    print(f"GroupKFold CV - {task_name}")
    print(f"Samples: {len(X)}, Subjects: {len(np.unique(groups))}")
    print('='*60)

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_preds, all_labels = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Weighted sampling
        class_counts = np.bincount(y_train.astype(int), minlength=5)
        sample_weights = 1.0 / (class_counts[y_train.astype(int)] + 1)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataset = AdvancedDataset(X_train, y_train, augment=True, mixup_prob=0.2)
        val_dataset = AdvancedDataset(X_val, y_val, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Create model
        model = model_fn().to(device)

        if is_ordinal:
            criterion = OrdinalLoss(num_classes=5)
        else:
            criterion = nn.MSELoss()

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_val_mae = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, device, is_ordinal)
            scheduler.step()

            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

        if best_state:
            model.load_state_dict(best_state)

        val_metrics = evaluate(model, val_loader, device, is_ordinal)
        all_preds.extend([val_metrics['mae']])  # Store fold MAE
        print(f"  Fold {fold+1}: Acc={val_metrics['exact_acc']*100:.1f}%, MAE={val_metrics['mae']:.3f}")

    # Run final evaluation on all folds combined
    # Re-run with predictions stored
    gkf = GroupKFold(n_splits=N_FOLDS)
    all_preds, all_labels = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        class_counts = np.bincount(y_train.astype(int), minlength=5)
        sample_weights = 1.0 / (class_counts[y_train.astype(int)] + 1)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataset = AdvancedDataset(X_train, y_train, augment=True, mixup_prob=0.2)
        val_dataset = AdvancedDataset(X_val, y_val, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = model_fn().to(device)
        criterion = OrdinalLoss(num_classes=5) if is_ordinal else nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_val_mae = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, device, is_ordinal)
            scheduler.step()
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break

        if best_state:
            model.load_state_dict(best_state)

        # Collect predictions
        model.eval()
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                if is_ordinal:
                    pred = model.predict(X_b)
                else:
                    pred = model(X_b)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y_b.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    preds_rounded = np.clip(np.round(all_preds), 0, 4).astype(int)

    mae = mean_absolute_error(all_labels, all_preds)
    acc = accuracy_score(all_labels, preds_rounded)
    within1 = np.mean(np.abs(preds_rounded - all_labels) <= 1)
    r = pearsonr(all_preds, all_labels)[0]

    print(f"\n{task_name} Final Results:")
    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  MAE: {mae:.3f}")
    print(f"  Within-1: {within1*100:.1f}%")
    print(f"  Pearson r: {r:.3f}")

    return {'exact_acc': acc, 'mae': mae, 'within1_acc': within1, 'pearson_r': r}


def evaluate_gait_models(device):
    print("\n" + "=" * 70)
    print("GAIT - Advanced Models with GroupKFold CV")
    print("=" * 70)

    # Load data
    with open('data/gait_train_v2.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/gait_valid_v2.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open('data/gait_test_v2.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
    y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

    train_ids = train_data.get('ids', [f'train_{i}' for i in range(len(train_data['y']))])
    valid_ids = valid_data.get('ids', [f'valid_{i}' for i in range(len(valid_data['y']))])
    test_ids = test_data.get('ids', [f'test_{i}' for i in range(len(test_data['y']))])
    all_ids = list(train_ids) + list(valid_ids) + list(test_ids)
    subjects = np.array([extract_subject_from_id(vid, task='gait') for vid in all_ids])

    print(f"\nData: {X.shape}, Subjects: {len(np.unique(subjects))}")

    input_size = X.shape[2]  # 129
    seq_len = X.shape[1]     # 300

    models = {
        'ST-GCN': lambda: STGCN(num_joints=33, in_channels=input_size//33, hidden_channels=64),
        'Mamba': lambda: MambaModel(input_size, hidden_size=128, num_layers=4),
        'MS-TCN': lambda: MSTCN(input_size, hidden_size=128, num_stages=3, num_layers=8),
        'PatchTST': lambda: PatchTST(input_size, patch_size=15, d_model=128, num_layers=3),
    }

    results = {}
    for name, model_fn in models.items():
        try:
            result = run_groupkfold(X, y, subjects, model_fn, device, f'Gait-{name}', is_ordinal=False)
            results[name] = result
        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = None

    return results


def evaluate_finger_models(device):
    print("\n" + "=" * 70)
    print("FINGER - Advanced Models with GroupKFold CV")
    print("=" * 70)

    # Load data
    with open('data/finger_train_v3.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/finger_valid_v3.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open('data/finger_test_v3.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
    y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

    train_ids = train_data.get('ids', [f'train_{i}' for i in range(len(train_data['y']))])
    valid_ids = valid_data.get('ids', [f'valid_{i}' for i in range(len(valid_data['y']))])
    test_ids = test_data.get('ids', [f'test_{i}' for i in range(len(test_data['y']))])
    all_ids = list(train_ids) + list(valid_ids) + list(test_ids)
    subjects = np.array([extract_subject_from_id(vid, task='finger') for vid in all_ids])

    print(f"\nData: {X.shape}, Subjects: {len(np.unique(subjects))}")

    input_size = X.shape[2]  # 98

    # Ordinal versions
    def make_stgcn():
        model = STGCN(num_joints=21, in_channels=input_size//21, hidden_channels=64)
        return OrdinalWrapper(model, num_classes=5)

    def make_mamba():
        model = MambaModel(input_size, hidden_size=128, num_layers=4)
        return OrdinalWrapper(model, num_classes=5)

    def make_mstcn():
        model = MSTCN(input_size, hidden_size=128, num_stages=3, num_layers=8)
        return OrdinalWrapper(model, num_classes=5)

    def make_patchtst():
        model = PatchTST(input_size, patch_size=10, d_model=128, num_layers=3)
        return OrdinalWrapper(model, num_classes=5)

    models = {
        'ST-GCN': make_stgcn,
        'Mamba': make_mamba,
        'MS-TCN': make_mstcn,
        'PatchTST': make_patchtst,
    }

    results = {}
    for name, model_fn in models.items():
        try:
            result = run_groupkfold(X, y, subjects, model_fn, device, f'Finger-{name}', is_ordinal=True)
            results[name] = result
        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = None

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all', choices=['all', 'gait', 'finger'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    np.random.seed(42)
    torch.manual_seed(42)

    all_results = {}

    if args.task in ['all', 'gait']:
        all_results['gait'] = evaluate_gait_models(device)

    if args.task in ['all', 'finger']:
        all_results['finger'] = evaluate_finger_models(device)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - Advanced Models")
    print("=" * 70)

    for task, results in all_results.items():
        print(f"\n{task.upper()}:")
        for model, metrics in results.items():
            if metrics:
                print(f"  {model}: Acc={metrics['exact_acc']*100:.1f}%, "
                      f"MAE={metrics['mae']:.3f}, r={metrics['pearson_r']:.3f}")
            else:
                print(f"  {model}: FAILED")

    # Save
    with open('results_advanced_models.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: results_advanced_models.pkl")


if __name__ == "__main__":
    main()
