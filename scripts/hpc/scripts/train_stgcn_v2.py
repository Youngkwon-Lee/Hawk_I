"""
ST-GCN (Spatial-Temporal Graph Convolutional Network) for PD4T
Direct implementation based on AAAI 2018 paper

Data:
- Gait: 33 MediaPipe Pose landmarks (x, y, z) + clinical features
- Finger: 21 MediaPipe Hand landmarks (x, y, z) + clinical features

Expected: +10-15% improvement over ML baseline
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

data_dir = Path(__file__).parent.parent / 'data'
results_dir = Path(__file__).parent.parent / 'results'

# ===================== SKELETON GRAPHS =====================

def get_mediapipe_pose_graph():
    """
    MediaPipe Pose: 33 landmarks
    Returns normalized adjacency matrix
    """
    num_nodes = 33

    # MediaPipe Pose connections
    edges = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        # Right arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        # Left leg
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    ]

    # Build adjacency with self-loops
    A = np.eye(num_nodes)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    # Normalize: D^(-1/2) * A * D^(-1/2)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(np.power(D, -0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.FloatTensor(A_norm)

def get_mediapipe_hand_graph():
    """
    MediaPipe Hand: 21 landmarks
    Returns normalized adjacency matrix
    """
    num_nodes = 21

    # MediaPipe Hand connections
    edges = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17),
    ]

    A = np.eye(num_nodes)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(np.power(D, -0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.FloatTensor(A_norm)

# ===================== ST-GCN MODULES =====================

class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution"""
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer('A', A)
        self.num_nodes = A.shape[0]

        # Learnable edge importance weights
        self.edge_importance = nn.Parameter(torch.ones_like(A))

        # Convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (N, C, T, V) - batch, channels, time, vertices
        N, C, T, V = x.size()

        # Graph conv: aggregate neighbor features
        A = self.A * self.edge_importance

        # Reshape for matmul: (N*C*T, V) @ (V, V) -> (N*C*T, V)
        x = x.permute(0, 2, 1, 3).contiguous()  # (N, T, C, V)
        x = x.view(N * T, C, V)
        x = torch.einsum('ncv,vw->ncw', x, A)  # Graph conv
        x = x.view(N, T, C, V)
        x = x.permute(0, 2, 1, 3).contiguous()  # (N, C, T, V)

        # 1x1 conv + BN
        x = self.conv(x)
        x = self.bn(x)

        return x

class TemporalConv(nn.Module):
    """Temporal Convolution with dilated convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class STGCNBlock(nn.Module):
    """ST-GCN Block: Spatial GCN + Temporal Conv + Residual"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, dropout=0.0):
        super().__init__()

        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.dropout(x)
        return self.relu(x + res)

class STGCN(nn.Module):
    """
    ST-GCN Model
    Input: (N, C, T, V) - batch, channels (xyz=3), time, vertices
    Output: (N, num_classes)
    """
    def __init__(self, in_channels, num_classes, A, hidden_channels=64, num_layers=6, dropout=0.3):
        super().__init__()

        self.register_buffer('A', A)

        # Input normalization
        self.bn_in = nn.BatchNorm1d(in_channels * A.shape[0])

        # ST-GCN layers with increasing channels
        self.layers = nn.ModuleList()
        channels = [in_channels] + [hidden_channels] * (num_layers - 2) + [hidden_channels * 2, hidden_channels * 2]

        for i in range(num_layers):
            stride = 2 if i == num_layers - 2 else 1  # Temporal downsampling
            self.layers.append(
                STGCNBlock(channels[i], channels[i+1], A, stride=stride, dropout=dropout)
            )

        # Global pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        # Input BN
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.bn_in(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        # ST-GCN blocks
        for layer in self.layers:
            x = layer(x)

        # Classification
        x = self.pool(x).view(N, -1)
        x = self.fc(x)

        return x

# ===================== DATASET =====================

class SkeletonDataset(Dataset):
    """Dataset for skeleton-based action recognition"""
    def __init__(self, X, y, num_joints, coords_per_joint=3):
        """
        X: (N, T, F) - raw features including coords + clinical
        y: (N,) - labels
        num_joints: number of skeleton joints
        coords_per_joint: typically 3 (x, y, z)
        """
        self.y = y
        self.num_joints = num_joints
        self.coords_per_joint = coords_per_joint

        # Extract only coordinate features (first num_joints * 3 features)
        coord_features = num_joints * coords_per_joint
        self.X_coords = X[:, :, :coord_features]  # (N, T, J*3)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X_coords[idx]  # (T, J*3)
        y = self.y[idx]

        # Reshape to (C, T, V) where C=3 (xyz), V=num_joints
        T = x.shape[0]
        x = x.reshape(T, self.num_joints, self.coords_per_joint)  # (T, V, C)
        x = x.transpose(0, 2, 1).transpose(0, 1)  # (C, T, V)

        return torch.FloatTensor(x), torch.LongTensor([y])[0]

# ===================== DATA LOADING =====================

def load_all_data(task='gait'):
    """Load and combine train/valid/test data"""
    with open(data_dir / f'{task}_train_v2.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(data_dir / f'{task}_valid_v2.pkl', 'rb') as f:
        valid = pickle.load(f)
    with open(data_dir / f'{task}_test_v2.pkl', 'rb') as f:
        test = pickle.load(f)

    X = np.vstack([train['X'], valid['X'], test['X']])
    y = np.hstack([train['y'], valid['y'], test['y']])
    ids = np.hstack([train['ids'], valid['ids'], test['ids']])

    return X, y, ids

def extract_subject_ids(ids):
    """Extract patient IDs: '15-001760_009' -> '009'"""
    return np.array([id.rsplit('_', 1)[-1] for id in ids])

# ===================== TRAINING =====================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            _, pred = out.max(1)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.numpy())

    return np.array(preds), np.array(targets)

def run_stgcn_loso(task='gait', epochs=100, patience=15):
    """Run ST-GCN with LOSO CV"""
    print('=' * 60)
    print(f'{task.upper()} - ST-GCN LOSO CV')
    print('=' * 60)

    # Load data
    X, y, ids = load_all_data(task)
    subjects = extract_subject_ids(ids)

    # Task-specific settings
    if task == 'gait':
        num_joints = 33
        A = get_mediapipe_pose_graph().to(device)
    else:
        num_joints = 21
        A = get_mediapipe_hand_graph().to(device)

    num_classes = len(np.unique(y))
    print(f'Samples: {len(y)}, Subjects: {len(np.unique(subjects))}, Classes: {num_classes}')
    print(f'Skeleton: {num_joints} joints, Input shape: (3, {X.shape[1]}, {num_joints})')

    # LOSO CV
    logo = LeaveOneGroupOut()
    fold_accs = []
    all_preds = []
    all_targets = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, subjects)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Datasets
        train_ds = SkeletonDataset(X_train, y_train, num_joints)
        test_ds = SkeletonDataset(X_test, y_test, num_joints)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds))

        # Model
        model = STGCN(
            in_channels=3,
            num_classes=num_classes,
            A=A,
            hidden_channels=64,
            num_layers=6,
            dropout=0.3
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training with early stopping
        best_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

            # Evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                preds, targets = evaluate(model, test_loader, device)
                acc = accuracy_score(targets, preds)
                if acc > best_acc:
                    best_acc = acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience // 10:
                    break

        # Final evaluation
        preds, targets = evaluate(model, test_loader, device)
        acc = accuracy_score(targets, preds)
        fold_accs.append(acc)
        all_preds.extend(preds)
        all_targets.extend(targets)

        if fold % 5 == 0:
            print(f'  Fold {fold+1:2d}/30: Acc={acc:.1%} (train_loss={train_loss:.3f})')

    # Results
    overall_acc = accuracy_score(all_targets, all_preds)
    overall_mae = mean_absolute_error(all_targets, all_preds)
    within1 = np.mean(np.abs(np.array(all_targets) - np.array(all_preds)) <= 1)

    print()
    print(f'  >> Mean: {np.mean(fold_accs):.1%} +/- {np.std(fold_accs):.1%}')
    print(f'  >> Overall: Acc={overall_acc:.1%}, MAE={overall_mae:.3f}, Within-1={within1:.1%}')

    return {
        'mean_acc': np.mean(fold_accs),
        'std_acc': np.std(fold_accs),
        'overall_acc': overall_acc,
        'overall_mae': overall_mae,
        'within1': within1
    }

# ===================== MAIN =====================

def main():
    print('=' * 60)
    print('ST-GCN for PD4T - LOSO Cross-Validation')
    print(f'Device: {device}')
    print('=' * 60)
    print()

    results = {}

    # Gait
    print('\n' + '='*60)
    results['gait'] = run_stgcn_loso('gait', epochs=100)

    # Finger
    print('\n' + '='*60)
    results['finger'] = run_stgcn_loso('finger', epochs=100)

    # Summary
    print('\n' + '='*60)
    print('FINAL SUMMARY - ST-GCN LOSO CV')
    print('='*60)
    print()
    print(f'GAIT (Baseline: 67.1%, ML LOSO: 70.4%):')
    print(f'  ST-GCN: {results["gait"]["mean_acc"]:.1%} +/- {results["gait"]["std_acc"]:.1%}')
    print(f'  Overall: {results["gait"]["overall_acc"]:.1%}, Within-1: {results["gait"]["within1"]:.1%}')
    print()
    print(f'FINGER (Baseline: 59.1%, ML LOSO: 58.3%):')
    print(f'  ST-GCN: {results["finger"]["mean_acc"]:.1%} +/- {results["finger"]["std_acc"]:.1%}')
    print(f'  Overall: {results["finger"]["overall_acc"]:.1%}, Within-1: {results["finger"]["within1"]:.1%}')

    # Save results
    with open(results_dir / 'stgcn_loso_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('\nResults saved to results/stgcn_loso_results.pkl')

if __name__ == '__main__':
    main()
