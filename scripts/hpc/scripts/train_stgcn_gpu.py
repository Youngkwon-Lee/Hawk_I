#!/usr/bin/env python3
"""
ST-GCN for PD4T - GPU Optimized Version
Run on HPC with SLURM

Usage: python train_stgcn_gpu.py [--task gait|finger|both] [--epochs 100]
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import time
import warnings
warnings.filterwarnings('ignore')

# ===================== CONFIG =====================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='both', choices=['gait', 'finger', 'both'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cv', type=str, default='loso', choices=['loso', '5fold'])
    return parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== SKELETON GRAPHS =====================

def get_pose_graph():
    """MediaPipe Pose: 33 landmarks"""
    num_nodes = 33
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    ]
    A = np.eye(num_nodes)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    D = np.sum(A, axis=1)
    D_inv = np.diag(np.power(D, -0.5))
    return torch.FloatTensor(D_inv @ A @ D_inv)

def get_hand_graph():
    """MediaPipe Hand: 21 landmarks"""
    num_nodes = 21
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    A = np.eye(num_nodes)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    D = np.sum(A, axis=1)
    D_inv = np.diag(np.power(D, -0.5))
    return torch.FloatTensor(D_inv @ A @ D_inv)

# ===================== ST-GCN MODEL =====================

class SpatialGraphConv(nn.Module):
    """Graph Convolution with learnable edge importance"""
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer('A', A)
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # x: (N, C, T, V)
        A = self.A * self.edge_importance
        x = torch.einsum('nctv,vw->nctw', x, A)
        x = self.conv(x)
        x = self.bn(x)
        return x

class STGCNBlock(nn.Module):
    """ST-GCN Block: Spatial GCN + Temporal Conv + Residual"""
    def __init__(self, in_ch, out_ch, A, stride=1, dropout=0.0):
        super().__init__()
        self.gcn = SpatialGraphConv(in_ch, out_ch, A)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.dropout(x)
        return self.relu(x + res)

class STGCN(nn.Module):
    """ST-GCN Model"""
    def __init__(self, in_channels, num_classes, A, hidden=64, num_layers=6, dropout=0.3):
        super().__init__()
        self.register_buffer('A', A)

        # Input BN
        self.bn_in = nn.BatchNorm1d(in_channels * A.shape[0])

        # ST-GCN layers
        channels = [in_channels] + [hidden] * (num_layers - 2) + [hidden * 2] * 2
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            stride = 2 if i == num_layers - 2 else 1
            self.layers.append(STGCNBlock(channels[i], channels[i+1], A, stride, dropout))

        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        N, C, T, V = x.size()

        # Input BN
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.bn_in(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        # ST-GCN
        for layer in self.layers:
            x = layer(x)

        # Classify
        x = self.pool(x).view(N, -1)
        return self.fc(x)

# ===================== DATASET =====================

class SkeletonDataset(Dataset):
    def __init__(self, X, y, num_joints):
        self.y = y
        self.num_joints = num_joints
        self.X = X[:, :, :num_joints * 3]  # Only coordinate features

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, J*3)
        T = x.shape[0]
        x = x.reshape(T, self.num_joints, 3)  # (T, V, C)
        x = np.transpose(x, (2, 0, 1))  # (C, T, V)
        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])[0]

# ===================== DATA LOADING =====================

def load_data(task, data_dir):
    """Load and combine train/valid/test data"""
    X_all, y_all, ids_all = [], [], []
    for split in ['train', 'valid', 'test']:
        path = data_dir / f'{task}_{split}_v2.pkl'
        with open(path, 'rb') as f:
            d = pickle.load(f)
        X_all.append(d['X'])
        y_all.append(d['y'])
        ids_all.append(d['ids'])

    X = np.vstack(X_all)
    y = np.hstack(y_all)
    ids = np.hstack(ids_all)
    subjects = np.array([id.rsplit('_', 1)[-1] for id in ids])

    return X, y, subjects

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            _, pred = out.max(1)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.numpy())

    return np.array(preds), np.array(targets)

def run_cv(task, args, data_dir, results_dir):
    """Run cross-validation"""
    print('=' * 60)
    print(f'{task.upper()} - ST-GCN {args.cv.upper()} CV')
    print('=' * 60)

    # Load data
    X, y, subjects = load_data(task, data_dir)

    # Task settings
    num_joints = 33 if task == 'gait' else 21
    A = (get_pose_graph() if task == 'gait' else get_hand_graph()).to(device)
    num_classes = len(np.unique(y))

    print(f'Samples: {len(y)}, Subjects: {len(np.unique(subjects))}, Classes: {num_classes}')
    print(f'Input: (3, {X.shape[1]}, {num_joints})')
    print(f'Device: {device}')
    print()

    # CV setup
    if args.cv == 'loso':
        cv = LeaveOneGroupOut()
        n_folds = len(np.unique(subjects))
    else:
        cv = GroupKFold(n_splits=5)
        n_folds = 5

    fold_accs = []
    all_preds = []
    all_targets = []

    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, subjects)):
        if len(test_idx) == 0:
            continue

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Datasets
        train_ds = SkeletonDataset(X_train, y_train, num_joints)
        test_ds = SkeletonDataset(X_test, y_test, num_joints)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), num_workers=2, pin_memory=True)

        # Model
        model = STGCN(
            in_channels=3,
            num_classes=num_classes,
            A=A,
            hidden=args.hidden,
            num_layers=6,
            dropout=0.3
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Train
        best_acc = 0
        patience = 15
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

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

        # Final eval
        preds, targets = evaluate(model, test_loader, device)
        acc = accuracy_score(targets, preds)
        fold_accs.append(acc)
        all_preds.extend(preds)
        all_targets.extend(targets)

        if fold % 5 == 0 or fold == n_folds - 1:
            elapsed = time.time() - start_time
            print(f'  Fold {fold+1:2d}/{n_folds}: Acc={acc:.1%} (elapsed: {elapsed:.0f}s)')

    # Results
    overall_acc = accuracy_score(all_targets, all_preds)
    overall_mae = mean_absolute_error(all_targets, all_preds)
    within1 = np.mean(np.abs(np.array(all_targets) - np.array(all_preds)) <= 1)

    print()
    print(f'  >> Mean: {np.mean(fold_accs):.1%} +/- {np.std(fold_accs):.1%}')
    print(f'  >> Overall: Acc={overall_acc:.1%}, MAE={overall_mae:.3f}, Within-1={within1:.1%}')
    print(f'  >> Time: {time.time() - start_time:.0f}s')

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print(f'\nConfusion Matrix:')
    print(cm)

    results = {
        'task': task,
        'mean_acc': np.mean(fold_accs),
        'std_acc': np.std(fold_accs),
        'overall_acc': overall_acc,
        'overall_mae': overall_mae,
        'within1': within1,
        'fold_accs': fold_accs,
        'confusion_matrix': cm
    }

    # Save
    with open(results_dir / f'stgcn_{task}_{args.cv}_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results

# ===================== MAIN =====================

def main():
    args = parse_args()

    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    results_dir = script_dir.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print('=' * 60)
    print('ST-GCN for PD4T')
    print('=' * 60)
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Settings: epochs={args.epochs}, batch={args.batch_size}, hidden={args.hidden}')
    print(f'CV: {args.cv}')
    print()

    results = {}

    if args.task in ['gait', 'both']:
        results['gait'] = run_cv('gait', args, data_dir, results_dir)

    if args.task in ['finger', 'both']:
        print()
        results['finger'] = run_cv('finger', args, data_dir, results_dir)

    # Summary
    print()
    print('=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)

    if 'gait' in results:
        r = results['gait']
        print(f'\nGAIT (Baseline: 67.1%, ML LOSO: 70.4%):')
        print(f'  ST-GCN: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%}')
        print(f'  Overall: {r["overall_acc"]:.1%}, Within-1: {r["within1"]:.1%}')

    if 'finger' in results:
        r = results['finger']
        print(f'\nFINGER (Baseline: 59.1%, ML LOSO: 58.3%):')
        print(f'  ST-GCN: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%}')
        print(f'  Overall: {r["overall_acc"]:.1%}, Within-1: {r["within1"]:.1%}')

    print('\nDone!')

if __name__ == '__main__':
    main()
