#!/usr/bin/env python3
"""
ST-GCN + Clinical Features for PD4T
Best approach: Skeleton graph features + Clinical features combined

Based on best ML results:
- Gait: 70.4% with 191 features (26 clinical + derived)
- Finger: 58.3% with 70 features (10 clinical)
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
    D = np.diag(np.power(np.sum(A, axis=1), -0.5))
    return torch.FloatTensor(D @ A @ D)

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
    D = np.diag(np.power(np.sum(A, axis=1), -0.5))
    return torch.FloatTensor(D @ A @ D)

# ===================== MODEL =====================

class SpatialGraphConv(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.register_buffer('A', A)
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        A = self.A * self.edge_importance
        x = torch.einsum('nctv,vw->nctw', x, A)
        return self.bn(self.conv(x))

class STGCNBlock(nn.Module):
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
        return self.relu(self.dropout(self.tcn(self.gcn(x))) + self.residual(x))

class ClinicalEncoder(nn.Module):
    """Encode clinical time-series features"""
    def __init__(self, in_features, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden, 7, padding=3)
        self.conv2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (N, T, F) -> (N, F, T)
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (N, hidden)
        return x

class STGCN_Clinical(nn.Module):
    """ST-GCN + Clinical Features Combined"""
    def __init__(self, num_classes, A, num_joints, clinical_features, hidden=64):
        super().__init__()

        # ST-GCN branch for skeleton coordinates
        self.bn_in = nn.BatchNorm1d(3 * num_joints)
        self.stgcn_layers = nn.ModuleList([
            STGCNBlock(3, hidden, A, dropout=0.2),
            STGCNBlock(hidden, hidden, A, dropout=0.2),
            STGCNBlock(hidden, hidden * 2, A, stride=2, dropout=0.2),
        ])
        self.stgcn_pool = nn.AdaptiveAvgPool2d(1)

        # Clinical features branch
        self.clinical_encoder = ClinicalEncoder(clinical_features, hidden)

        # Combined classifier
        combined_dim = hidden * 2 + hidden  # stgcn output + clinical output
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )

        self.num_joints = num_joints
        self.clinical_features = clinical_features

    def forward(self, x_coords, x_clinical):
        # x_coords: (N, 3, T, V)
        # x_clinical: (N, T, F_clinical)

        N, C, T, V = x_coords.size()

        # ST-GCN branch
        x = x_coords.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.bn_in(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        for layer in self.stgcn_layers:
            x = layer(x)

        stgcn_out = self.stgcn_pool(x).view(N, -1)  # (N, hidden*2)

        # Clinical branch
        clinical_out = self.clinical_encoder(x_clinical)  # (N, hidden)

        # Combine and classify
        combined = torch.cat([stgcn_out, clinical_out], dim=1)
        return self.classifier(combined)

# ===================== DATASET =====================

class PD4TDataset(Dataset):
    def __init__(self, X, y, num_joints):
        """
        X: (N, T, F) where F = num_joints*3 + clinical_features
        """
        self.y = y
        self.num_joints = num_joints
        coord_dim = num_joints * 3

        # Split coordinates and clinical features
        self.X_coords = X[:, :, :coord_dim]  # (N, T, J*3)
        self.X_clinical = X[:, :, coord_dim:]  # (N, T, F_clinical)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Coordinates -> (3, T, V)
        coords = self.X_coords[idx]  # (T, J*3)
        T = coords.shape[0]
        coords = coords.reshape(T, self.num_joints, 3)
        coords = np.transpose(coords, (2, 0, 1))  # (3, T, V)

        # Clinical features -> (T, F)
        clinical = self.X_clinical[idx]  # (T, F)

        return (torch.FloatTensor(coords),
                torch.FloatTensor(clinical),
                torch.LongTensor([self.y[idx]])[0])

# ===================== DATA LOADING =====================

def load_data(task, data_dir):
    X_all, y_all, ids_all = [], [], []
    for split in ['train', 'valid', 'test']:
        with open(data_dir / f'{task}_{split}_v2.pkl', 'rb') as f:
            d = pickle.load(f)
        X_all.append(d['X'])
        y_all.append(d['y'])
        ids_all.append(d['ids'])

    X = np.vstack(X_all)
    y = np.hstack(y_all)
    ids = np.hstack(ids_all)
    subjects = np.array([id.rsplit('_', 1)[-1] for id in ids])

    # Get clinical feature count
    num_joints = 33 if task == 'gait' else 21
    clinical_features = X.shape[2] - num_joints * 3

    return X, y, subjects, clinical_features

# ===================== TRAINING =====================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for coords, clinical, y in loader:
        coords, clinical, y = coords.to(device), clinical.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(coords, clinical)
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
        for coords, clinical, y in loader:
            coords, clinical = coords.to(device), clinical.to(device)
            out = model(coords, clinical)
            _, pred = out.max(1)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.numpy())

    return np.array(preds), np.array(targets)

def run_cv(task, args, data_dir, results_dir):
    print('=' * 60)
    print(f'{task.upper()} - ST-GCN + Clinical Features')
    print('=' * 60)

    X, y, subjects, clinical_feat_count = load_data(task, data_dir)

    num_joints = 33 if task == 'gait' else 21
    A = (get_pose_graph() if task == 'gait' else get_hand_graph()).to(device)
    num_classes = len(np.unique(y))

    print(f'Samples: {len(y)}, Subjects: {len(np.unique(subjects))}')
    print(f'Skeleton: {num_joints} joints, Clinical: {clinical_feat_count} features')
    print(f'Device: {device}')
    print()

    cv = LeaveOneGroupOut() if args.cv == 'loso' else GroupKFold(n_splits=5)
    n_folds = len(np.unique(subjects)) if args.cv == 'loso' else 5

    fold_accs = []
    all_preds, all_targets = [], []
    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, subjects)):
        if len(test_idx) == 0:
            continue

        train_ds = PD4TDataset(X[train_idx], y[train_idx], num_joints)
        test_ds = PD4TDataset(X[test_idx], y[test_idx], num_joints)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), num_workers=2, pin_memory=True)

        model = STGCN_Clinical(
            num_classes=num_classes,
            A=A,
            num_joints=num_joints,
            clinical_features=clinical_feat_count,
            hidden=args.hidden
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_acc = 0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

            if (epoch + 1) % 20 == 0:
                preds, targets = evaluate(model, test_loader, device)
                acc = accuracy_score(targets, preds)
                best_acc = max(best_acc, acc)

        preds, targets = evaluate(model, test_loader, device)
        acc = accuracy_score(targets, preds)
        fold_accs.append(acc)
        all_preds.extend(preds)
        all_targets.extend(targets)

        if fold % 5 == 0 or fold == n_folds - 1:
            print(f'  Fold {fold+1:2d}/{n_folds}: Acc={acc:.1%} (elapsed: {time.time()-start_time:.0f}s)')

    overall_acc = accuracy_score(all_targets, all_preds)
    within1 = np.mean(np.abs(np.array(all_targets) - np.array(all_preds)) <= 1)

    print()
    print(f'  >> Mean: {np.mean(fold_accs):.1%} +/- {np.std(fold_accs):.1%}')
    print(f'  >> Overall: Acc={overall_acc:.1%}, Within-1={within1:.1%}')
    print(f'  >> Time: {time.time()-start_time:.0f}s')

    cm = confusion_matrix(all_targets, all_preds)
    print(f'\nConfusion Matrix:\n{cm}')

    results = {
        'task': task,
        'mean_acc': np.mean(fold_accs),
        'std_acc': np.std(fold_accs),
        'overall_acc': overall_acc,
        'within1': within1,
    }

    with open(results_dir / f'stgcn_clinical_{task}_{args.cv}.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results

def main():
    args = parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    results_dir = script_dir.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print('=' * 60)
    print('ST-GCN + Clinical Features for PD4T')
    print('=' * 60)
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print()

    results = {}

    if args.task in ['gait', 'both']:
        results['gait'] = run_cv('gait', args, data_dir, results_dir)

    if args.task in ['finger', 'both']:
        print()
        results['finger'] = run_cv('finger', args, data_dir, results_dir)

    print()
    print('=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)

    if 'gait' in results:
        r = results['gait']
        print(f'\nGAIT (Baseline: 67.1%, Best ML: 70.4%):')
        print(f'  ST-GCN+Clinical: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%}')

    if 'finger' in results:
        r = results['finger']
        print(f'\nFINGER (Baseline: 59.1%, Best ML: 58.3%):')
        print(f'  ST-GCN+Clinical: {r["mean_acc"]:.1%} +/- {r["std_acc"]:.1%}')

if __name__ == '__main__':
    main()
