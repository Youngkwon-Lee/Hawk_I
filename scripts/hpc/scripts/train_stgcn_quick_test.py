"""
ST-GCN Quick Test - 5-fold CV instead of 30-fold LOSO
For local testing before HPC
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

data_dir = Path(__file__).parent.parent / 'data'

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

# ===================== SIMPLE ST-GCN =====================

class SimpleSTGCN(nn.Module):
    """Simplified ST-GCN for quick testing"""
    def __init__(self, in_channels, num_classes, A, hidden=32):
        super().__init__()
        self.register_buffer('A', A)
        V = A.shape[0]

        # Spatial GCN layers
        self.gcn1 = nn.Conv2d(in_channels, hidden, 1)
        self.gcn2 = nn.Conv2d(hidden, hidden, 1)

        # Temporal conv
        self.tcn1 = nn.Conv2d(hidden, hidden, (9, 1), padding=(4, 0))
        self.tcn2 = nn.Conv2d(hidden, hidden, (9, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(hidden)
        self.bn2 = nn.BatchNorm2d(hidden)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        A = self.A

        # GCN + TCN block 1
        x = torch.einsum('nctv,vw->nctw', x, A)
        x = self.gcn1(x)
        x = self.tcn1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        # GCN + TCN block 2
        x = torch.einsum('nctv,vw->nctw', x, A)
        x = self.gcn2(x)
        x = self.tcn2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        # Classify
        x = self.pool(x).view(N, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ===================== DATASET =====================

class SkeletonDataset(Dataset):
    def __init__(self, X, y, num_joints):
        self.y = y
        self.num_joints = num_joints
        coord_feats = num_joints * 3
        self.X = X[:, :, :coord_feats]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, J*3)
        T = x.shape[0]
        x = x.reshape(T, self.num_joints, 3)  # (T, V, C)
        x = np.transpose(x, (2, 0, 1))  # (C, T, V) where C=3
        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])[0]

# ===================== TRAINING =====================

def load_data(task):
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

def run_quick_cv(task='gait', epochs=30):
    print(f'\n{"="*50}')
    print(f'{task.upper()} - ST-GCN 5-Fold CV (Quick Test)')
    print('='*50)

    X, y, ids = load_data(task)
    subjects = np.array([id.rsplit('_', 1)[-1] for id in ids])

    num_joints = 33 if task == 'gait' else 21
    A = (get_pose_graph() if task == 'gait' else get_hand_graph()).to(device)
    num_classes = len(np.unique(y))

    print(f'Samples: {len(y)}, Classes: {num_classes}')

    gkf = GroupKFold(n_splits=5)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, subjects)):
        train_ds = SkeletonDataset(X[train_idx], y[train_idx], num_joints)
        test_ds = SkeletonDataset(X[test_idx], y[test_idx], num_joints)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=len(test_ds))

        model = SimpleSTGCN(3, num_classes, A, hidden=32).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            for x, yb in train_loader:
                x, yb = x.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for x, yb in test_loader:
                x = x.to(device)
                pred = model(x).argmax(1).cpu().numpy()
                acc = accuracy_score(yb.numpy(), pred)
                fold_accs.append(acc)
                print(f'  Fold {fold+1}: {acc:.1%}')

    print(f'\n  >> Mean: {np.mean(fold_accs):.1%} +/- {np.std(fold_accs):.1%}')
    return np.mean(fold_accs)

if __name__ == '__main__':
    print('ST-GCN Quick Test (5-fold CV)')
    print(f'Device: {device}')

    gait_acc = run_quick_cv('gait', epochs=30)
    finger_acc = run_quick_cv('finger', epochs=30)

    print('\n' + '='*50)
    print('SUMMARY')
    print('='*50)
    print(f'Gait:   {gait_acc:.1%} (baseline: 67.1%, ML: 70.4%)')
    print(f'Finger: {finger_acc:.1%} (baseline: 59.1%, ML: 58.3%)')
