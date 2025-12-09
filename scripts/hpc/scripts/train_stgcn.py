"""
ST-GCN (Spatial-Temporal Graph Convolutional Network) for PD4T
Based on: "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"

Key Innovation:
- Graph convolution captures spatial relations between joints
- Temporal convolution captures motion dynamics
- Expected improvement: +10-15% over baseline ML models
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_dir = Path(__file__).parent.parent / 'data'
results_dir = Path(__file__).parent.parent / 'results'

# ===================== GRAPH DEFINITION =====================

def get_gait_graph():
    """Define skeleton graph for gait analysis (MediaPipe Pose)
    Returns adjacency matrix for 33 landmarks
    """
    # MediaPipe Pose connections (simplified for key body parts)
    # 0: nose, 11-12: shoulders, 23-24: hips, 25-26: knees, 27-28: ankles
    num_nodes = 33

    # Self-connections
    edges = [(i, i) for i in range(num_nodes)]

    # Body connections (MediaPipe Pose topology)
    body_edges = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Torso
        (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        # Right arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        # Left leg
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    ]

    edges.extend(body_edges)
    edges.extend([(j, i) for i, j in body_edges])  # Bidirectional

    # Build adjacency matrix
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1

    # Normalize adjacency matrix
    D = np.sum(A, axis=1)
    D_inv = np.power(D, -0.5)
    D_inv[np.isinf(D_inv)] = 0
    D_inv = np.diag(D_inv)
    A_norm = D_inv @ A @ D_inv

    return torch.FloatTensor(A_norm)

def get_finger_graph():
    """Define skeleton graph for finger tapping (MediaPipe Hands)
    Returns adjacency matrix for 21 landmarks per hand
    """
    num_nodes = 21

    # Self-connections
    edges = [(i, i) for i in range(num_nodes)]

    # Hand connections
    hand_edges = [
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

    edges.extend(hand_edges)
    edges.extend([(j, i) for i, j in hand_edges])

    A = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1

    D = np.sum(A, axis=1)
    D_inv = np.power(D, -0.5)
    D_inv[np.isinf(D_inv)] = 0
    D_inv = np.diag(D_inv)
    A_norm = D_inv @ A @ D_inv

    return torch.FloatTensor(A_norm)

# ===================== ST-GCN MODEL =====================

class GraphConvolution(nn.Module):
    """Graph Convolution Layer"""
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.A = A
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (N, C, T, V) - batch, channels, time, vertices
        N, C, T, V = x.size()

        # Graph convolution: multiply by adjacency matrix
        A = self.A.to(x.device)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        x = x.view(N * T, V, C)
        x = torch.matmul(A, x)  # Graph convolution
        x = x.view(N, T, V, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)

        # 1x1 convolution
        x = self.conv(x)
        x = self.bn(x)

        return x

class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()

        # Spatial graph convolution
        self.gcn = GraphConvolution(in_channels, out_channels, A)

        # Temporal convolution
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1),
                     stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.relu(x + res)
        return x

class STGCN(nn.Module):
    """ST-GCN Model for Skeleton-Based Classification"""
    def __init__(self, in_channels, num_classes, A, num_layers=4, hidden_dim=64):
        super().__init__()

        self.A = A

        # Input batch normalization
        self.bn_in = nn.BatchNorm1d(in_channels * A.shape[0])

        # ST-GCN blocks
        layers = []
        channels = [in_channels] + [hidden_dim] * num_layers

        for i in range(num_layers):
            stride = 2 if i == num_layers - 1 else 1
            layers.append(STGCNBlock(channels[i], channels[i+1], A, stride))

        self.stgcn_layers = nn.ModuleList(layers)

        # Global average pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        # Input normalization
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.bn_in(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        # ST-GCN blocks
        for layer in self.stgcn_layers:
            x = layer(x)

        # Classification
        x = self.pool(x)
        x = x.view(N, -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

# ===================== DATA LOADING =====================

class PD4TDataset(Dataset):
    def __init__(self, X, y, task='gait'):
        self.X = X
        self.y = y
        self.task = task

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, F) - time steps, features
        y = self.y[idx]

        # Reshape for ST-GCN: (C, T, V)
        if self.task == 'gait':
            # Use coordinate features (x, y, z per landmark) + clinical features
            # For simplicity, use clinical features as single-node graph
            # Shape: (F, T, 1) -> treat each feature as a channel
            x = x.T[:, :, np.newaxis]  # (F, T, 1)
        else:
            # Same for finger
            x = x.T[:, :, np.newaxis]

        return torch.FloatTensor(x), torch.LongTensor([y])[0]

def load_data(task='gait'):
    """Load and combine train/valid/test data"""
    with open(data_dir / f'{task}_train_v2.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir / f'{task}_valid_v2.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open(data_dir / f'{task}_test_v2.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
    y = np.hstack([train_data['y'], valid_data['y'], test_data['y']])
    ids = np.hstack([train_data['ids'], valid_data['ids'], test_data['ids']])
    features = train_data['features']

    return X, y, ids, features

def extract_subject_ids(ids):
    """Extract patient IDs from video IDs"""
    return np.array([id.rsplit('_', 1)[-1] for id in ids])

# ===================== TRAINING =====================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            predictions.extend(predicted.cpu().numpy())
            targets.extend(y.cpu().numpy())

    return correct / total, predictions, targets

def run_stgcn_loso(task='gait', epochs=50):
    """Run ST-GCN with LOSO CV"""
    print('=' * 60)
    print(f'{task.upper()} - ST-GCN with LOSO CV')
    print('=' * 60)

    # Load data
    X_raw, y, ids, features = load_data(task)
    subjects = extract_subject_ids(ids)

    unique_subjects = np.unique(subjects)
    print(f'Total samples: {len(y)}, Subjects: {len(unique_subjects)}')
    print(f'Features: {X_raw.shape[1]} channels, {X_raw.shape[2]} time steps')

    # Simple adjacency matrix (fully connected for feature-level graph)
    num_nodes = 1  # Treat as single node with multiple channels
    A = torch.eye(num_nodes)

    num_classes = len(np.unique(y))
    in_channels = X_raw.shape[2]  # Number of features

    logo = LeaveOneGroupOut()
    fold_accs = []
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_raw, y, subjects)):
        # Prepare data
        X_train, y_train = X_raw[train_idx], y[train_idx]
        X_test, y_test = X_raw[test_idx], y[test_idx]

        # Create datasets
        train_dataset = PD4TDataset(X_train, y_train, task)
        test_dataset = PD4TDataset(X_test, y_test, task)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        # Initialize model
        model = STGCN(
            in_channels=in_channels,
            num_classes=num_classes,
            A=A,
            num_layers=3,
            hidden_dim=64
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # Training
        best_acc = 0
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                test_acc, _, _ = evaluate(model, test_loader)
                best_acc = max(best_acc, test_acc)

        # Final evaluation
        test_acc, preds, targets = evaluate(model, test_loader)
        fold_accs.append(test_acc)
        y_true_all.extend(targets)
        y_pred_all.extend(preds)

        if fold % 10 == 0:
            print(f'  Fold {fold+1:2d}/30: Acc={test_acc:.1%}')

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_within1 = np.mean(np.abs(np.array(y_true_all) - np.array(y_pred_all)) <= 1)

    print()
    print(f'  >> Mean: {np.mean(fold_accs):.1%} +/- {np.std(fold_accs):.1%}')
    print(f'  >> Overall: Acc={overall_acc:.1%}, Within-1={overall_within1:.1%}')

    return {
        'mean_acc': np.mean(fold_accs),
        'std_acc': np.std(fold_accs),
        'overall_acc': overall_acc,
        'overall_within1': overall_within1,
    }

# ===================== SIMPLIFIED TCN MODEL =====================

class TemporalConvNet(nn.Module):
    """Simplified TCN for time-series classification"""
    def __init__(self, in_channels, num_classes, hidden_dim=64):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (N, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class PD4TDatasetTCN(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, F)
        y = self.y[idx]
        x = x.T  # (F, T) - channels first
        return torch.FloatTensor(x), torch.LongTensor([y])[0]

def run_tcn_loso(task='gait', epochs=50):
    """Run TCN with LOSO CV"""
    print('=' * 60)
    print(f'{task.upper()} - TCN with LOSO CV')
    print('=' * 60)

    # Load data
    X_raw, y, ids, features = load_data(task)
    subjects = extract_subject_ids(ids)

    unique_subjects = np.unique(subjects)
    print(f'Total samples: {len(y)}, Subjects: {len(unique_subjects)}')

    num_classes = len(np.unique(y))
    in_channels = X_raw.shape[2]

    logo = LeaveOneGroupOut()
    fold_accs = []
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_raw, y, subjects)):
        X_train, y_train = X_raw[train_idx], y[train_idx]
        X_test, y_test = X_raw[test_idx], y[test_idx]

        train_dataset = PD4TDatasetTCN(X_train, y_train)
        test_dataset = PD4TDatasetTCN(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        model = TemporalConvNet(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=64
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()

        test_acc, preds, targets = evaluate(model, test_loader)
        fold_accs.append(test_acc)
        y_true_all.extend(targets)
        y_pred_all.extend(preds)

        if fold % 10 == 0:
            print(f'  Fold {fold+1:2d}/30: Acc={test_acc:.1%}')

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_within1 = np.mean(np.abs(np.array(y_true_all) - np.array(y_pred_all)) <= 1)

    print()
    print(f'  >> Mean: {np.mean(fold_accs):.1%} +/- {np.std(fold_accs):.1%}')
    print(f'  >> Overall: Acc={overall_acc:.1%}, Within-1={overall_within1:.1%}')

    return {
        'mean_acc': np.mean(fold_accs),
        'std_acc': np.std(fold_accs),
        'overall_acc': overall_acc,
        'overall_within1': overall_within1,
    }

def main():
    print('=' * 60)
    print('PD4T Deep Learning Models with LOSO CV')
    print('=' * 60)
    print()

    results = {}

    # TCN (simpler, often works well)
    print('\n--- TCN Model ---\n')
    results['gait_tcn'] = run_tcn_loso('gait', epochs=30)
    results['finger_tcn'] = run_tcn_loso('finger', epochs=30)

    # Summary
    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print()
    print(f'GAIT (Baseline: 67.1%, LOSO ML: 70.4%):')
    print(f'  TCN: {results["gait_tcn"]["mean_acc"]:.1%} +/- {results["gait_tcn"]["std_acc"]:.1%}')
    print()
    print(f'FINGER (Baseline: 59.1%, LOSO ML: 58.3%):')
    print(f'  TCN: {results["finger_tcn"]["mean_acc"]:.1%} +/- {results["finger_tcn"]["std_acc"]:.1%}')

    # Save
    with open(results_dir / 'deep_learning_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()
