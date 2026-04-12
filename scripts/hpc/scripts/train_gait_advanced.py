"""
Advanced Gait Training with TCN, GNN, and Ensemble Models
Run on HPC with GPU

Current best: SpatialTemporalLSTM 72.2%
Target: 80%+

Models:
1. TCN (Temporal Convolutional Network) - 시계열 특화
2. GCN-LSTM (Graph + LSTM) - 관절 연결 정보 활용
3. Ensemble (Voting/Stacking)

Usage:
    python scripts/train_gait_advanced.py
    python scripts/train_gait_advanced.py --model tcn
    python scripts/train_gait_advanced.py --model gcn
    python scripts/train_gait_advanced.py --model ensemble
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'train_file': 'data/gait_train_v2.pkl',
    'valid_file': 'data/gait_valid_v2.pkl',
    'test_file': 'data/gait_test_v2.pkl',
    'seq_len': 300,
    'input_size': 129,  # 99 raw + 30 clinical
    'raw_size': 99,
    'clinical_size': 30,
    'hidden_size': 128,
    'num_classes': 5,  # UPDRS 0-4
}

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
N_FOLDS = 5

# Pose landmark connections for GCN (MediaPipe Pose)
POSE_EDGES = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32),
    # Face (simplified)
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
]


# ============================================================
# Dataset
# ============================================================
class GaitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# TCN (Temporal Convolutional Network)
# ============================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, input_size, num_channels=[64, 128, 128, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        x = self.network(x)  # (batch, channels, seq)
        x = x.transpose(1, 2)  # (batch, seq, channels)

        # Attention pooling
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = torch.sum(attn_weights * x, dim=1)  # (batch, channels)

        return self.classifier(x).squeeze(-1)


# ============================================================
# Graph Convolutional Network + LSTM
# ============================================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (batch, nodes, features)
        # adj: (nodes, nodes)
        support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
        output = torch.matmul(adj, support)  # (batch, nodes, out_features)
        return output + self.bias


class GCNLSTM(nn.Module):
    """Graph Convolutional Network + LSTM for skeleton data"""
    def __init__(self, num_nodes=33, node_features=3, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_features = node_features

        # Graph convolutions
        self.gc1 = GraphConvolution(node_features, 32)
        self.gc2 = GraphConvolution(32, 64)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            num_nodes * 64, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        # Clinical features encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + 64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Build adjacency matrix
        self.register_buffer('adj', self._build_adjacency())

    def _build_adjacency(self):
        """Build normalized adjacency matrix from pose edges"""
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        for i, j in POSE_EDGES:
            if i < self.num_nodes and j < self.num_nodes:
                adj[i, j] = 1
                adj[j, i] = 1

        # Add self-loops
        adj = adj + torch.eye(self.num_nodes)

        # Normalize
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        return adj_normalized

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Split raw (99 = 33 nodes * 3 coords) and clinical (30)
        raw = x[:, :, :99].reshape(batch_size, seq_len, 33, 3)
        clinical = x[:, :, 99:]

        # Apply GCN per timestep
        gcn_outputs = []
        for t in range(seq_len):
            node_features = raw[:, t]  # (batch, nodes, 3)
            h = F.relu(self.gc1(node_features, self.adj))
            h = F.relu(self.gc2(h, self.adj))  # (batch, nodes, 64)
            h = h.reshape(batch_size, -1)  # (batch, nodes*64)
            gcn_outputs.append(h)

        gcn_out = torch.stack(gcn_outputs, dim=1)  # (batch, seq, nodes*64)

        # LSTM
        lstm_out, _ = self.lstm(gcn_out)  # (batch, seq, hidden*2)

        # Attention
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Clinical features (use mean over time)
        clinical_enc = self.clinical_encoder(clinical.mean(dim=1))  # (batch, 64)

        # Combine and classify
        combined = torch.cat([context, clinical_enc], dim=-1)
        return self.classifier(combined).squeeze(-1)


# ============================================================
# Dilated CNN (FastEval style)
# ============================================================
class DilatedCNN(nn.Module):
    """Dilated CNN inspired by FastEval Parkinsonism (88% accuracy)"""
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, padding=8, dilation=8)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(dropout)

        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))

        # (batch, channels, seq) -> (batch, seq, channels)
        x = x.transpose(1, 2)

        # Attention pooling
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = torch.sum(attn_weights * x, dim=1)

        return self.classifier(x).squeeze(-1)


# ============================================================
# SpatialTemporalLSTM (baseline)
# ============================================================
class SpatialTemporalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3,
                 raw_size=99, clinical_size=30):
        super().__init__()

        self.spatial_encoder = nn.Sequential(
            nn.Linear(raw_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_size, hidden_size // 2),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.raw_size = raw_size
        self.clinical_size = clinical_size

    def forward(self, x):
        raw = x[:, :, :self.raw_size]
        clinical = x[:, :, self.raw_size:self.raw_size + self.clinical_size]

        spatial = self.spatial_encoder(raw)
        clinical_enc = self.clinical_encoder(clinical)
        combined = torch.cat([spatial, clinical_enc], dim=-1)

        lstm_out, _ = self.lstm(combined)

        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.classifier(context).squeeze(-1)


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                pred = model(X)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    preds_rounded = np.clip(np.round(preds), 0, 4)

    mae = mean_absolute_error(labels, preds)
    exact_acc = accuracy_score(labels, preds_rounded)
    within1 = np.mean(np.abs(preds_rounded - labels) <= 1)

    if len(np.unique(labels)) > 1:
        r, _ = pearsonr(preds, labels)
    else:
        r = 0

    return {
        'mae': mae,
        'exact_acc': exact_acc,
        'within1_acc': within1,
        'pearson_r': r,
        'preds': preds,
        'labels': labels
    }


def train_model(model, train_loader, valid_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate(model, valid_loader, device)
        scheduler.step()

        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, "
                  f"Val MAE={val_metrics['mae']:.3f}, "
                  f"Exact={val_metrics['exact_acc']*100:.1f}%")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_mae


def run_cross_validation(X, y, model_class, model_kwargs, device):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_loader = DataLoader(GaitDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(GaitDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model = model_class(**model_kwargs).to(device)
        model, _ = train_model(model, train_loader, val_loader, device)

        val_metrics = evaluate(model, val_loader, device)
        fold_results.append(val_metrics)

        print(f"  Fold {fold+1}: MAE={val_metrics['mae']:.3f}, "
              f"Exact={val_metrics['exact_acc']*100:.1f}%, "
              f"Pearson={val_metrics['pearson_r']:.3f}")

    return {
        'cv_mae': np.mean([r['mae'] for r in fold_results]),
        'cv_exact_acc': np.mean([r['exact_acc'] for r in fold_results]),
        'cv_within1_acc': np.mean([r['within1_acc'] for r in fold_results]),
        'cv_pearson_r': np.mean([r['pearson_r'] for r in fold_results]),
        'fold_results': fold_results
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'tcn', 'gcn', 'dilated', 'baseline'])
    args = parser.parse_args()

    print("=" * 70)
    print("GAIT ADVANCED TRAINING - TCN, GCN, Dilated CNN")
    print("Current best: SpatialTemporalLSTM 72.2%")
    print("Target: 80%+")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading data...")
    with open(CONFIG['train_file'], 'rb') as f:
        train_data = pickle.load(f)
    with open(CONFIG['valid_file'], 'rb') as f:
        valid_data = pickle.load(f)

    X = np.vstack([train_data['X'], valid_data['X']])
    y = np.concatenate([train_data['y'], valid_data['y']])

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.bincount(y.astype(int))}")

    # Models to train
    models_config = {
        'TCN': (TCN, {'input_size': 129, 'num_channels': [64, 128, 128, 64], 'dropout': 0.3}),
        'GCNLSTM': (GCNLSTM, {'num_nodes': 33, 'node_features': 3, 'hidden_size': 128, 'dropout': 0.3}),
        'DilatedCNN': (DilatedCNN, {'input_size': 129, 'hidden_size': 128, 'dropout': 0.3}),
        'SpatialTemporalLSTM': (SpatialTemporalLSTM, {
            'input_size': 129, 'hidden_size': 128, 'raw_size': 99, 'clinical_size': 30
        }),
    }

    if args.model != 'all':
        model_map = {'tcn': 'TCN', 'gcn': 'GCNLSTM', 'dilated': 'DilatedCNN', 'baseline': 'SpatialTemporalLSTM'}
        models_config = {model_map[args.model]: models_config[model_map[args.model]]}

    results = {}

    for name, (model_class, kwargs) in models_config.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print('='*70)

        try:
            cv_results = run_cross_validation(X, y, model_class, kwargs, device)
            results[name] = cv_results

            print(f"\n{name} CV Results:")
            print(f"  MAE: {cv_results['cv_mae']:.3f}")
            print(f"  Exact Accuracy: {cv_results['cv_exact_acc']*100:.1f}%")
            print(f"  Within-1: {cv_results['cv_within1_acc']*100:.1f}%")
            print(f"  Pearson r: {cv_results['cv_pearson_r']:.3f}")

        except Exception as e:
            print(f"Error training {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print('='*70)
    print(f"{'Model':<25} {'MAE':>8} {'Exact':>10} {'Within-1':>10} {'Pearson':>10}")
    print("-" * 70)

    for name, res in sorted(results.items(), key=lambda x: -x[1]['cv_exact_acc']):
        print(f"{name:<25} {res['cv_mae']:>8.3f} "
              f"{res['cv_exact_acc']*100:>9.1f}% "
              f"{res['cv_within1_acc']*100:>9.1f}% "
              f"{res['cv_pearson_r']:>10.3f}")

    # Save results
    output_file = 'results_gait_advanced.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
