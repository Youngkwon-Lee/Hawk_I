"""
Final Test Set Evaluation for All Models
- Gait: DilatedCNN, TCN, GCNLSTM, SpatialTemporalLSTM
- Finger: AttentionCNNBiLSTM, FeatureAwareModel, AdvancedFeaturesOnly

Train on full train+valid, evaluate on held-out test set.

Usage:
    python scripts/evaluate_final.py --task all
    python scripts/evaluate_final.py --task gait
    python scripts/evaluate_final.py --task finger
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

# ============================================================
# Dataset
# ============================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# GAIT MODELS
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
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                       dilation=dilation_size, padding=(kernel_size-1)*dilation_size,
                                       dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.network(x)
        y = y.mean(dim=2)
        return self.fc(y).squeeze(-1)


class DilatedCNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, 3, padding=8, dilation=8)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.mean(dim=2)
        return self.fc(x).squeeze(-1)


class GCNLSTM(nn.Module):
    def __init__(self, num_nodes=33, node_features=3, hidden_size=128, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features

        self.node_embed = nn.Linear(node_features, hidden_size // 2)
        self.gcn1 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.gcn2 = nn.Linear(hidden_size // 2, hidden_size // 2)

        self.lstm = nn.LSTM(num_nodes * hidden_size // 2, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, features = x.shape
        raw_features = features - (features - self.num_nodes * self.node_features)

        x_pose = x[:, :, :self.num_nodes * self.node_features]
        x_pose = x_pose.view(batch_size, seq_len, self.num_nodes, self.node_features)

        x_nodes = F.relu(self.node_embed(x_pose))
        x_nodes = F.relu(self.gcn1(x_nodes))
        x_nodes = self.dropout(x_nodes)
        x_nodes = F.relu(self.gcn2(x_nodes))

        x_flat = x_nodes.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x_flat)
        out = lstm_out[:, -1, :]
        return self.fc(out).squeeze(-1)


class SpatialTemporalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, raw_size, clinical_size, dropout=0.3):
        super().__init__()
        self.raw_size = raw_size
        self.clinical_size = clinical_size

        self.spatial_conv = nn.Sequential(
            nn.Conv1d(raw_size, hidden_size, 3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(hidden_size + clinical_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        raw = x[:, :, :self.raw_size].permute(0, 2, 1)
        clinical = x[:, :, self.raw_size:]

        spatial = self.spatial_conv(raw).permute(0, 2, 1)
        combined = torch.cat([spatial, clinical], dim=2)

        lstm_out, _ = self.lstm(combined)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.classifier(context).squeeze(-1)


# ============================================================
# FINGER MODELS
# ============================================================
class AttentionCNNBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.classifier(context).squeeze(-1)


class FeatureAwareModel(nn.Module):
    def __init__(self, raw_size=63, basic_size=10, advanced_size=25, hidden_size=128, dropout=0.3):
        super().__init__()
        self.raw_size = raw_size
        self.basic_size = basic_size
        self.advanced_size = advanced_size

        self.raw_encoder = nn.Sequential(
            nn.Conv1d(raw_size, hidden_size // 2, 3, padding=1),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU()
        )

        self.basic_encoder = nn.Sequential(
            nn.Conv1d(basic_size, hidden_size // 4, 3, padding=1),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU()
        )

        self.advanced_encoder = nn.Sequential(
            nn.Conv1d(advanced_size, hidden_size // 4, 3, padding=1),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        raw = x[:, :, :self.raw_size].permute(0, 2, 1)
        basic = x[:, :, self.raw_size:self.raw_size + self.basic_size].permute(0, 2, 1)
        advanced = x[:, :, self.raw_size + self.basic_size:].permute(0, 2, 1)

        raw_feat = self.raw_encoder(raw).permute(0, 2, 1)
        basic_feat = self.basic_encoder(basic).permute(0, 2, 1)
        advanced_feat = self.advanced_encoder(advanced).permute(0, 2, 1)

        combined = torch.cat([raw_feat, basic_feat, advanced_feat], dim=2)
        lstm_out, _ = self.lstm(combined)

        return self.classifier(lstm_out[:, -1, :]).squeeze(-1)


class AdvancedFeaturesOnly(nn.Module):
    def __init__(self, input_size=25, hidden_size=64, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # Use only advanced features (last 25)
        x = x[:, :, -25:]
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :]).squeeze(-1)


# ============================================================
# Training and Evaluation
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
    preds_rounded = np.clip(np.round(preds), 0, 4).astype(int)

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
        'preds_rounded': preds_rounded,
        'labels': labels
    }


def train_and_evaluate(model, train_loader, valid_loader, test_loader, device, model_name):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    print(f"\n  Training {model_name}...")

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

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, Val MAE={val_metrics['mae']:.3f}, "
                  f"Exact={val_metrics['exact_acc']*100:.1f}%")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)

    print(f"\n  {model_name} Test Results:")
    print(f"    MAE: {test_metrics['mae']:.3f}")
    print(f"    Exact Accuracy: {test_metrics['exact_acc']*100:.1f}%")
    print(f"    Within-1: {test_metrics['within1_acc']*100:.1f}%")
    print(f"    Pearson r: {test_metrics['pearson_r']:.3f}")

    # Confusion matrix
    print(f"\n    Confusion Matrix:")
    cm = confusion_matrix(test_metrics['labels'], test_metrics['preds_rounded'],
                         labels=list(range(5)))
    print(f"    {'':>10} |   0    1    2    3    4")
    print(f"    {'-'*45}")
    for i in range(5):
        row = cm[i]
        if row.sum() > 0:
            print(f"    {i:>10} | {row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>4} {row[4]:>4}")

    return test_metrics, model


def evaluate_gait(device):
    print("\n" + "=" * 70)
    print("GAIT - FINAL TEST SET EVALUATION")
    print("=" * 70)

    # Load data
    with open('data/gait_train_v2.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/gait_valid_v2.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open('data/gait_test_v2.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X_train = np.vstack([train_data['X'], valid_data['X']])
    y_train = np.concatenate([train_data['y'], valid_data['y']])
    X_test, y_test = test_data['X'], test_data['y']

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Split train into train/valid (90/10)
    n_valid = int(len(X_train) * 0.1)
    indices = np.random.permutation(len(X_train))
    train_idx, valid_idx = indices[n_valid:], indices[:n_valid]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[valid_idx], y_train[valid_idx]

    train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # Models
    models = {
        'DilatedCNN': DilatedCNN(input_size=129, hidden_size=128, dropout=0.3),
        'TCN': TCN(input_size=129, num_channels=[64, 128, 128, 64], dropout=0.3),
        'GCNLSTM': GCNLSTM(num_nodes=33, node_features=3, hidden_size=128, dropout=0.3),
        'SpatialTemporalLSTM': SpatialTemporalLSTM(input_size=129, hidden_size=128,
                                                    raw_size=99, clinical_size=30),
    }

    results = {}
    for name, model in models.items():
        model = model.to(device)
        test_metrics, trained_model = train_and_evaluate(
            model, train_loader, valid_loader, test_loader, device, name
        )
        results[name] = test_metrics

        # Save best model
        torch.save(trained_model.state_dict(), f'models/gait_{name.lower()}_final.pth')

    return results


def evaluate_finger(device):
    print("\n" + "=" * 70)
    print("FINGER - FINAL TEST SET EVALUATION")
    print("=" * 70)

    # Load data
    with open('data/finger_train_v3.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/finger_valid_v3.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open('data/finger_test_v3.pkl', 'rb') as f:
        test_data = pickle.load(f)

    X_train = np.vstack([train_data['X'], valid_data['X']])
    y_train = np.concatenate([train_data['y'], valid_data['y']])
    X_test, y_test = test_data['X'], test_data['y']

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Split train into train/valid (90/10)
    n_valid = int(len(X_train) * 0.1)
    indices = np.random.permutation(len(X_train))
    train_idx, valid_idx = indices[n_valid:], indices[:n_valid]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[valid_idx], y_train[valid_idx]

    # Weighted sampling for class imbalance
    class_counts = np.bincount(y_tr.astype(int), minlength=5)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = class_weights[y_tr.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=BATCH_SIZE, sampler=sampler)
    valid_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # Models
    models = {
        'AttentionCNNBiLSTM': AttentionCNNBiLSTM(input_size=98, hidden_size=128, dropout=0.3),
        'FeatureAwareModel': FeatureAwareModel(raw_size=63, basic_size=10, advanced_size=25,
                                               hidden_size=128, dropout=0.3),
        'AdvancedFeaturesOnly': AdvancedFeaturesOnly(input_size=25, hidden_size=64, dropout=0.3),
    }

    results = {}
    for name, model in models.items():
        model = model.to(device)
        test_metrics, trained_model = train_and_evaluate(
            model, train_loader, valid_loader, test_loader, device, name
        )
        results[name] = test_metrics

        # Save best model
        torch.save(trained_model.state_dict(), f'models/finger_{name.lower()}_final.pth')

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all', choices=['all', 'gait', 'finger'])
    args = parser.parse_args()

    print("=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("Train on full train+valid, evaluate on held-out test set")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)

    all_results = {}

    if args.task in ['all', 'gait']:
        gait_results = evaluate_gait(device)
        all_results['gait'] = gait_results

    if args.task in ['all', 'finger']:
        finger_results = evaluate_finger(device)
        all_results['finger'] = finger_results

    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - TEST SET RESULTS")
    print("=" * 70)

    for task, results in all_results.items():
        print(f"\n{task.upper()}:")
        print(f"{'Model':<25} {'MAE':>8} {'Exact':>10} {'Within-1':>10} {'Pearson':>10}")
        print("-" * 70)

        for name, res in sorted(results.items(), key=lambda x: -x[1]['exact_acc']):
            print(f"{name:<25} {res['mae']:>8.3f} "
                  f"{res['exact_acc']*100:>9.1f}% "
                  f"{res['within1_acc']*100:>9.1f}% "
                  f"{res['pearson_r']:>10.3f}")

    # Save all results
    with open('results_final_test.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: results_final_test.pkl")


if __name__ == "__main__":
    main()
