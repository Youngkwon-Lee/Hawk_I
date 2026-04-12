"""
Advanced Models v2 - Fixed and Optimized
ST-GCN shape 수정, Mamba 최적화, 빠른 학습

Usage:
    python scripts/evaluate_advanced_models_v2.py --task gait
    python scripts/evaluate_advanced_models_v2.py --task finger
"""
import os
import pickle
import argparse
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
# Configuration - 빠른 학습을 위해 조정
# ============================================================
BATCH_SIZE = 64  # 증가
NUM_EPOCHS = 60  # 감소
LEARNING_RATE = 0.001  # 증가
EARLY_STOPPING_PATIENCE = 15
N_FOLDS = 5


def extract_subject_from_id(video_id, task='finger'):
    video_id = video_id.replace('.mp4', '')
    return video_id.rsplit('_', 1)[-1]


# ============================================================
# Dataset
# ============================================================
class SimpleDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment and np.random.random() < 0.5:
            x = x + torch.randn_like(x) * 0.02
        return x, y


# ============================================================
# 1. ST-GCN Fixed - 실제 feature 구조에 맞게 수정
# ============================================================
class STGCN_Fixed(nn.Module):
    """ST-GCN adapted for our feature structure (not raw joints)"""
    def __init__(self, input_size, hidden_size=128, dropout=0.3, num_classes=1):
        super().__init__()

        # Spatial processing (treat features as graph nodes)
        self.spatial_conv1 = nn.Conv1d(input_size, hidden_size, 1)
        self.spatial_conv2 = nn.Conv1d(hidden_size, hidden_size, 1)

        # Temporal processing
        self.temporal_conv1 = nn.Conv1d(hidden_size, hidden_size, 9, padding=4)
        self.temporal_conv2 = nn.Conv1d(hidden_size, hidden_size, 9, padding=4)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Attention for temporal aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq)

        # Spatial
        x = F.relu(self.bn1(self.spatial_conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.spatial_conv2(x)))

        # Temporal
        x = F.relu(self.bn3(self.temporal_conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.temporal_conv2(x)))

        x = x.permute(0, 2, 1)  # (batch, seq, hidden)

        # Attention pooling
        attn_weights = F.softmax(self.attention(x), dim=1)
        x = (attn_weights * x).sum(dim=1)

        return self.fc(x).squeeze(-1)


# ============================================================
# 2. Mamba-Lite - 빠른 버전
# ============================================================
class MambaLite(nn.Module):
    """Lightweight Mamba-inspired model - using parallel SSM approximation"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, num_classes=1):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)

        # Use 1D convolutions to approximate SSM (much faster)
        self.ssm_layers = nn.ModuleList()
        for i in range(num_layers):
            self.ssm_layers.append(nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size * 2, 7, padding=3, groups=hidden_size),
                nn.GLU(dim=1),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.LayerNorm([hidden_size]),
            ))

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.input_proj(x)  # (batch, seq, hidden)

        for layer in self.ssm_layers:
            residual = x
            x = x.permute(0, 2, 1)  # (batch, hidden, seq)
            x = layer[0](x)  # Conv + GLU
            x = layer[1](x)
            x = layer[2](x)  # 1x1 conv
            x = x.permute(0, 2, 1)  # (batch, seq, hidden)
            x = layer[3](x)  # LayerNorm
            x = self.dropout(x) + residual

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling

        return self.fc(x).squeeze(-1)


# ============================================================
# 3. MS-TCN Lite - 빠른 버전
# ============================================================
class MSTCNLite(nn.Module):
    """Lightweight Multi-Stage TCN"""
    def __init__(self, input_size, hidden_size=128, num_stages=2, num_layers=6,
                 dropout=0.3, num_classes=1):
        super().__init__()

        self.input_conv = nn.Conv1d(input_size, hidden_size, 1)

        self.stages = nn.ModuleList()
        for s in range(num_stages):
            layers = nn.ModuleList()
            for i in range(num_layers):
                dilation = 2 ** i
                layers.append(nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))
            self.stages.append(layers)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq)
        x = self.input_conv(x)

        for stage in self.stages:
            for layer in stage:
                residual = x
                x = layer(x) + residual

        x = x.mean(dim=-1)  # Global pooling
        return self.fc(x).squeeze(-1)


# ============================================================
# 4. PatchTST Lite - 빠른 버전
# ============================================================
class PatchTSTLite(nn.Module):
    """Lightweight Patch Time Series Transformer"""
    def __init__(self, input_size, patch_size=20, d_model=128, nhead=4,
                 num_layers=2, dropout=0.3, num_classes=1):
        super().__init__()

        self.patch_size = patch_size
        self.patch_proj = nn.Linear(patch_size * input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # Create patches
        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        num_patches = (seq_len + pad_len) // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size * features)
        x = self.patch_proj(x)

        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)

        return self.fc(x).squeeze(-1)


# ============================================================
# 5. Ensemble Model - 여러 모델 결합
# ============================================================
class EnsembleModel(nn.Module):
    """Ensemble of multiple architectures"""
    def __init__(self, input_size, hidden_size=64, dropout=0.3, num_classes=1):
        super().__init__()

        # Simple CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 7, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, 5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # Combine
        self.fc = nn.Linear(hidden_size + hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # CNN path
        cnn_out = self.cnn(x.permute(0, 2, 1))  # (batch, hidden, seq)
        cnn_out = cnn_out.mean(dim=-1)  # (batch, hidden)

        # LSTM path
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        lstm_out = lstm_out.mean(dim=1)  # (batch, hidden*2)

        # Combine
        combined = torch.cat([cnn_out, lstm_out], dim=-1)
        combined = self.dropout(combined)

        return self.fc(combined).squeeze(-1)


# ============================================================
# Ordinal Wrapper
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
    def __init__(self, base_model, num_classes=5):
        super().__init__()
        self.base = base_model
        self.num_classes = num_classes

        # Replace fc layer
        if hasattr(self.base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        return self.base(x)

    def predict(self, x):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1).float()


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
            pred = model.predict(X) if is_ordinal else model(X)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    preds_rounded = np.clip(np.round(preds), 0, 4).astype(int)

    mae = mean_absolute_error(labels, preds)
    acc = accuracy_score(labels, preds_rounded)
    within1 = np.mean(np.abs(preds_rounded - labels) <= 1)
    r = pearsonr(preds, labels)[0] if len(np.unique(labels)) > 1 else 0

    return {'mae': mae, 'exact_acc': acc, 'within1_acc': within1, 'pearson_r': r}


def run_groupkfold(X, y, groups, model_class, model_kwargs, device, task_name, is_ordinal=False):
    print(f"\n{'='*60}")
    print(f"GroupKFold CV - {task_name}")
    print('='*60)

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_preds, all_labels = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        class_counts = np.bincount(y_train.astype(int), minlength=5)
        sample_weights = 1.0 / (class_counts[y_train.astype(int)] + 1)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(SimpleDataset(X_train, y_train, True),
                                  batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(SimpleDataset(X_val, y_val, False),
                                batch_size=BATCH_SIZE)

        model = model_class(**model_kwargs).to(device)
        if is_ordinal:
            model = OrdinalWrapper(model, num_classes=5).to(device)
            criterion = OrdinalLoss(num_classes=5)
        else:
            criterion = nn.MSELoss()

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        best_mae, best_state, patience = float('inf'), None, 0

        for epoch in range(NUM_EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer, device)
            metrics = evaluate(model, val_loader, device, is_ordinal)
            scheduler.step()

            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Collect predictions
        model.eval()
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device)
                pred = model.predict(X_b) if is_ordinal else model(X_b)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y_b.numpy())

        print(f"  Fold {fold+1}: Acc={metrics['exact_acc']*100:.1f}%, MAE={metrics['mae']:.3f}")

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    preds_rounded = np.clip(np.round(preds), 0, 4).astype(int)

    final = {
        'exact_acc': accuracy_score(labels, preds_rounded),
        'mae': mean_absolute_error(labels, preds),
        'within1_acc': np.mean(np.abs(preds_rounded - labels) <= 1),
        'pearson_r': pearsonr(preds, labels)[0]
    }

    print(f"\n{task_name} FINAL: Acc={final['exact_acc']*100:.1f}%, MAE={final['mae']:.3f}, r={final['pearson_r']:.3f}")
    return final


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    all_results = {}

    # GAIT
    if args.task in ['all', 'gait']:
        print("\n" + "=" * 70)
        print("GAIT - Advanced Models v2")
        print("=" * 70)

        with open('data/gait_train_v2.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('data/gait_valid_v2.pkl', 'rb') as f:
            valid_data = pickle.load(f)
        with open('data/gait_test_v2.pkl', 'rb') as f:
            test_data = pickle.load(f)

        X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
        y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

        all_ids = (list(train_data.get('ids', [])) +
                   list(valid_data.get('ids', [])) +
                   list(test_data.get('ids', [])))
        if not all_ids:
            all_ids = [f'sample_{i}' for i in range(len(y))]
        subjects = np.array([extract_subject_from_id(vid) for vid in all_ids])

        print(f"Data: {X.shape}, Subjects: {len(np.unique(subjects))}")
        input_size = X.shape[2]

        models = {
            'ST-GCN': (STGCN_Fixed, {'input_size': input_size, 'hidden_size': 128}),
            'Mamba-Lite': (MambaLite, {'input_size': input_size, 'hidden_size': 128, 'num_layers': 2}),
            'MS-TCN': (MSTCNLite, {'input_size': input_size, 'hidden_size': 128, 'num_stages': 2}),
            'PatchTST': (PatchTSTLite, {'input_size': input_size, 'patch_size': 20, 'd_model': 128}),
            'Ensemble': (EnsembleModel, {'input_size': input_size, 'hidden_size': 64}),
        }

        gait_results = {}
        for name, (model_class, kwargs) in models.items():
            try:
                result = run_groupkfold(X, y, subjects, model_class, kwargs, device, f'Gait-{name}')
                gait_results[name] = result
            except Exception as e:
                print(f"  {name} failed: {e}")
                gait_results[name] = None

        all_results['gait'] = gait_results

    # FINGER
    if args.task in ['all', 'finger']:
        print("\n" + "=" * 70)
        print("FINGER - Advanced Models v2")
        print("=" * 70)

        with open('data/finger_train_v3.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('data/finger_valid_v3.pkl', 'rb') as f:
            valid_data = pickle.load(f)
        with open('data/finger_test_v3.pkl', 'rb') as f:
            test_data = pickle.load(f)

        X = np.vstack([train_data['X'], valid_data['X'], test_data['X']])
        y = np.concatenate([train_data['y'], valid_data['y'], test_data['y']])

        all_ids = (list(train_data.get('ids', [])) +
                   list(valid_data.get('ids', [])) +
                   list(test_data.get('ids', [])))
        if not all_ids:
            all_ids = [f'sample_{i}' for i in range(len(y))]
        subjects = np.array([extract_subject_from_id(vid) for vid in all_ids])

        print(f"Data: {X.shape}, Subjects: {len(np.unique(subjects))}")
        input_size = X.shape[2]

        models = {
            'ST-GCN': (STGCN_Fixed, {'input_size': input_size, 'hidden_size': 128}),
            'Mamba-Lite': (MambaLite, {'input_size': input_size, 'hidden_size': 128, 'num_layers': 2}),
            'MS-TCN': (MSTCNLite, {'input_size': input_size, 'hidden_size': 128, 'num_stages': 2}),
            'PatchTST': (PatchTSTLite, {'input_size': input_size, 'patch_size': 15, 'd_model': 128}),
            'Ensemble': (EnsembleModel, {'input_size': input_size, 'hidden_size': 64}),
        }

        finger_results = {}
        for name, (model_class, kwargs) in models.items():
            try:
                result = run_groupkfold(X, y, subjects, model_class, kwargs, device, f'Finger-{name}', is_ordinal=True)
                finger_results[name] = result
            except Exception as e:
                print(f"  {name} failed: {e}")
                finger_results[name] = None

        all_results['finger'] = finger_results

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - Advanced Models v2")
    print("=" * 70)

    for task, results in all_results.items():
        print(f"\n{task.upper()}:")
        for model, metrics in results.items():
            if metrics:
                print(f"  {model}: Acc={metrics['exact_acc']*100:.1f}%, MAE={metrics['mae']:.3f}, r={metrics['pearson_r']:.3f}")
            else:
                print(f"  {model}: FAILED")

    with open('results_advanced_v2.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved to: results_advanced_v2.pkl")


if __name__ == "__main__":
    main()
