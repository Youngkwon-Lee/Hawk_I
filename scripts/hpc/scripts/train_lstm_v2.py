"""
Train LSTM models on PD4T v2 data (Raw 3D + Clinical features)
Run on HPC with GPU

Features:
- Finger Tapping: 73 per frame (63 raw + 10 clinical)
- Gait: 129 per frame (99 raw + 30 clinical)

Usage:
    python scripts/train_lstm_v2.py --task finger  # Finger Tapping
    python scripts/train_lstm_v2.py --task gait    # Gait
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
TASK_CONFIG = {
    'finger': {
        'train_file': 'data/finger_train_v2.pkl',
        'valid_file': 'data/finger_valid_v2.pkl',
        'test_file': 'data/finger_test_v2.pkl',
        'seq_len': 150,
        'input_size': 73,  # 63 raw + 10 clinical
        'hidden_size': 128,
        'num_layers': 2,
    },
    'gait': {
        'train_file': 'data/gait_train_v2.pkl',
        'valid_file': 'data/gait_valid_v2.pkl',
        'test_file': 'data/gait_test_v2.pkl',
        'seq_len': 300,
        'input_size': 129,  # 99 raw + 30 clinical
        'hidden_size': 128,
        'num_layers': 2,
    }
}

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
N_FOLDS = 5


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
# Models
# ============================================================
class AttentionLSTM(nn.Module):
    """Bidirectional LSTM with Attention"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        return self.classifier(context).squeeze(-1)


class TransformerModel(nn.Module):
    """Transformer Encoder for sequence classification"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, nhead=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)  # (batch, seq, hidden)
        x = self.transformer(x)  # (batch, seq, hidden)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x).squeeze(-1)


class SpatialTemporalLSTM(nn.Module):
    """Process spatial (landmarks) and temporal (sequence) separately"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3,
                 raw_size=63, clinical_size=10):
        super().__init__()

        # Spatial encoder for raw landmarks
        self.spatial_encoder = nn.Sequential(
            nn.Linear(raw_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2)
        )

        # Clinical encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_size, hidden_size // 2),
            nn.ReLU(),
        )

        # Temporal LSTM
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.raw_size = raw_size
        self.clinical_size = clinical_size

    def forward(self, x):
        # Split features
        raw = x[:, :, :self.raw_size]
        clinical = x[:, :, self.raw_size:self.raw_size + self.clinical_size]

        # Encode
        spatial = self.spatial_encoder(raw)
        clinical_enc = self.clinical_encoder(clinical)

        # Combine
        combined = torch.cat([spatial, clinical_enc], dim=-1)

        # Temporal
        lstm_out, _ = self.lstm(combined)

        # Attention
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

        if scaler:  # Mixed precision
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

    # Round predictions for classification metrics
    preds_rounded = np.clip(np.round(preds), 0, 4)

    mae = mean_absolute_error(labels, preds)
    exact_acc = accuracy_score(labels, preds_rounded)
    within1 = np.mean(np.abs(preds_rounded - labels) <= 1)

    # Pearson correlation
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


def train_model(model, train_loader, valid_loader, device, config):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate(model, valid_loader, device)

        scheduler.step(val_metrics['mae'])

        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_state = model.state_dict().copy()
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


def run_cross_validation(X, y, model_class, config, device):
    """Run K-Fold cross validation"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Create model
        if model_class == SpatialTemporalLSTM:
            if config['input_size'] == 73:  # Finger
                model = model_class(
                    config['input_size'], config['hidden_size'],
                    config['num_layers'], raw_size=63, clinical_size=10
                ).to(device)
            else:  # Gait
                model = model_class(
                    config['input_size'], config['hidden_size'],
                    config['num_layers'], raw_size=99, clinical_size=30
                ).to(device)
        else:
            model = model_class(
                config['input_size'], config['hidden_size'],
                config['num_layers']
            ).to(device)

        model, best_mae = train_model(model, train_loader, val_loader, device, config)

        # Final evaluation
        val_metrics = evaluate(model, val_loader, device)
        fold_results.append(val_metrics)

        print(f"  Fold {fold+1} Results: MAE={val_metrics['mae']:.3f}, "
              f"Exact={val_metrics['exact_acc']*100:.1f}%, "
              f"Within-1={val_metrics['within1_acc']*100:.1f}%")

    # Aggregate results
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_exact = np.mean([r['exact_acc'] for r in fold_results])
    avg_within1 = np.mean([r['within1_acc'] for r in fold_results])
    avg_r = np.mean([r['pearson_r'] for r in fold_results])

    return {
        'cv_mae': avg_mae,
        'cv_exact_acc': avg_exact,
        'cv_within1_acc': avg_within1,
        'cv_pearson_r': avg_r,
        'fold_results': fold_results
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='finger', choices=['finger', 'gait'])
    args = parser.parse_args()

    config = TASK_CONFIG[args.task]

    print("=" * 60)
    print(f"PD4T {args.task.upper()} - LSTM Training v2 (Raw 3D + Clinical)")
    print(f"Features: {config['input_size']} per frame")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print(f"\nLoading data from {config['train_file']}...")
    with open(config['train_file'], 'rb') as f:
        train_data = pickle.load(f)
    with open(config['valid_file'], 'rb') as f:
        valid_data = pickle.load(f)

    # Combine train + valid for CV
    X = np.vstack([train_data['X'], valid_data['X']])
    y = np.concatenate([train_data['y'], valid_data['y']])

    print(f"Combined data: {X.shape}")
    print(f"Labels: {np.bincount(y.astype(int))}")

    # Models to evaluate
    models = [
        ('AttentionLSTM', AttentionLSTM),
        ('Transformer', TransformerModel),
        ('SpatialTemporalLSTM', SpatialTemporalLSTM),
    ]

    results = {}

    for name, model_class in models:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print('='*60)

        try:
            cv_results = run_cross_validation(X, y, model_class, config, device)
            results[name] = cv_results

            print(f"\n{name} CV Results:")
            print(f"  MAE: {cv_results['cv_mae']:.3f}")
            print(f"  Exact Accuracy: {cv_results['cv_exact_acc']*100:.1f}%")
            print(f"  Within-1 Accuracy: {cv_results['cv_within1_acc']*100:.1f}%")
            print(f"  Pearson r: {cv_results['cv_pearson_r']:.3f}")

        except Exception as e:
            print(f"Error training {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)
    print(f"{'Model':<25} {'MAE':>8} {'Exact':>8} {'Within-1':>10} {'Pearson':>8}")
    print("-" * 60)

    for name, res in results.items():
        print(f"{name:<25} {res['cv_mae']:>8.3f} "
              f"{res['cv_exact_acc']*100:>7.1f}% "
              f"{res['cv_within1_acc']*100:>9.1f}% "
              f"{res['cv_pearson_r']:>8.3f}")

    # Save results
    output_file = f'results_{args.task}_v2.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
