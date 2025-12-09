"""
Train Finger Tapping v3 models with enhanced features
Based on prior research: CNN-BiLSTM + Attention (93% accuracy)

Key improvements:
1. Enhanced features (v3): 98 features (63 raw + 10 basic + 25 advanced)
2. Attention CNN-BiLSTM architecture
3. Class weighting for imbalanced data
4. Ordinal regression option

Usage:
    python scripts/train_finger_v3.py
    python scripts/train_finger_v3.py --loss ordinal
    python scripts/train_finger_v3.py --model attention_cnn_bilstm
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'train_file': 'data/finger_train_v3.pkl',
    'valid_file': 'data/finger_valid_v3.pkl',
    'test_file': 'data/finger_test_v3.pkl',
    'seq_len': 150,
    'input_size': 98,  # 63 raw + 10 basic + 25 advanced
    'raw_size': 63,
    'basic_clinical_size': 10,
    'advanced_size': 25,
    'hidden_size': 128,
    'num_classes': 5,
}

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
N_FOLDS = 5


# ============================================================
# Dataset with Class Weighting
# ============================================================
class FingerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_weighted_sampler(y):
    """Create weighted sampler for imbalanced classes"""
    class_counts = np.bincount(y.astype(int), minlength=5)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = class_weights[y.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y),
        replacement=True
    )
    return sampler


# ============================================================
# Attention CNN-BiLSTM (Based on arXiv:2510.10121)
# ============================================================
class AttentionCNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM with Attention mechanism
    Architecture from 93% accuracy paper
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        # 1D CNN for local feature extraction
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        # BiLSTM for temporal modeling
        self.lstm1 = nn.LSTM(
            128, hidden_size, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Second BiLSTM
        self.lstm2 = nn.LSTM(
            hidden_size * 2, hidden_size, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0
        )

        # CNN features branch
        self.cnn_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + 64, 250),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(250, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # CNN branch
        x_cnn = x.transpose(1, 2)  # (batch, features, seq)
        x_cnn = F.relu(self.bn1(self.conv1(x_cnn)))
        x_cnn = self.pool1(x_cnn)
        x_cnn = F.relu(self.bn2(self.conv2(x_cnn)))

        cnn_features = self.cnn_fc(x_cnn)  # (batch, 64)

        # LSTM branch
        x_lstm = x_cnn.transpose(1, 2)  # (batch, seq/2, 128)
        lstm1_out, _ = self.lstm1(x_lstm)  # (batch, seq/2, hidden*2)

        # Attention
        attn_weights = self.attention(lstm1_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm1_out, dim=1, keepdim=True)

        # Expand context and concatenate
        context_expanded = context.expand(-1, lstm1_out.shape[1], -1)
        lstm2_input = lstm1_out + context_expanded

        lstm2_out, _ = self.lstm2(lstm2_input)

        # Final attention pooling
        attn_weights2 = self.attention(lstm2_out)
        attn_weights2 = torch.softmax(attn_weights2, dim=1)
        lstm_features = torch.sum(attn_weights2 * lstm2_out, dim=1)  # (batch, hidden*2)

        # Concatenate CNN and LSTM features
        combined = torch.cat([lstm_features, cnn_features], dim=-1)

        return self.classifier(combined).squeeze(-1)


# ============================================================
# Feature-Aware Model (Separate encoders for feature groups)
# ============================================================
class FeatureAwareModel(nn.Module):
    """Process different feature groups separately then combine"""
    def __init__(self, raw_size=63, basic_size=10, advanced_size=25, hidden_size=128, dropout=0.3):
        super().__init__()

        # Raw 3D encoder (spatial patterns)
        self.raw_encoder = nn.Sequential(
            nn.Linear(raw_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        # Basic clinical encoder
        self.basic_encoder = nn.Sequential(
            nn.Linear(basic_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Advanced temporal encoder (important features!)
        self.advanced_encoder = nn.Sequential(
            nn.Linear(advanced_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )

        # Combined feature size: 32 + 16 + 32 = 80
        combined_size = 80

        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            combined_size, hidden_size, num_layers=2,
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
        self.basic_size = basic_size
        self.advanced_size = advanced_size

    def forward(self, x):
        # Split features
        raw = x[:, :, :self.raw_size]
        basic = x[:, :, self.raw_size:self.raw_size + self.basic_size]
        advanced = x[:, :, self.raw_size + self.basic_size:]

        # Encode each group
        raw_enc = self.raw_encoder(raw)
        basic_enc = self.basic_encoder(basic)
        advanced_enc = self.advanced_encoder(advanced)

        # Combine
        combined = torch.cat([raw_enc, basic_enc, advanced_enc], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(combined)

        # Attention
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.classifier(context).squeeze(-1)


# ============================================================
# Advanced Features Only Model (for comparison)
# ============================================================
class AdvancedFeaturesOnly(nn.Module):
    """Use only the advanced temporal features (25) for classification"""
    def __init__(self, advanced_size=25, hidden_size=64, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            advanced_size, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.advanced_start = 63 + 10  # raw + basic

    def forward(self, x):
        # Use only advanced features
        advanced = x[:, :, self.advanced_start:]

        lstm_out, _ = self.lstm(advanced)

        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.classifier(context).squeeze(-1)


# ============================================================
# Loss Functions
# ============================================================
class OrdinalLoss(nn.Module):
    """Ordinal regression loss - respects score ordering"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        # Convert to binary cumulative labels
        target_int = target.long()
        cum_labels = torch.zeros(len(target), self.num_classes - 1, device=pred.device)
        for i in range(self.num_classes - 1):
            cum_labels[:, i] = (target_int > i).float()

        # Expand predictions to match
        pred_expanded = pred.unsqueeze(1).expand(-1, self.num_classes - 1)

        # Binary cross entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(pred_expanded, cum_labels)
        return loss


class FocalLoss(nn.Module):
    """Focal loss for class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        focal_weight = (1 + mse) ** self.gamma
        return (self.alpha * focal_weight * mse).mean()


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

    # Per-class accuracy
    cm = confusion_matrix(labels, preds_rounded, labels=range(5))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)

    return {
        'mae': mae,
        'exact_acc': exact_acc,
        'within1_acc': within1,
        'pearson_r': r,
        'per_class_acc': per_class_acc,
        'preds': preds,
        'labels': labels,
        'confusion_matrix': cm
    }


def train_model(model, train_loader, valid_loader, device, use_focal=False):
    if use_focal:
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
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


def run_cross_validation(X, y, model_class, model_kwargs, device, use_weighted_sampling=True, use_focal=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = FingerDataset(X_train, y_train)
        val_dataset = FingerDataset(X_val, y_val)

        if use_weighted_sampling:
            sampler = get_weighted_sampler(y_train)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = model_class(**model_kwargs).to(device)
        model, _ = train_model(model, train_loader, val_loader, device, use_focal)

        val_metrics = evaluate(model, val_loader, device)
        fold_results.append(val_metrics)

        print(f"  Fold {fold+1}: MAE={val_metrics['mae']:.3f}, "
              f"Exact={val_metrics['exact_acc']*100:.1f}%, "
              f"Per-class: {[f'{a*100:.0f}%' for a in val_metrics['per_class_acc']]}")

    return {
        'cv_mae': np.mean([r['mae'] for r in fold_results]),
        'cv_exact_acc': np.mean([r['exact_acc'] for r in fold_results]),
        'cv_within1_acc': np.mean([r['within1_acc'] for r in fold_results]),
        'cv_pearson_r': np.mean([r['pearson_r'] for r in fold_results]),
        'cv_per_class_acc': np.mean([r['per_class_acc'] for r in fold_results], axis=0),
        'fold_results': fold_results
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'attention_cnn', 'feature_aware', 'advanced_only'])
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'focal'])
    parser.add_argument('--no-weighting', action='store_true', help='Disable class weighting')
    args = parser.parse_args()

    print("=" * 70)
    print("FINGER TAPPING V3 - Enhanced Features Training")
    print("Features: 98 (63 raw + 10 basic + 25 advanced temporal)")
    print("Current best: SpatialTemporalLSTM 57.8%")
    print("Target: 80%+ (based on prior research)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading data...")
    try:
        with open(CONFIG['train_file'], 'rb') as f:
            train_data = pickle.load(f)
        with open(CONFIG['valid_file'], 'rb') as f:
            valid_data = pickle.load(f)
    except FileNotFoundError:
        print("ERROR: v3 data files not found!")
        print("Please run: python scripts/prepare_finger_v3.py")
        return

    X = np.vstack([train_data['X'], valid_data['X']])
    y = np.concatenate([train_data['y'], valid_data['y']])

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.bincount(y.astype(int))}")
    print(f"Class imbalance ratio: {np.max(np.bincount(y.astype(int))) / np.min(np.bincount(y.astype(int)[np.bincount(y.astype(int)) > 0]):.1f}x")

    # Models to train
    models_config = {
        'AttentionCNNBiLSTM': (AttentionCNNBiLSTM, {
            'input_size': CONFIG['input_size'], 'hidden_size': 128, 'dropout': 0.3
        }),
        'FeatureAwareModel': (FeatureAwareModel, {
            'raw_size': 63, 'basic_size': 10, 'advanced_size': 25, 'hidden_size': 128
        }),
        'AdvancedFeaturesOnly': (AdvancedFeaturesOnly, {
            'advanced_size': 25, 'hidden_size': 64
        }),
    }

    if args.model != 'all':
        model_map = {
            'attention_cnn': 'AttentionCNNBiLSTM',
            'feature_aware': 'FeatureAwareModel',
            'advanced_only': 'AdvancedFeaturesOnly'
        }
        models_config = {model_map[args.model]: models_config[model_map[args.model]]}

    use_focal = args.loss == 'focal'
    use_weighted = not args.no_weighting

    print(f"\nSettings:")
    print(f"  Loss: {args.loss}")
    print(f"  Weighted sampling: {use_weighted}")

    results = {}

    for name, (model_class, kwargs) in models_config.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print('='*70)

        try:
            cv_results = run_cross_validation(
                X, y, model_class, kwargs, device,
                use_weighted_sampling=use_weighted,
                use_focal=use_focal
            )
            results[name] = cv_results

            print(f"\n{name} CV Results:")
            print(f"  MAE: {cv_results['cv_mae']:.3f}")
            print(f"  Exact Accuracy: {cv_results['cv_exact_acc']*100:.1f}%")
            print(f"  Within-1: {cv_results['cv_within1_acc']*100:.1f}%")
            print(f"  Pearson r: {cv_results['cv_pearson_r']:.3f}")
            print(f"  Per-class accuracy:")
            for i, acc in enumerate(cv_results['cv_per_class_acc']):
                print(f"    Score {i}: {acc*100:.1f}%")

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
    output_file = 'results_finger_v3.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
