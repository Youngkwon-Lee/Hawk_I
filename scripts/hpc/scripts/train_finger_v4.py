"""
Finger Tapping v4 - Enhanced Class Imbalance Handling
- Focal Loss for hard examples
- Stronger class weighting
- Label smoothing
- Mixup augmentation
- Ordinal regression

Problem: Models predicting all Score 1 (majority class)
Solution: Multiple strategies to force learning minority classes

Usage:
    python scripts/train_finger_v4.py
    python scripts/train_finger_v4.py --loss focal
    python scripts/train_finger_v4.py --loss ordinal
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
    'basic_size': 10,
    'advanced_size': 25,
}

BATCH_SIZE = 32
NUM_EPOCHS = 150
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 25
N_FOLDS = 5


# ============================================================
# Loss Functions
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        # For regression, convert to classification-like focal loss
        ce_loss = F.mse_loss(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            # Apply class-specific weights
            target_classes = targets.long().clamp(0, len(self.alpha) - 1)
            alpha_t = self.alpha[target_classes]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class OrdinalLoss(nn.Module):
    """Ordinal Regression Loss - predicts cumulative probabilities"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # logits: (batch, num_classes-1) cumulative probabilities
        # targets: (batch,) integer labels 0-4

        # Create binary labels for each threshold
        batch_size = targets.size(0)
        binary_targets = torch.zeros(batch_size, self.num_classes - 1, device=targets.device)

        for k in range(self.num_classes - 1):
            binary_targets[:, k] = (targets > k).float()

        # Binary cross entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(logits, binary_targets)
        return loss


class WeightedMSELoss(nn.Module):
    """MSE with stronger penalty for minority class errors"""
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        mse = (inputs - targets) ** 2
        target_classes = targets.long().clamp(0, len(self.class_weights) - 1)
        weights = self.class_weights[target_classes]
        return (mse * weights).mean()


# ============================================================
# Data Augmentation
# ============================================================
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def time_warp(x, sigma=0.1):
    """Time warping augmentation"""
    batch_size, seq_len, features = x.shape

    # Random time warping
    warp = torch.randn(batch_size, seq_len, 1, device=x.device) * sigma
    time_steps = torch.linspace(0, 1, seq_len, device=x.device).view(1, -1, 1)
    warped_time = time_steps + warp
    warped_time = warped_time.clamp(0, 1)

    # Simple noise addition as approximation
    return x + torch.randn_like(x) * 0.05


# ============================================================
# Dataset with Augmentation
# ============================================================
class AugmentedDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            # Random noise
            if np.random.random() < 0.5:
                x = x + torch.randn_like(x) * 0.02

            # Random scaling
            if np.random.random() < 0.3:
                scale = 0.9 + np.random.random() * 0.2
                x = x * scale

            # Random time shift
            if np.random.random() < 0.3:
                shift = np.random.randint(-5, 6)
                x = torch.roll(x, shift, dims=0)

        return x, y


# ============================================================
# Models
# ============================================================
class ImprovedAttentionModel(nn.Module):
    """Improved model with dropout and better regularization"""
    def __init__(self, input_size=98, hidden_size=128, dropout=0.4, num_classes=5):
        super().__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        # BiLSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=4, dropout=dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # Conv layers
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Self-attention
        lstm_out = lstm_out.permute(1, 0, 2)  # (seq, batch, features)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)  # (batch, seq, features)

        # Global average pooling
        pooled = attn_out.mean(dim=1)

        return self.classifier(pooled).squeeze(-1)


class OrdinalModel(nn.Module):
    """Ordinal regression model"""
    def __init__(self, input_size=98, hidden_size=128, dropout=0.4, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Ordinal output: num_classes-1 thresholds
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes - 1)  # 4 thresholds for 5 classes
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        pooled = lstm_out.mean(dim=1)

        return self.classifier(pooled)  # (batch, num_classes-1)

    def predict(self, x):
        """Convert ordinal logits to class predictions"""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        # Sum cumulative probabilities to get class
        preds = (probs > 0.5).sum(dim=1).float()
        return preds


class EnsembleModel(nn.Module):
    """Ensemble of multiple models"""
    def __init__(self, input_size=98, hidden_size=128, dropout=0.4):
        super().__init__()

        self.model1 = ImprovedAttentionModel(input_size, hidden_size, dropout)
        self.model2 = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, 1)
        )

        self.combine = nn.Linear(2, 1)

    def forward(self, x):
        out1 = self.model1(x).unsqueeze(1)
        out2 = self.model2(x.permute(0, 2, 1))
        combined = torch.cat([out1, out2], dim=1)
        return self.combine(combined).squeeze(-1)


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler=None, use_mixup=False, is_ordinal=False):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if use_mixup and not is_ordinal:
            X, y_a, y_b, lam = mixup_data(X, y, alpha=0.2)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                pred = model(X)
                if use_mixup and not is_ordinal:
                    loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
                else:
                    loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X)
            if use_mixup and not is_ordinal:
                loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
            else:
                loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, is_ordinal=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            if is_ordinal:
                pred = model.predict(X)
            else:
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


def get_class_weights(y, power=2):
    """Calculate stronger class weights"""
    class_counts = np.bincount(y.astype(int), minlength=5)
    # Inverse frequency with power for stronger weighting
    weights = 1.0 / (class_counts + 1) ** power
    # Normalize
    weights = weights / weights.sum() * 5
    return torch.FloatTensor(weights)


def train_model(model, train_loader, valid_loader, criterion, device,
                use_mixup=False, is_ordinal=False):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                  device, scaler, use_mixup, is_ordinal)
        val_metrics = evaluate(model, valid_loader, device, is_ordinal)
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


def run_experiment(X, y, model_class, model_kwargs, criterion_class, criterion_kwargs,
                   device, name, use_mixup=False, is_ordinal=False):
    """Run 5-fold CV experiment"""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print('='*60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Strong class weighting for sampler
        class_counts = np.bincount(y_train.astype(int), minlength=5)
        sample_weights = 1.0 / (class_counts[y_train.astype(int)] + 1)
        sample_weights = sample_weights ** 1.5  # Stronger weighting
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights) * 2)  # Oversample

        train_dataset = AugmentedDataset(X_train, y_train, augment=True)
        val_dataset = AugmentedDataset(X_val, y_val, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Create model and criterion
        model = model_class(**model_kwargs).to(device)

        if 'class_weights' in criterion_kwargs:
            criterion_kwargs['class_weights'] = get_class_weights(y_train).to(device)
        if 'alpha' in criterion_kwargs and criterion_kwargs['alpha'] == 'auto':
            criterion_kwargs['alpha'] = get_class_weights(y_train).to(device)

        criterion = criterion_class(**criterion_kwargs)

        model, _ = train_model(model, train_loader, val_loader, criterion,
                               device, use_mixup, is_ordinal)

        val_metrics = evaluate(model, val_loader, device, is_ordinal)
        fold_results.append(val_metrics)

        # Print confusion matrix for this fold
        cm = confusion_matrix(val_metrics['labels'], val_metrics['preds_rounded'], labels=list(range(5)))
        print(f"  Fold {fold+1}: MAE={val_metrics['mae']:.3f}, "
              f"Exact={val_metrics['exact_acc']*100:.1f}%, "
              f"Pearson={val_metrics['pearson_r']:.3f}")
        print(f"  Predictions per class: {np.bincount(val_metrics['preds_rounded'].astype(int), minlength=5)}")

    cv_results = {
        'cv_mae': np.mean([r['mae'] for r in fold_results]),
        'cv_exact_acc': np.mean([r['exact_acc'] for r in fold_results]),
        'cv_within1_acc': np.mean([r['within1_acc'] for r in fold_results]),
        'cv_pearson_r': np.mean([r['pearson_r'] for r in fold_results]),
        'fold_results': fold_results
    }

    print(f"\n{name} CV Results:")
    print(f"  MAE: {cv_results['cv_mae']:.3f}")
    print(f"  Exact Accuracy: {cv_results['cv_exact_acc']*100:.1f}%")
    print(f"  Within-1: {cv_results['cv_within1_acc']*100:.1f}%")
    print(f"  Pearson r: {cv_results['cv_pearson_r']:.3f}")

    return cv_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='all',
                        choices=['all', 'focal', 'ordinal', 'weighted', 'mse'])
    args = parser.parse_args()

    print("=" * 70)
    print("FINGER TAPPING v4 - ENHANCED CLASS IMBALANCE HANDLING")
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
    print(f"Labels: {np.bincount(y.astype(int), minlength=5)}")

    # Calculate class weights
    class_weights = get_class_weights(y)
    print(f"Class weights: {class_weights.numpy()}")

    # Experiments
    experiments = {
        'FocalLoss+Attention': {
            'model': ImprovedAttentionModel,
            'model_kwargs': {'input_size': 98, 'hidden_size': 128, 'dropout': 0.4},
            'criterion': FocalLoss,
            'criterion_kwargs': {'alpha': 'auto', 'gamma': 2.0},
            'use_mixup': True,
            'is_ordinal': False
        },
        'WeightedMSE+Attention': {
            'model': ImprovedAttentionModel,
            'model_kwargs': {'input_size': 98, 'hidden_size': 128, 'dropout': 0.4},
            'criterion': WeightedMSELoss,
            'criterion_kwargs': {'class_weights': None},  # Will be set per fold
            'use_mixup': True,
            'is_ordinal': False
        },
        'OrdinalRegression': {
            'model': OrdinalModel,
            'model_kwargs': {'input_size': 98, 'hidden_size': 128, 'dropout': 0.4},
            'criterion': OrdinalLoss,
            'criterion_kwargs': {'num_classes': 5},
            'use_mixup': False,
            'is_ordinal': True
        },
        'Ensemble+Focal': {
            'model': EnsembleModel,
            'model_kwargs': {'input_size': 98, 'hidden_size': 128, 'dropout': 0.4},
            'criterion': FocalLoss,
            'criterion_kwargs': {'alpha': 'auto', 'gamma': 2.5},
            'use_mixup': True,
            'is_ordinal': False
        },
    }

    if args.loss != 'all':
        loss_map = {
            'focal': 'FocalLoss+Attention',
            'ordinal': 'OrdinalRegression',
            'weighted': 'WeightedMSE+Attention',
            'mse': None  # Use standard MSE baseline
        }
        if loss_map[args.loss]:
            experiments = {loss_map[args.loss]: experiments[loss_map[args.loss]]}

    results = {}

    for name, config in experiments.items():
        try:
            cv_results = run_experiment(
                X, y,
                config['model'],
                config['model_kwargs'],
                config['criterion'],
                config['criterion_kwargs'],
                device,
                name,
                config['use_mixup'],
                config['is_ordinal']
            )
            results[name] = cv_results
        except Exception as e:
            print(f"Error in {name}: {e}")
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
    output_file = 'results_finger_v4.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
