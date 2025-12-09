"""
Finger Tapping v4 Deep Learning - HPC GPU Version
Clinical kinematic features (35개) + GroupKFold CV

Usage (HPC):
    sbatch jobs/train_finger_v4.sh

Usage (Local):
    python scripts/train_finger_v4_gpu.py --model all
"""
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import pearsonr
import argparse
from datetime import datetime


# ============================================================
# Configuration
# ============================================================
class Config:
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    RESULT_DIR = "./results"

    NUM_FEATURES = 35

    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    PATIENCE = 20

    N_FOLDS = 5

    HIDDEN_SIZE = 128
    DROPOUT = 0.3

    USE_AMP = True


# ============================================================
# Models
# ============================================================
class DeepMLP(nn.Module):
    """Deep MLP for tabular data"""
    def __init__(self, input_size, hidden_size=128, dropout=0.3, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    """MLP with residual connections"""
    def __init__(self, input_size, hidden_size=128, dropout=0.3, num_classes=5):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)

        self.block1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.block1(x)
        x = nn.functional.relu(x)
        x = x + self.block2(x)
        return self.classifier(x)


class AttentionMLP(nn.Module):
    """MLP with self-attention on features"""
    def __init__(self, input_size, hidden_size=128, dropout=0.3, num_classes=5):
        super().__init__()

        self.feature_embed = nn.Linear(1, 16)
        self.attention = nn.MultiheadAttention(16, num_heads=4, dropout=dropout, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(input_size * 16, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        self.input_size = input_size

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)  # (batch, 35, 1)
        x = self.feature_embed(x)  # (batch, 35, 16)
        x, _ = self.attention(x, x, x)  # (batch, 35, 16)
        x = x.reshape(batch_size, -1)  # (batch, 35*16) - fixed
        return self.classifier(x)


class TabTransformer(nn.Module):
    """Transformer for tabular data"""
    def __init__(self, input_size, hidden_size=128, dropout=0.3, num_classes=5, nhead=4, num_layers=2):
        super().__init__()

        self.embed_dim = 32
        self.feature_embed = nn.Linear(1, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, input_size, self.embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=nhead,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, 35, 1)
        x = self.feature_embed(x)  # (batch, 35, 32)
        x = x + self.pos_embed
        x = self.transformer(x)  # (batch, 35, 32)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


# ============================================================
# Data Loading
# ============================================================
def load_v4_data(config):
    """Load v4 clinical kinematic features"""
    print("Loading v4 data (35 clinical features)...")

    with open(os.path.join(config.DATA_DIR, 'finger_train_v4.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(config.DATA_DIR, 'finger_valid_v4.pkl'), 'rb') as f:
        valid = pickle.load(f)
    with open(os.path.join(config.DATA_DIR, 'finger_test_v4.pkl'), 'rb') as f:
        test = pickle.load(f)

    X = np.vstack([train['X'], valid['X'], test['X']])
    y = np.hstack([train['y'], valid['y'], test['y']])

    all_ids = list(train['ids']) + list(valid['ids']) + list(test['ids'])
    subjects = np.array([vid.rsplit('_', 1)[-1] for vid in all_ids])

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Data shape: {X.shape}")
    print(f"Subjects: {len(np.unique(subjects))}")
    print(f"Label distribution: {np.bincount(y.astype(int), minlength=5)}")

    return X, y, subjects


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        outputs = model(X_t)
        preds = outputs.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y, preds)
    mae = mean_absolute_error(y, preds)
    r, _ = pearsonr(y, preds)

    return {'accuracy': acc, 'mae': mae, 'pearson': r, 'preds': preds}


def groupkfold_cv(X, y, subjects, config, device, model_class):
    """GroupKFold Cross-Validation (subject-level split)"""
    print(f"\n{'='*60}")
    print(f"GroupKFold CV (K={config.N_FOLDS}) - {model_class.__name__}")
    print('='*60)

    gkf = GroupKFold(n_splits=config.N_FOLDS)

    fold_results = []
    all_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, subjects)):
        print(f"\n--- Fold {fold+1}/{config.N_FOLDS} ---")
        print(f"Train subjects: {len(np.unique(subjects[train_idx]))}, Val subjects: {len(np.unique(subjects[val_idx]))}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Normalize
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std

        # Model
        model = model_class(
            input_size=X.shape[1],
            hidden_size=config.HIDDEN_SIZE,
            dropout=config.DROPOUT,
            num_classes=5
        ).to(device)

        # DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4 if device.type == 'cuda' else 0,
            pin_memory=True if device.type == 'cuda' else False
        )

        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        # Mixed precision scaler for GPU
        scaler = torch.cuda.amp.GradScaler() if (config.USE_AMP and device.type == 'cuda') else None

        best_val_acc = 0
        best_state = None
        patience_counter = 0

        for epoch in range(config.EPOCHS):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )

            # Validate
            val_results = evaluate(model, X_val_norm, y_val, device)
            scheduler.step(val_results['mae'])

            if val_results['accuracy'] > best_val_acc:
                best_val_acc = val_results['accuracy']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_state)
        model.to(device)

        # Final evaluation
        results = evaluate(model, X_val_norm, y_val, device)
        all_preds[val_idx] = results['preds']
        fold_results.append(results)

        print(f"  Accuracy: {100*results['accuracy']:.1f}%, MAE: {results['mae']:.3f}, r: {results['pearson']:.3f}")

        # Clear GPU
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Summary
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])

    # Overall Pearson
    overall_r, _ = pearsonr(y, all_preds)

    print(f"\n{'='*60}")
    print(f"CV Results - {model_class.__name__}")
    print('='*60)
    print(f"Accuracy: {100*avg_acc:.1f}% (+/- {100*np.std([r['accuracy'] for r in fold_results]):.1f}%)")
    print(f"MAE: {avg_mae:.3f}")
    print(f"Pearson r: {overall_r:.3f}")

    return {
        'model_name': model_class.__name__,
        'accuracy': avg_acc,
        'mae': avg_mae,
        'pearson': overall_r,
        'fold_results': fold_results
    }


def train_final_model(X, y, subjects, config, device, model_class):
    """Train final model on all data and save"""
    print(f"\n{'='*60}")
    print(f"Training Final Model: {model_class.__name__}")
    print('='*60)

    # Normalize
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - mean) / std

    # Model
    model = model_class(
        input_size=X.shape[1],
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT,
        num_classes=5
    ).to(device)

    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_norm),
        torch.LongTensor(y)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    scaler = torch.cuda.amp.GradScaler() if (config.USE_AMP and device.type == 'cuda') else None

    best_loss = float('inf')
    best_state = None

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}")

    model.load_state_dict(best_state)

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_class.__name__.lower()
    save_path = os.path.join(config.MODEL_DIR, f"finger_v4_{model_name}_{timestamp}.pth")

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model_class.__name__,
        'input_size': X.shape[1],
        'hidden_size': config.HIDDEN_SIZE,
        'dropout': config.DROPOUT,
        'norm_mean': mean,
        'norm_std': std,
    }, save_path)

    print(f"Model saved: {save_path}")

    return model, save_path


def main():
    parser = argparse.ArgumentParser(description='Finger Tapping v4 Deep Learning')
    parser.add_argument('--model', type=str, default='all',
                       choices=['mlp', 'residual', 'attention', 'transformer', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--save_best', action='store_true', help='Save best model after CV')
    args = parser.parse_args()

    # Config
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.HIDDEN_SIZE = args.hidden
    config.USE_AMP = not args.no_amp

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("Finger Tapping v4 Deep Learning - GPU Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Hidden Size: {config.HIDDEN_SIZE}")
    print(f"Mixed Precision: {config.USE_AMP}")

    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Load data
    X, y, subjects = load_v4_data(config)

    # Models
    models = {
        'mlp': DeepMLP,
        'residual': ResidualMLP,
        'attention': AttentionMLP,
        'transformer': TabTransformer
    }

    if args.model == 'all':
        model_list = list(models.values())
    else:
        model_list = [models[args.model]]

    # Results
    all_results = []

    for model_class in model_list:
        cv_results = groupkfold_cv(X, y, subjects, config, device, model_class)
        all_results.append(cv_results)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - Finger Tapping v4 Deep Learning")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':>12} {'MAE':>10} {'Pearson r':>10}")
    print("-" * 55)

    best_model = None
    best_acc = 0
    for r in sorted(all_results, key=lambda x: -x['accuracy']):
        print(f"{r['model_name']:<20} {100*r['accuracy']:>11.1f}% {r['mae']:>10.3f} {r['pearson']:>10.3f}")
        if r['accuracy'] > best_acc:
            best_acc = r['accuracy']
            best_model = r['model_name']

    # Compare with RF baseline
    print(f"\n--- Baseline Comparison ---")
    print(f"RandomForest (v4): 57.9%")
    print(f"Best DL ({best_model}): {100*best_acc:.1f}%")
    print(f"Improvement: +{100*(best_acc - 0.579):.1f}%")

    # Save best model if requested
    if args.save_best:
        best_model_class = models[best_model.lower().replace('mlp', 'mlp').replace('deep', '')]
        train_final_model(X, y, subjects, config, device, best_model_class)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.RESULT_DIR, f"finger_v4_dl_results_{timestamp}.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
