"""
Train ALL 4 Tasks with CORAL Ordinal + RAW SKELETON + K-Fold CV

Key differences from single split version:
- 5-Fold Stratified Cross Validation
- Reports average metrics across folds
- Saves best model from best fold

Expected results (from experiments):
- Gait: Pearson 0.807
- Finger: Pearson 0.555
- Hand: Pearson 0.593
- Leg: Pearson 0.307

Usage:
    python train_all_coral_raw_kfold.py --task gait --epochs 200
    python train_all_coral_raw_kfold.py --task all --epochs 200
"""
import os
import math
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import argparse
from datetime import datetime


# ============================================================
# Configuration
# ============================================================
class Config:
    DATA_DIR = "./data"
    MODEL_DIR = "./models/trained"
    RESULT_DIR = "./results"

    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.02
    PATIENCE = 30

    # K-Fold settings
    N_FOLDS = 5

    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    DROPOUT = 0.4
    USE_AMP = True

    NUM_CLASSES = 5  # UPDRS 0-4


# ============================================================
# CORAL Loss for Ordinal Regression
# ============================================================
class CoralLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        levels = torch.zeros(labels.size(0), self.num_classes - 1, device=labels.device)
        for i in range(self.num_classes - 1):
            levels[:, i] = (labels > i).float()
        loss = F.binary_cross_entropy_with_logits(logits, levels)
        return loss


def ordinal_to_label(logits):
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).sum(dim=1)
    return predictions


# ============================================================
# Mamba Block (Stable Version)
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto", d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv-1, groups=self.d_inner)

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_in = x_conv.transpose(1, 2)
        x_in = F.silu(x_in)

        x_proj_out = self.x_proj(x_in)
        dt, B, C = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log)

        y = self.selective_scan(x_in, dt, A, B, C)
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output

    def selective_scan(self, x, dt, A, B, C):
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        outputs = []
        state = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            B_t = B[:, t, :]
            C_t = C[:, t, :]

            dA = torch.exp(torch.clamp(dt_t.unsqueeze(-1) * A.unsqueeze(0), min=-10, max=10))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            state = state * dA + x_t.unsqueeze(-1) * dB
            state = torch.clamp(state, min=-10, max=10)

            y_t = (state * C_t.unsqueeze(1)).sum(dim=-1)
            y_t = y_t + self.D * x_t

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


# ============================================================
# Mamba Model with CORAL
# ============================================================
class MambaCoralModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4,
                 num_classes=5, dropout=0.4):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([
            MambaBlock(hidden_size) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.coral_head = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, x):
        x = self.input_proj(x)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = self.dropout(x)
            x = x + residual

        x = x.mean(dim=1)
        logits = self.coral_head(x)

        return logits


# ============================================================
# Dataset
# ============================================================
class SkeletonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        if use_amp and device.type == 'cuda':
            with autocast():
                logits = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_expected = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            preds = ordinal_to_label(logits)

            probs = torch.sigmoid(logits)
            expected = probs.sum(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_expected.extend(expected.cpu().numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    expected = np.array(all_expected)

    mae = np.mean(np.abs(preds - labels))
    exact = np.mean(preds == labels)
    within_1 = np.mean(np.abs(preds - labels) <= 1)

    if np.std(expected) > 0 and np.std(labels) > 0:
        pearson, _ = stats.pearsonr(expected, labels)
        spearman, _ = stats.spearmanr(expected, labels)
    else:
        pearson, spearman = 0.0, 0.0

    return {
        'mae': mae, 'exact': exact, 'within_1': within_1,
        'pearson': pearson, 'spearman': spearman,
    }


def train_single_fold(X_train, y_train, X_val, y_val, config, device, fold_num):
    """Train a single fold and return results"""
    train_dataset = SkeletonDataset(X_train, y_train)
    val_dataset = SkeletonDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    input_size = X_train.shape[2]
    model = MambaCoralModel(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    ).to(device)

    criterion = CoralLoss(num_classes=config.NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scaler = GradScaler() if config.USE_AMP else None

    best_pearson = -float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                  scaler, device, config.USE_AMP)
        scheduler.step()

        results = evaluate(model, val_loader, device)

        if results['pearson'] > best_pearson:
            best_pearson = results['pearson']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 40 == 0:
            print(f"  Fold {fold_num} Epoch {epoch+1}: Loss={train_loss:.4f}, "
                  f"MAE={results['mae']:.3f}, Exact={results['exact']*100:.1f}%, "
                  f"Pearson={results['pearson']:.3f}")

        if patience_counter >= config.PATIENCE:
            print(f"  Fold {fold_num} Early stopping at epoch {epoch+1}")
            break

    # Load best model and get final results
    model.load_state_dict(best_model_state)
    final_results = evaluate(model, val_loader, device)

    return final_results, best_model_state, input_size


def train_kfold(task, config, device):
    """Train model using K-Fold Cross Validation"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {task.upper()} (CORAL + RAW + {config.N_FOLDS}-Fold CV)")
    print(f"{'='*60}")

    # Task-specific data paths
    task_files = {
        'gait': ('gait_train.pkl', 'gait_test.pkl'),
        'finger': ('finger_tapping_train.pkl', 'finger_tapping_test.pkl'),
        'hand': ('hand_movement_train.pkl', 'hand_movement_test.pkl'),
        'leg': ('leg_agility_train_v2.pkl', 'leg_agility_test_v2.pkl'),
    }

    train_file, test_file = task_files.get(task, (f'{task}_train.pkl', f'{task}_test.pkl'))

    # Try multiple paths
    possible_paths = [
        (os.path.join(config.DATA_DIR, train_file), os.path.join(config.DATA_DIR, test_file)),
        (f"./{train_file}", f"./{test_file}"),
        (train_file, test_file),
    ]

    if task == 'finger':
        possible_paths.extend([
            (os.path.join(config.DATA_DIR, 'finger_train.pkl'), os.path.join(config.DATA_DIR, 'finger_test.pkl')),
            ('./finger_train.pkl', './finger_test.pkl'),
        ])
    elif task == 'hand':
        possible_paths.extend([
            ('./hand_movement_train.pkl', './hand_movement_test.pkl'),
        ])

    train_path, test_path = None, None
    for tp, tep in possible_paths:
        if os.path.exists(tp):
            train_path, test_path = tp, tep
            break

    print(f"Train data: {train_path}")

    if not train_path or not os.path.exists(train_path):
        print(f"ERROR: Data not found for {task}")
        return None

    # Load ALL data (combine train + test for K-Fold)
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    X_train_orig = train_data['X']
    y_train_orig = train_data['y']

    # Also load test data if exists for final evaluation
    X_test_final, y_test_final = None, None
    if test_path and os.path.exists(test_path):
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        X_test_final = test_data['X']
        y_test_final = test_data['y']
        print(f"Test data: {test_path}")
        print(f"Test shape: {X_test_final.shape}")

    print(f"Train shape: {X_train_orig.shape}")
    print(f"Using RAW skeleton (no enhanced features)")

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    best_fold_pearson = -float('inf')
    best_fold_model = None
    best_fold_input_size = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_orig, y_train_orig)):
        print(f"\n--- Fold {fold+1}/{config.N_FOLDS} ---")

        X_train = X_train_orig[train_idx]
        y_train = y_train_orig[train_idx]
        X_val = X_train_orig[val_idx]
        y_val = y_train_orig[val_idx]

        results, model_state, input_size = train_single_fold(
            X_train, y_train, X_val, y_val, config, device, fold+1
        )

        fold_results.append(results)
        print(f"  Fold {fold+1} Results: MAE={results['mae']:.3f}, "
              f"Exact={results['exact']*100:.1f}%, Pearson={results['pearson']:.3f}")

        # Track best fold
        if results['pearson'] > best_fold_pearson:
            best_fold_pearson = results['pearson']
            best_fold_model = model_state
            best_fold_input_size = input_size

    # Average results across folds
    avg_results = {
        'mae': np.mean([r['mae'] for r in fold_results]),
        'exact': np.mean([r['exact'] for r in fold_results]),
        'within_1': np.mean([r['within_1'] for r in fold_results]),
        'pearson': np.mean([r['pearson'] for r in fold_results]),
        'spearman': np.mean([r['spearman'] for r in fold_results]),
        'pearson_std': np.std([r['pearson'] for r in fold_results]),
    }

    print(f"\n{'='*60}")
    print(f"K-FOLD CV RESULTS - {task.upper()} (CORAL + RAW)")
    print(f"{'='*60}")
    print(f"MAE:      {avg_results['mae']:.3f} (+/- {np.std([r['mae'] for r in fold_results]):.3f})")
    print(f"Exact:    {avg_results['exact']*100:.1f}% (+/- {np.std([r['exact'] for r in fold_results])*100:.1f}%)")
    print(f"Within1:  {avg_results['within_1']*100:.1f}%")
    print(f"Pearson:  {avg_results['pearson']:.3f} (+/- {avg_results['pearson_std']:.3f})")
    print(f"Spearman: {avg_results['spearman']:.3f}")

    # Evaluate on held-out test set if available
    if X_test_final is not None:
        print(f"\n--- Final Test Set Evaluation ---")
        # Create model with best fold weights
        model = MambaCoralModel(
            input_size=best_fold_input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        ).to(device)
        model.load_state_dict(best_fold_model)

        test_dataset = SkeletonDataset(X_test_final, y_test_final)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_results = evaluate(model, test_loader, device)

        print(f"Test MAE:      {test_results['mae']:.3f}")
        print(f"Test Exact:    {test_results['exact']*100:.1f}%")
        print(f"Test Pearson:  {test_results['pearson']:.3f}")

    # Save best model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, f"{task}_coral_raw_kfold_best.pth")

    save_dict = {
        'model_state_dict': best_fold_model,
        'config': {
            'input_size': best_fold_input_size,
            'hidden_size': config.HIDDEN_SIZE,
            'num_layers': config.NUM_LAYERS,
            'num_classes': config.NUM_CLASSES,
            'dropout': config.DROPOUT,
        },
        'cv_results': avg_results,
        'fold_results': fold_results,
        'task': task,
        'method': 'CORAL_RAW_KFOLD',
        'n_folds': config.N_FOLDS,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(save_dict, model_path)
    print(f"\n[SAVED] Model saved to: {model_path}")
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[SAVED] File size: {file_size:.2f} MB")

    return avg_results


def main():
    parser = argparse.ArgumentParser(description='Train all tasks with CORAL + RAW + K-Fold CV')
    parser.add_argument('--task', type=str, default='all',
                        choices=['gait', 'finger', 'hand', 'leg', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.N_FOLDS = args.folds

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("TRAIN ALL TASKS - CORAL + RAW SKELETON + K-FOLD CV")
    print(f"{'='*60}")
    print(f"K-Fold: {config.N_FOLDS} folds")
    print(f"Epochs per fold: {config.EPOCHS}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    tasks = ['gait', 'finger', 'hand', 'leg'] if args.task == 'all' else [args.task]

    all_results = {}

    for task in tasks:
        results = train_kfold(task, config, device)
        if results:
            all_results[task] = results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - ALL TASKS (CORAL + RAW + K-FOLD CV)")
    print(f"{'='*60}")
    print(f"{'Task':<10} {'MAE':>8} {'Exact':>10} {'Pearson':>12} {'Expected':>10}")
    print("-" * 55)

    expected = {'gait': 0.807, 'finger': 0.555, 'hand': 0.593, 'leg': 0.307}

    for task, results in all_results.items():
        exp = expected.get(task, 0.0)
        print(f"{task.upper():<10} {results['mae']:>8.3f} {results['exact']*100:>9.1f}% "
              f"{results['pearson']:>8.3f} +/-{results['pearson_std']:.2f}  ({exp:.3f})")

    print(f"\nModels saved to: {config.MODEL_DIR}/")
    for task in all_results.keys():
        print(f"  - {task}_coral_raw_kfold_best.pth")


if __name__ == "__main__":
    main()
