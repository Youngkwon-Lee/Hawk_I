"""
Train ALL 4 Tasks with Model Saving (.pth)
- Gait: CORAL Ordinal
- Finger: Mamba + Enhanced
- Hand: CORAL Ordinal
- Leg: Mamba + CORAL

Usage:
    python train_all_tasks_with_save.py --task gait --epochs 200
    python train_all_tasks_with_save.py --task all --epochs 200
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
# Mamba Block
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

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            state = state * dA + x_t.unsqueeze(-1) * dB
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

        # CORAL head: K-1 binary classifiers
        self.coral_head = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, x):
        x = self.input_proj(x)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = self.dropout(x)
            x = x + residual

        # Global average pooling
        x = x.mean(dim=1)

        # CORAL logits
        logits = self.coral_head(x)

        return logits


# ============================================================
# Feature Engineering
# ============================================================
class FeatureEngineer:
    @staticmethod
    def add_velocity(x):
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        return velocity

    @staticmethod
    def add_acceleration(x):
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        return acceleration

    @staticmethod
    def engineer_features(X):
        enhanced = []
        for sample in X:
            velocity = FeatureEngineer.add_velocity(sample)
            acceleration = FeatureEngineer.add_acceleration(sample)
            combined = np.concatenate([sample, velocity, acceleration], axis=-1)
            enhanced.append(combined)
        return np.array(enhanced)


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
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
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

            # Expected value
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

    pearson, _ = stats.pearsonr(expected, labels)
    spearman, _ = stats.spearmanr(expected, labels)

    return {
        'mae': mae, 'exact': exact, 'within_1': within_1,
        'pearson': pearson, 'spearman': spearman,
        'predictions': preds, 'labels': labels, 'expected': expected
    }


def train_and_save_model(task, config, device):
    """Train model for a specific task and save weights"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {task.upper()}")
    print(f"{'='*60}")

    # Load data
    train_path = os.path.join(config.DATA_DIR, f"{task}_train.pkl")
    test_path = os.path.join(config.DATA_DIR, f"{task}_test.pkl")

    # Try alternative paths
    if not os.path.exists(train_path):
        train_path = f"./{task}_train.pkl"
        test_path = f"./{task}_test.pkl"

    if not os.path.exists(train_path):
        # Task-specific naming
        task_names = {
            'gait': 'gait',
            'finger': 'finger_tapping',
            'hand': 'hand_movement',
            'leg': 'leg_agility'
        }
        actual_name = task_names.get(task, task)
        train_path = os.path.join(config.DATA_DIR, f"{actual_name}_train.pkl")
        test_path = os.path.join(config.DATA_DIR, f"{actual_name}_test.pkl")

    if not os.path.exists(train_path):
        # Try v2 files for leg
        if task == 'leg':
            train_path = os.path.join(config.DATA_DIR, "leg_agility_train_v2.pkl")
            test_path = os.path.join(config.DATA_DIR, "leg_agility_test_v2.pkl")

    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")

    if not os.path.exists(train_path):
        print(f"ERROR: Data not found for {task}")
        return None

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Feature engineering
    X_train_enhanced = FeatureEngineer.engineer_features(X_train)
    X_test_enhanced = FeatureEngineer.engineer_features(X_test)

    print(f"Enhanced shape: {X_train_enhanced.shape}")

    # Create datasets
    train_dataset = SkeletonDataset(X_train_enhanced, y_train)
    test_dataset = SkeletonDataset(X_test_enhanced, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Create model
    input_size = X_train_enhanced.shape[2]
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

    # Training
    best_mae = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                  scaler, device, config.USE_AMP)
        scheduler.step()

        # Evaluate
        results = evaluate(model, test_loader, device)

        if results['mae'] < best_mae:
            best_mae = results['mae']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, MAE={results['mae']:.3f}, "
                  f"Exact={results['exact']*100:.1f}%, Pearson={results['pearson']:.3f}")

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    final_results = evaluate(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {task.upper()}")
    print(f"{'='*60}")
    print(f"MAE:      {final_results['mae']:.3f}")
    print(f"Exact:    {final_results['exact']*100:.1f}%")
    print(f"Within1:  {final_results['within_1']*100:.1f}%")
    print(f"Pearson:  {final_results['pearson']:.3f}")
    print(f"Spearman: {final_results['spearman']:.3f}")

    # ============================================================
    # SAVE MODEL (.pth)
    # ============================================================
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    model_path = os.path.join(config.MODEL_DIR, f"{task}_mamba_coral_best.pth")

    # Save model state dict and config
    save_dict = {
        'model_state_dict': best_model_state,
        'config': {
            'input_size': input_size,
            'hidden_size': config.HIDDEN_SIZE,
            'num_layers': config.NUM_LAYERS,
            'num_classes': config.NUM_CLASSES,
            'dropout': config.DROPOUT
        },
        'results': final_results,
        'task': task,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(save_dict, model_path)
    print(f"\n[SAVED] Model saved to: {model_path}")

    # Verify save
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[SAVED] File size: {file_size:.2f} MB")

    return final_results


def main():
    parser = argparse.ArgumentParser(description='Train all tasks with model saving')
    parser.add_argument('--task', type=str, default='all',
                        choices=['gait', 'finger', 'hand', 'leg', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("TRAIN ALL TASKS WITH MODEL SAVING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    tasks = ['gait', 'finger', 'hand', 'leg'] if args.task == 'all' else [args.task]

    all_results = {}

    for task in tasks:
        results = train_and_save_model(task, config, device)
        if results:
            all_results[task] = results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - ALL TASKS")
    print(f"{'='*60}")
    print(f"{'Task':<15} {'MAE':>8} {'Exact':>10} {'Pearson':>10}")
    print("-" * 45)
    for task, results in all_results.items():
        print(f"{task.upper():<15} {results['mae']:>8.3f} {results['exact']*100:>9.1f}% {results['pearson']:>10.3f}")

    print(f"\nModels saved to: {config.MODEL_DIR}/")
    print("Files:")
    for task in all_results.keys():
        print(f"  - {task}_mamba_coral_best.pth")


if __name__ == "__main__":
    main()
