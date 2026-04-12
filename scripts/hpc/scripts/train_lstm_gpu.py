"""
PD4T Finger Tapping LSTM Training - HPC GPU Version
Optimized for HPC Innovation Hub (V100 GPU)

Usage:
    cd ~/hawkeye
    python scripts/train_lstm_gpu.py

Features:
- GPU acceleration with CUDA
- Mixed Precision Training (FP16)
- K-Fold Cross-Validation
- Checkpoint & Resume support
- Multiple model architectures (LSTM, BiLSTM+Attention, Transformer)
"""
import os
import sys
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import argparse
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
class Config:
    # Paths (HPC)
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    RESULT_DIR = "./results"

    # Data
    SEQUENCE_LENGTH = 150
    NUM_FEATURES = 10  # Clinical features

    # Training
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    PATIENCE = 15

    # K-Fold
    N_FOLDS = 5

    # Model
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3

    # GPU
    USE_AMP = True  # Mixed precision


# ============================================================
# Models
# ============================================================
class AttentionLSTM(nn.Module):
    """Bidirectional LSTM with Attention"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
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
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context).squeeze(-1)


class TransformerModel(nn.Module):
    """Transformer Encoder for sequence classification"""
    def __init__(self, input_size, hidden_size=128, num_layers=4, dropout=0.3, nhead=8):
        super().__init__()

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x).squeeze(-1)


class ConvLSTM(nn.Module):
    """1D CNN + LSTM hybrid"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq/2, 128)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last timestep
        return self.classifier(x).squeeze(-1)


# ============================================================
# Data Loading
# ============================================================
def load_data(config):
    """Load preprocessed data"""
    print("Loading data...")

    train_path = os.path.join(config.DATA_DIR, "train_data.pkl")
    test_path = os.path.join(config.DATA_DIR, "test_data.pkl")

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train labels: {np.bincount(y_train.astype(int), minlength=5)}")
    print(f"Test labels: {np.bincount(y_test.astype(int), minlength=5)}")

    return X_train, y_train, X_test, y_test


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        preds = model(X_t).cpu().numpy()
        preds = np.clip(preds, 0, 4)

    mae = np.mean(np.abs(y - preds))
    preds_rounded = np.round(preds)
    exact = np.mean(preds_rounded == y) * 100
    within1 = np.mean(np.abs(y - preds_rounded) <= 1) * 100

    return {'mae': mae, 'exact': exact, 'within1': within1, 'preds': preds}


def kfold_cv(X, y, config, device, model_class):
    """K-Fold Cross-Validation"""
    print(f"\n{'='*60}")
    print(f"K-Fold Cross-Validation (K={config.N_FOLDS})")
    print(f"Model: {model_class.__name__}")
    print('='*60)

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    y_binned = np.clip(y, 0, 3).astype(int)

    fold_results = []
    all_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        print(f"\n--- Fold {fold+1}/{config.N_FOLDS} ---")

        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Normalize
        mean = X_train_fold.mean(axis=(0, 1), keepdims=True)
        std = X_train_fold.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_norm = (X_train_fold - mean) / std
        X_val_norm = (X_val_fold - mean) / std

        # Model
        input_size = X.shape[2]
        model = model_class(
            input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ).to(device)

        # DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_fold)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        scaler = GradScaler() if config.USE_AMP else None

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(config.EPOCHS):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer,
                scaler, device, config.USE_AMP
            )

            model.eval()
            with torch.no_grad():
                val_X_t = torch.FloatTensor(X_val_norm).to(device)
                val_preds = model(val_X_t)
                val_loss = criterion(
                    val_preds,
                    torch.FloatTensor(y_val_fold).to(device)
                ).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        model.to(device)

        # Evaluate
        results = evaluate(model, X_val_norm, y_val_fold, device)
        all_preds[val_idx] = results['preds']
        fold_results.append(results)

        print(f"  MAE: {results['mae']:.3f}, Exact: {results['exact']:.1f}%, Within1: {results['within1']:.1f}%")

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    # Summary
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_exact = np.mean([r['exact'] for r in fold_results])
    avg_within1 = np.mean([r['within1'] for r in fold_results])

    # Overall metrics
    overall_mae = np.mean(np.abs(y - all_preds))
    overall_exact = np.mean(np.round(all_preds) == y) * 100
    overall_within1 = np.mean(np.abs(y - np.round(all_preds)) <= 1) * 100
    pearson, _ = stats.pearsonr(y, all_preds)

    print(f"\n{'='*60}")
    print(f"CV Results - {model_class.__name__}")
    print('='*60)
    print(f"Average MAE: {avg_mae:.3f} (+/- {np.std([r['mae'] for r in fold_results]):.3f})")
    print(f"Average Exact: {avg_exact:.1f}%")
    print(f"Average Within1: {avg_within1:.1f}%")
    print(f"Pearson r: {pearson:.3f}")

    return {
        'model_name': model_class.__name__,
        'avg_mae': avg_mae,
        'avg_exact': avg_exact,
        'avg_within1': avg_within1,
        'pearson': pearson,
        'fold_results': fold_results
    }


def train_final_model(X_train, y_train, X_test, y_test, config, device, model_class):
    """Train final model and evaluate on test set"""
    print(f"\n{'='*60}")
    print(f"Training Final Model: {model_class.__name__}")
    print('='*60)

    # Normalize
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    norm_params = {'mean': mean.squeeze(), 'std': std.squeeze()}

    # Model
    input_size = X_train.shape[2]
    model = model_class(
        input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler() if config.USE_AMP else None

    best_loss = float('inf')
    best_state = None

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, config.USE_AMP
        )

        scheduler.step(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}")

    model.load_state_dict(best_state)
    model.to(device)

    # Test evaluation
    test_results = evaluate(model, X_test_norm, y_test, device)

    print(f"\nTest Results:")
    print(f"  MAE: {test_results['mae']:.3f}")
    print(f"  Exact: {test_results['exact']:.1f}%")
    print(f"  Within1: {test_results['within1']:.1f}%")

    return model, norm_params, test_results


def main():
    parser = argparse.ArgumentParser(description='PD4T LSTM Training')
    parser.add_argument('--model', type=str, default='all',
                       choices=['lstm', 'transformer', 'convlstm', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    args = parser.parse_args()

    # Config
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.USE_AMP = not args.no_amp

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print("PD4T Finger Tapping LSTM Training - HPC GPU")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Mixed Precision: {config.USE_AMP}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")

    # Create directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test = load_data(config)

    # Models to train
    models = {
        'lstm': AttentionLSTM,
        'transformer': TransformerModel,
        'convlstm': ConvLSTM
    }

    if args.model == 'all':
        model_list = list(models.values())
    else:
        model_list = [models[args.model]]

    # Results
    all_results = []

    for model_class in model_list:
        # K-Fold CV
        cv_results = kfold_cv(X_train, y_train, config, device, model_class)
        all_results.append(cv_results)

        # Train final model
        model, norm_params, test_results = train_final_model(
            X_train, y_train, X_test, y_test, config, device, model_class
        )

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_class.__name__.lower()
        save_path = os.path.join(config.MODEL_DIR, f"{model_name}_finger_tapping_{timestamp}.pth")

        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model_class.__name__,
            'input_size': X_train.shape[2],
            'hidden_size': config.HIDDEN_SIZE,
            'num_layers': config.NUM_LAYERS,
            'dropout': config.DROPOUT,
            'norm_params': norm_params,
            'cv_results': cv_results,
            'test_results': test_results
        }, save_path)

        print(f"Model saved: {save_path}")

        # Clear GPU
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'CV MAE':>10} {'CV Exact':>12} {'Pearson':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['model_name']:<20} {r['avg_mae']:>10.3f} {r['avg_exact']:>11.1f}% {r['pearson']:>10.3f}")

    # Save results summary
    result_path = os.path.join(config.RESULT_DIR, f"training_results_{timestamp}.txt")
    with open(result_path, 'w') as f:
        f.write(f"PD4T Finger Tapping Training Results\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Device: {device}\n\n")
        for r in all_results:
            f.write(f"{r['model_name']}:\n")
            f.write(f"  CV MAE: {r['avg_mae']:.3f}\n")
            f.write(f"  CV Exact: {r['avg_exact']:.1f}%\n")
            f.write(f"  CV Within1: {r['avg_within1']:.1f}%\n")
            f.write(f"  Pearson r: {r['pearson']:.3f}\n\n")

    print(f"\nResults saved: {result_path}")
    print("\nTraining Complete!")


if __name__ == "__main__":
    main()
