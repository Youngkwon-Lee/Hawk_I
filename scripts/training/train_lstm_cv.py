"""
LSTM with K-Fold Cross-Validation
Better generalization estimation with stratified K-fold
"""
import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINED_MODELS_DIR, CACHE_DIR, ensure_dirs

# Paths (from centralized config)
OUTPUT_DIR = str(TRAINED_MODELS_DIR)
ensure_dirs()


def extract_clinical_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract clinically relevant time-series features"""
    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]
    wrist_pos = landmarks[:, 0:3]

    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1)
    dist_velocity = np.gradient(finger_distance)
    dist_accel = np.gradient(dist_velocity)

    thumb_vel = landmarks[:, 63+12:63+15]
    thumb_speed = np.linalg.norm(thumb_vel, axis=1)

    index_vel = landmarks[:, 63+24:63+27]
    index_speed = np.linalg.norm(index_vel, axis=1)

    combined_speed = thumb_speed + index_speed

    thumb_from_wrist = np.linalg.norm(thumb_pos - wrist_pos, axis=1)
    index_from_wrist = np.linalg.norm(index_pos - wrist_pos, axis=1)

    hand_size = np.maximum(thumb_from_wrist, index_from_wrist) + 0.001
    normalized_distance = finger_distance / hand_size

    features = np.stack([
        finger_distance,
        dist_velocity,
        dist_accel,
        thumb_speed,
        index_speed,
        combined_speed,
        thumb_from_wrist,
        index_from_wrist,
        normalized_distance,
        hand_size,
    ], axis=1)

    return features


class AttentionLSTM(nn.Module):
    """LSTM with Attention - proven architecture from v3"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
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

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context).squeeze()


def train_fold(model, train_loader, val_X, val_y, device, epochs=80):
    """Train model for one fold"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    return model


def evaluate_fold(model, X, y, device):
    """Evaluate model on a fold"""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        y_pred = model(X_t).cpu().numpy()

    y_pred = np.clip(y_pred, 0, 4)
    y_pred_rounded = np.round(y_pred)

    mae = np.mean(np.abs(y - y_pred))
    exact = np.mean(y_pred_rounded == y) * 100
    within1 = np.mean(np.abs(y - y_pred_rounded) <= 1) * 100

    # Correlation
    if len(np.unique(y)) > 1:
        pearson_r, _ = stats.pearsonr(y, y_pred)
    else:
        pearson_r = 0

    return {
        'mae': mae,
        'exact': exact,
        'within1': within1,
        'pearson': pearson_r,
        'predictions': y_pred,
        'targets': y
    }


def main():
    print("=" * 70)
    print("LSTM with Stratified K-Fold Cross-Validation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("\nLoading cached data...")
    with open(f"{CACHE_DIR}/train_landmarks.pkl", 'rb') as f:
        X_train_raw, y_train = pickle.load(f)

    with open(f"{CACHE_DIR}/test_landmarks.pkl", 'rb') as f:
        X_test_raw, y_test = pickle.load(f)

    # Combine for CV
    X_all_raw = np.vstack([X_train_raw, X_test_raw])
    y_all = np.concatenate([y_train, y_test])

    print(f"Total samples: {len(y_all)}")
    print(f"Class distribution:")
    for score in range(5):
        count = (y_all == score).sum()
        print(f"  Score {score}: {count} ({count/len(y_all)*100:.1f}%)")

    # Extract features
    print("\nExtracting clinical features...")
    X_all = np.array([extract_clinical_features(seq) for seq in X_all_raw])
    print(f"Feature shape: {X_all.shape}")

    # Normalize (will be done per-fold)
    input_size = X_all.shape[2]

    # K-Fold CV
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"\n{'=' * 70}")
    print(f"Starting {n_folds}-Fold Cross-Validation")
    print(f"{'=' * 70}")

    fold_results = []
    all_predictions = np.zeros(len(y_all))
    all_targets = np.zeros(len(y_all))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Split
        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val_fold = X_all[val_idx]
        y_val_fold = y_all[val_idx]

        # Normalize per fold
        mean = X_train_fold.mean(axis=(0, 1), keepdims=True)
        std = X_train_fold.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_norm = (X_train_fold - mean) / std
        X_val_norm = (X_val_fold - mean) / std

        # Tensors
        X_train_t = torch.FloatTensor(X_train_norm)
        y_train_t = torch.FloatTensor(y_train_fold)
        X_val_t = torch.FloatTensor(X_val_norm).to(device)
        y_val_t = torch.FloatTensor(y_val_fold).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Model
        model = AttentionLSTM(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)

        # Train
        model = train_fold(model, train_loader, X_val_t, y_val_t, device, epochs=80)

        # Evaluate
        results = evaluate_fold(model, X_val_norm, y_val_fold, device)
        fold_results.append(results)

        # Store predictions for overall metrics
        all_predictions[val_idx] = results['predictions']
        all_targets[val_idx] = results['targets']

        print(f"Fold {fold + 1}: MAE={results['mae']:.3f}, Exact={results['exact']:.1f}%, Within1={results['within1']:.1f}%")

    # Aggregate results
    print(f"\n{'=' * 70}")
    print("Cross-Validation Results")
    print(f"{'=' * 70}")

    mae_scores = [r['mae'] for r in fold_results]
    exact_scores = [r['exact'] for r in fold_results]
    within1_scores = [r['within1'] for r in fold_results]
    pearson_scores = [r['pearson'] for r in fold_results]

    print(f"\nMAE:      {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    print(f"Exact:    {np.mean(exact_scores):.1f}% ± {np.std(exact_scores):.1f}%")
    print(f"Within 1: {np.mean(within1_scores):.1f}% ± {np.std(within1_scores):.1f}%")
    print(f"Pearson:  {np.mean(pearson_scores):.3f} ± {np.std(pearson_scores):.3f}")

    # Overall metrics from aggregated predictions
    print(f"\n--- Overall (Aggregated) ---")
    all_pred_rounded = np.round(all_predictions)
    overall_mae = np.mean(np.abs(all_targets - all_predictions))
    overall_exact = np.mean(all_pred_rounded == all_targets) * 100
    overall_within1 = np.mean(np.abs(all_targets - all_pred_rounded) <= 1) * 100
    overall_pearson, _ = stats.pearsonr(all_targets, all_predictions)

    print(f"MAE:      {overall_mae:.3f}")
    print(f"Exact:    {overall_exact:.1f}%")
    print(f"Within 1: {overall_within1:.1f}%")
    print(f"Pearson:  {overall_pearson:.3f}")

    # Per-class accuracy
    print(f"\nPer-class accuracy (Overall):")
    for score in range(5):
        mask = all_targets == score
        if mask.sum() > 0:
            acc = (all_pred_rounded[mask] == score).mean() * 100
            mae_class = np.mean(np.abs(all_predictions[mask] - score))
            print(f"  Score {score}: Acc={acc:.1f}%, MAE={mae_class:.3f} ({mask.sum()} samples)")

    # Comparison
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<35} {'MAE':<12} {'Exact':<12} {'Within 1':<12}")
    print("-" * 70)
    print(f"{'RF (baseline)':<35} {'0.489':<12} {'54.8%':<12} {'96.3%':<12}")
    print(f"{'LSTM v3 (train/test split)':<35} {'0.547':<12} {'55.9%':<12} {'96.5%':<12}")
    print(f"{'LSTM CV (5-fold mean)':<35} {np.mean(mae_scores):.3f}{'':>9} {np.mean(exact_scores):.1f}%{'':>7} {np.mean(within1_scores):.1f}%")
    print(f"{'LSTM CV (aggregated)':<35} {overall_mae:.3f}{'':>9} {overall_exact:.1f}%{'':>7} {overall_within1:.1f}%")

    # Confidence interval
    print(f"\n95% Confidence Intervals:")
    print(f"  MAE:   [{np.mean(mae_scores) - 1.96*np.std(mae_scores):.3f}, {np.mean(mae_scores) + 1.96*np.std(mae_scores):.3f}]")
    print(f"  Exact: [{np.mean(exact_scores) - 1.96*np.std(exact_scores):.1f}%, {np.mean(exact_scores) + 1.96*np.std(exact_scores):.1f}%]")


if __name__ == "__main__":
    main()
