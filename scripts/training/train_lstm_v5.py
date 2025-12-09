"""
LSTM v5 - Research-Based Improvements
Based on recent literature:
1. Time-Series Augmentation (Jittering, Time Warping, Window Slicing)
2. Ordinal Regression Loss (respects order relationship)
3. SMOTE for minority class oversampling
4. Dilated convolution layer before LSTM

References:
- FastEval Parkinsonism (Nature, 2024)
- Multi-Shared-Task Self-Supervised CNN-LSTM (MDPI, 2024)
- Time Series Ordinal Classification (arXiv, 2023)
"""
import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINED_MODELS_DIR, CACHE_DIR, ensure_dirs

# Paths (from centralized config)
OUTPUT_DIR = str(TRAINED_MODELS_DIR)
ensure_dirs()


# ============================================================
# 1. TIME-SERIES AUGMENTATION
# ============================================================

def jitter(x, sigma=0.03):
    """Add random noise to time series"""
    return x + np.random.normal(0, sigma, x.shape)


def scaling(x, sigma=0.1):
    """Scale time series by random factor"""
    factor = np.random.normal(1.0, sigma, (1, x.shape[1]))
    return x * factor


def time_warp(x, sigma=0.2, num_knots=4):
    """Warp time axis using smooth random curves"""
    orig_steps = np.arange(x.shape[0])

    # Create random warping anchors
    random_warps = np.random.normal(1.0, sigma, num_knots + 2)
    warp_steps = np.linspace(0, x.shape[0] - 1, num_knots + 2)

    # Interpolate to get smooth warping
    time_warp_fn = interp1d(warp_steps, warp_steps * random_warps,
                            kind='quadratic', fill_value='extrapolate')
    warped_steps = time_warp_fn(orig_steps)
    warped_steps = np.clip(warped_steps, 0, x.shape[0] - 1)

    # Apply warping to each feature
    result = np.zeros_like(x)
    for i in range(x.shape[1]):
        result[:, i] = np.interp(orig_steps, warped_steps, x[:, i])

    return result


def magnitude_warp(x, sigma=0.2, num_knots=4):
    """Warp magnitude using smooth random curves"""
    orig_steps = np.arange(x.shape[0])

    # Create random magnitude warping
    random_warps = np.random.normal(1.0, sigma, num_knots + 2)
    warp_steps = np.linspace(0, x.shape[0] - 1, num_knots + 2)

    # Interpolate
    mag_warp_fn = interp1d(warp_steps, random_warps, kind='quadratic',
                           fill_value='extrapolate')
    mag_warp = mag_warp_fn(orig_steps).reshape(-1, 1)

    return x * mag_warp


def window_slice(x, reduce_ratio=0.9):
    """Randomly slice a window from time series"""
    target_len = int(x.shape[0] * reduce_ratio)
    if target_len >= x.shape[0]:
        return x

    start = np.random.randint(0, x.shape[0] - target_len)
    sliced = x[start:start + target_len]

    # Resample to original length
    orig_steps = np.arange(x.shape[0])
    new_steps = np.linspace(0, target_len - 1, x.shape[0])

    result = np.zeros_like(x)
    for i in range(x.shape[1]):
        result[:, i] = np.interp(new_steps, np.arange(target_len), sliced[:, i])

    return result


def augment_sequence(x, aug_prob=0.5):
    """Apply random augmentations to a sequence"""
    augmented = x.copy()

    if np.random.random() < aug_prob:
        augmented = jitter(augmented, sigma=0.02)

    if np.random.random() < aug_prob:
        augmented = scaling(augmented, sigma=0.1)

    if np.random.random() < aug_prob * 0.5:  # Less frequent
        augmented = time_warp(augmented, sigma=0.15)

    if np.random.random() < aug_prob * 0.5:
        augmented = magnitude_warp(augmented, sigma=0.15)

    return augmented


# ============================================================
# 2. SMOTE FOR TIME SERIES
# ============================================================

def smote_timeseries(X, y, target_counts=None):
    """
    SMOTE for time series data
    Generates synthetic samples by interpolating between neighbors
    """
    from sklearn.neighbors import NearestNeighbors

    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    if target_counts is None:
        # Balance to max class count
        target_counts = {c: max_count for c in classes}

    X_resampled = [X]
    y_resampled = [y]

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        cls_samples = X[cls_indices]
        current_count = len(cls_indices)
        needed = target_counts.get(cls, current_count) - current_count

        if needed <= 0 or current_count < 2:
            continue

        # Flatten for KNN
        cls_flat = cls_samples.reshape(current_count, -1)

        # Find k nearest neighbors
        k = min(5, current_count - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(cls_flat)

        synthetic = []
        for _ in range(needed):
            # Random sample
            idx = np.random.randint(0, current_count)
            sample = cls_samples[idx]

            # Find neighbors
            distances, indices = nn.kneighbors([cls_flat[idx]])
            neighbor_idx = indices[0, np.random.randint(1, k + 1)]
            neighbor = cls_samples[neighbor_idx]

            # Interpolate
            alpha = np.random.random()
            new_sample = sample + alpha * (neighbor - sample)
            synthetic.append(new_sample)

        if synthetic:
            X_resampled.append(np.array(synthetic))
            y_resampled.append(np.full(len(synthetic), cls))

    return np.vstack(X_resampled), np.concatenate(y_resampled)


# ============================================================
# 3. ORDINAL REGRESSION LOSS
# ============================================================

class OrdinalLoss(nn.Module):
    """
    Ordinal regression loss that respects class ordering
    Uses cumulative link model approach
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        """
        predictions: (batch,) continuous predictions
        targets: (batch,) integer labels 0-4
        """
        # MSE component
        mse_loss = F.mse_loss(predictions, targets.float())

        # Ordinal penalty: penalize predictions that skip classes
        pred_rounded = torch.round(predictions).clamp(0, self.num_classes - 1)
        ordinal_error = torch.abs(pred_rounded - targets.float())

        # Extra penalty for errors > 1 (skipping classes)
        skip_penalty = F.relu(ordinal_error - 1.0).mean()

        return mse_loss + 0.3 * skip_penalty


class CumulativeOrdinalLoss(nn.Module):
    """
    Cumulative link ordinal loss
    Predicts P(Y > k) for each threshold k
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        # Learnable thresholds
        self.thresholds = nn.Parameter(torch.arange(num_classes - 1).float())

    def forward(self, logits, targets):
        """
        logits: (batch, 1) - latent score
        targets: (batch,) - class labels 0 to num_classes-1
        """
        # Compute P(Y > k) for each k
        # Using sigmoid of (latent - threshold)
        batch_size = logits.size(0)

        # Expand for broadcasting: (batch, num_thresholds)
        logits_expanded = logits.expand(-1, self.num_classes - 1)
        thresholds = self.thresholds.unsqueeze(0).expand(batch_size, -1)

        # Cumulative probabilities
        cum_probs = torch.sigmoid(logits_expanded - thresholds)

        # Convert to class probabilities
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        probs = torch.zeros(batch_size, self.num_classes, device=logits.device)
        probs[:, 0] = 1 - cum_probs[:, 0]
        for k in range(1, self.num_classes - 1):
            probs[:, k] = cum_probs[:, k-1] - cum_probs[:, k]
        probs[:, -1] = cum_probs[:, -1]

        # Cross entropy loss
        probs = probs.clamp(min=1e-7)
        targets_onehot = F.one_hot(targets.long(), self.num_classes).float()
        loss = -torch.sum(targets_onehot * torch.log(probs), dim=1).mean()

        return loss


# ============================================================
# 4. MODEL WITH DILATED CONVOLUTION
# ============================================================

class DilatedConvLSTM(nn.Module):
    """
    Dilated CNN + LSTM architecture inspired by FastEval Parkinsonism
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()

        # Dilated convolution layers (capture multi-scale temporal patterns)
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=4, dilation=4)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq, features)

        # Conv expects (batch, channels, seq)
        x = x.permute(0, 2, 1)

        # Dilated convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Back to (batch, seq, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        # Output
        return self.classifier(context).squeeze()


# ============================================================
# 5. FEATURE EXTRACTION (from v3)
# ============================================================

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


def process_cached_data(use_smote=True, use_augmentation=True):
    """Load and process cached data with augmentation and SMOTE"""
    print("Loading cached data...")

    with open(f"{CACHE_DIR}/train_landmarks.pkl", 'rb') as f:
        X_train_raw, y_train = pickle.load(f)

    with open(f"{CACHE_DIR}/test_landmarks.pkl", 'rb') as f:
        X_test_raw, y_test = pickle.load(f)

    print(f"Raw train shape: {X_train_raw.shape}")
    print(f"Raw test shape: {X_test_raw.shape}")

    # Extract clinical features
    print("Extracting clinical features...")
    X_train = np.array([extract_clinical_features(seq) for seq in X_train_raw])
    X_test = np.array([extract_clinical_features(seq) for seq in X_test_raw])

    print(f"Clinical train shape: {X_train.shape}")

    # Apply SMOTE for minority classes
    if use_smote:
        print("\nApplying SMOTE for minority classes...")
        classes, counts = np.unique(y_train, return_counts=True)
        print(f"Before SMOTE: {dict(zip(classes.astype(int), counts))}")

        # Target: boost Score 3 and 4
        target_counts = {
            0: int(counts[0]),
            1: int(counts[1]),
            2: int(counts[2]),
            3: min(300, int(counts[1] * 0.15)),  # Boost to 15% of majority
            4: min(150, int(counts[1] * 0.08)),  # Boost to 8% of majority
        }

        X_train, y_train = smote_timeseries(X_train, y_train, target_counts)

        classes, counts = np.unique(y_train, return_counts=True)
        print(f"After SMOTE: {dict(zip(classes.astype(int), counts))}")

    # Apply augmentation
    if use_augmentation:
        print("\nApplying time-series augmentation...")
        augmented_X = []
        augmented_y = []

        for i in range(len(X_train)):
            augmented_X.append(X_train[i])
            augmented_y.append(y_train[i])

            # Extra augmentation for minority classes
            n_aug = 2 if y_train[i] >= 3 else 1
            for _ in range(n_aug):
                aug_seq = augment_sequence(X_train[i], aug_prob=0.6)
                augmented_X.append(aug_seq)
                augmented_y.append(y_train[i])

        X_train = np.array(augmented_X)
        y_train = np.array(augmented_y)
        print(f"After augmentation: {X_train.shape}")

    # Normalize
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


# ============================================================
# 6. TRAINING
# ============================================================

def train_model(model_type='dilated_lstm', loss_type='ordinal'):
    """Train model with specified configuration"""
    print("=" * 70)
    print(f"LSTM v5 - {model_type} with {loss_type} loss")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    X_train, y_train, X_test, y_test = process_cached_data(
        use_smote=True,
        use_augmentation=True
    )

    # Class distribution
    print("\nFinal class distribution (Train):")
    for score in range(5):
        count = (y_train == score).sum()
        print(f"  Score {score}: {count} ({count/len(y_train)*100:.1f}%)")

    # Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model
    input_size = X_train.shape[2]

    if model_type == 'dilated_lstm':
        model = DilatedConvLSTM(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)
    else:
        from train_lstm_v2 import AttentionLSTM
        model = AttentionLSTM(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    # Loss
    if loss_type == 'ordinal':
        criterion = OrdinalLoss(num_classes=5)
    elif loss_type == 'cumulative':
        criterion = CumulativeOrdinalLoss(num_classes=5).to(device)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    # Validation split
    val_size = int(len(X_train_t) * 0.15)
    indices = torch.randperm(len(X_train_t))
    val_indices = indices[:val_size]

    X_val = X_train_t[val_indices].to(device)
    y_val = y_train_t[val_indices].to(device)

    # Training
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    model_path = f"{OUTPUT_DIR}/lstm_v5_{model_type}_{loss_type}.pth"

    for epoch in range(150):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 15 == 0:
            print(f"Epoch {epoch+1}/150 - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(model_path))

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()

    y_pred = np.clip(y_pred, 0, 4)
    y_pred_rounded = np.round(y_pred)
    y_test_np = y_test

    # Metrics
    mae = np.mean(np.abs(y_test_np - y_pred))
    rmse = np.sqrt(np.mean((y_test_np - y_pred) ** 2))
    exact = np.mean(y_pred_rounded == y_test_np) * 100
    within1 = np.mean(np.abs(y_test_np - y_pred_rounded) <= 1) * 100

    # R² and correlations
    ss_res = np.sum((y_test_np - y_pred) ** 2)
    ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    from scipy import stats
    pearson_r, _ = stats.pearsonr(y_test_np, y_pred)
    spearman_r, _ = stats.spearmanr(y_test_np, y_pred)

    print(f"\nResults ({model_type} + {loss_type}):")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  Pearson r: {pearson_r:.3f}")
    print(f"  Spearman ρ: {spearman_r:.3f}")
    print(f"  Exact Accuracy: {exact:.1f}%")
    print(f"  Within 1 Point: {within1:.1f}%")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for score in range(5):
        mask = y_test_np == score
        if mask.sum() > 0:
            acc = (y_pred_rounded[mask] == score).mean() * 100
            mae_class = np.mean(np.abs(y_pred[mask] - score))
            print(f"  Score {score}: Acc={acc:.1f}%, MAE={mae_class:.3f} ({mask.sum()} samples)")

    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'pearson': pearson_r, 'spearman': spearman_r,
        'exact': exact, 'within1': within1
    }


def main():
    print("\n" + "=" * 70)
    print("LSTM v5 - Research-Based Improvements")
    print("  - Time-Series Augmentation")
    print("  - SMOTE for Minority Classes")
    print("  - Ordinal Regression Loss")
    print("  - Dilated CNN + LSTM")
    print("=" * 70)

    # Train with Dilated CNN-LSTM + Ordinal Loss
    results = train_model(model_type='dilated_lstm', loss_type='ordinal')

    # Comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<30} {'MAE':<8} {'Exact':<10} {'Within 1':<10} {'R²':<8} {'Pearson':<8}")
    print("-" * 75)
    print(f"{'RF (baseline)':<30} {'0.489':<8} {'54.8%':<10} {'96.3%':<10} {'-':<8} {'-':<8}")
    print(f"{'LSTM v3 (best previous)':<30} {'0.547':<8} {'55.9%':<10} {'96.5%':<10} {'0.20':<8} {'0.46':<8}")
    print(f"{'LSTM v5 (research-based)':<30} {results['mae']:.3f}{'':>5} {results['exact']:.1f}%{'':>5} {results['within1']:.1f}%{'':>5} {results['r2']:.2f}{'':>5} {results['pearson']:.2f}")


if __name__ == "__main__":
    main()
