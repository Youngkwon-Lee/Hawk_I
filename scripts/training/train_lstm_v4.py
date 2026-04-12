"""
LSTM v4 - Top Correlated Features Only
Based on correlation analysis:
- Variability metrics are most predictive (std, variability, range)
- Negative correlation: higher variability = lower score (more normal)

Top features:
1. dist_accel_std (r=-0.446)
2. dist_velocity_variability (r=-0.438)
3. dist_velocity_std (r=-0.436)
4. finger_distance_variability (r=-0.430)
5. dist_accel_variability (r=-0.425)
"""
import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINED_MODELS_DIR, CACHE_DIR, ensure_dirs

# Paths (from centralized config)
OUTPUT_DIR = str(TRAINED_MODELS_DIR)
ensure_dirs()

def extract_clinical_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract clinically relevant time-series features"""
    frames = landmarks.shape[0]

    # Key indices: Thumb tip (4) -> 12:15, Index tip (8) -> 24:27
    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]
    wrist_pos = landmarks[:, 0:3]

    # 1. Finger distance (aperture)
    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1)

    # 2. Opening/closing velocity
    dist_velocity = np.gradient(finger_distance)

    # 3. Acceleration
    dist_accel = np.gradient(dist_velocity)

    # 4. Thumb velocity magnitude
    thumb_vel = landmarks[:, 63+12:63+15]
    thumb_speed = np.linalg.norm(thumb_vel, axis=1)

    # 5. Index velocity magnitude
    index_vel = landmarks[:, 63+24:63+27]
    index_speed = np.linalg.norm(index_vel, axis=1)

    # 6. Combined finger speed
    combined_speed = thumb_speed + index_speed

    # 7-8. Distance from wrist
    thumb_from_wrist = np.linalg.norm(thumb_pos - wrist_pos, axis=1)
    index_from_wrist = np.linalg.norm(index_pos - wrist_pos, axis=1)

    # 9. Normalized distance
    hand_size = np.maximum(thumb_from_wrist, index_from_wrist) + 0.001
    normalized_distance = finger_distance / hand_size

    # Stack base features: (frames, 10)
    base_features = np.stack([
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

    return base_features


def compute_variability_features(base_features: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute rolling variability features - these are the most predictive!

    For each base feature, compute rolling:
    - std (standard deviation)
    - range (max - min)
    - CV (coefficient of variation)
    """
    frames, num_features = base_features.shape

    # Output: original features + variability features
    variability_features = []

    for i in range(frames):
        start = max(0, i - window_size // 2)
        end = min(frames, i + window_size // 2 + 1)
        window = base_features[start:end]

        frame_features = []
        for f in range(num_features):
            feat_window = window[:, f]

            # Original value
            frame_features.append(base_features[i, f])

            # Rolling std (most predictive!)
            std = np.std(feat_window)
            frame_features.append(std)

            # Rolling range
            range_val = np.max(feat_window) - np.min(feat_window)
            frame_features.append(range_val)

            # Rolling CV
            mean = np.mean(feat_window)
            cv = std / (abs(mean) + 0.001)
            frame_features.append(cv)

        variability_features.append(frame_features)

    return np.array(variability_features)


def extract_top_features_only(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract only the TOP correlated features based on analysis:
    - Focus on variability metrics for dist_accel, dist_velocity, finger_distance
    """
    frames = landmarks.shape[0]

    # Extract base signals
    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]

    # Core signals
    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1)
    dist_velocity = np.gradient(finger_distance)
    dist_accel = np.gradient(dist_velocity)

    # Compute rolling variability (window=15 frames)
    window = 15

    def rolling_stats(signal):
        """Compute rolling std, range, cv"""
        result = []
        for i in range(len(signal)):
            start = max(0, i - window // 2)
            end = min(len(signal), i + window // 2 + 1)
            w = signal[start:end]

            std = np.std(w)
            range_val = np.max(w) - np.min(w)
            mean = np.mean(w)
            cv = std / (abs(mean) + 0.001)

            result.append([std, range_val, cv])
        return np.array(result)

    # Get variability features for top signals
    fd_stats = rolling_stats(finger_distance)  # (frames, 3)
    dv_stats = rolling_stats(dist_velocity)    # (frames, 3)
    da_stats = rolling_stats(dist_accel)       # (frames, 3)

    # Also include the raw signals for context
    features = np.column_stack([
        finger_distance,       # 0: raw distance
        dist_velocity,         # 1: raw velocity
        dist_accel,            # 2: raw acceleration
        fd_stats,              # 3-5: finger_distance std, range, cv
        dv_stats,              # 6-8: dist_velocity std, range, cv
        da_stats,              # 9-11: dist_accel std, range, cv
    ])

    return features  # (frames, 12)


def process_cached_data(use_top_features: bool = True):
    """Load and process cached data"""
    print("Loading cached data...")

    with open(f"{CACHE_DIR}/train_landmarks.pkl", 'rb') as f:
        X_train_raw, y_train = pickle.load(f)

    with open(f"{CACHE_DIR}/test_landmarks.pkl", 'rb') as f:
        X_test_raw, y_test = pickle.load(f)

    print(f"Raw train shape: {X_train_raw.shape}")
    print(f"Raw test shape: {X_test_raw.shape}")

    # Convert to features
    print("Extracting features...")
    if use_top_features:
        print("Using TOP CORRELATED features only (12 features)")
        X_train = np.array([extract_top_features_only(seq) for seq in X_train_raw])
        X_test = np.array([extract_top_features_only(seq) for seq in X_test_raw])
    else:
        print("Using full variability features (40 features)")
        X_train = np.array([compute_variability_features(extract_clinical_features(seq)) for seq in X_train_raw])
        X_test = np.array([compute_variability_features(extract_clinical_features(seq)) for seq in X_test_raw])

    print(f"Feature train shape: {X_train.shape}")
    print(f"Feature test shape: {X_test.shape}")

    # Normalize
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


class SimpleLSTM(nn.Module):
    """Simpler LSTM for focused features"""
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last timestep
        out = lstm_out[:, -1, :]
        return self.classifier(out).squeeze()


class AttentionLSTM(nn.Module):
    """LSTM with attention for variability features"""
    def __init__(self, input_size, hidden_size=48, num_layers=2, dropout=0.25):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 24),
            nn.Tanh(),
            nn.Linear(24, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Attention weights
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)

        return self.classifier(context).squeeze()


def train_and_evaluate(model_name: str, use_attention: bool = True, use_top_features: bool = True):
    """Train and evaluate model"""
    print("=" * 60)
    print(f"LSTM v4 - {model_name}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    X_train, y_train, X_test, y_test = process_cached_data(use_top_features)

    # Class distribution
    print("\nClass distribution (Train):")
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
    if use_attention:
        model = AttentionLSTM(input_size, hidden_size=48, num_layers=2, dropout=0.25).to(device)
    else:
        model = SimpleLSTM(input_size, hidden_size=32, num_layers=1, dropout=0.2).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    # Training setup - pure regression
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
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    model_path = f"{OUTPUT_DIR}/lstm_v4_{model_name}.pth"

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
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

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

    print(f"\n{model_name} Results:")
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
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson_r,
        'spearman': spearman_r,
        'exact': exact,
        'within1': within1
    }


def main():
    print("\n" + "=" * 70)
    print("LSTM v4 - Top Correlated Features (Variability Focus)")
    print("=" * 70)

    # Train with top features + attention
    results = train_and_evaluate("top_features_attention", use_attention=True, use_top_features=True)

    # Comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} {'MAE':<8} {'Exact':<10} {'Within 1':<10} {'R²':<8} {'Pearson':<8}")
    print("-" * 70)
    print(f"{'RF (baseline)':<25} {'0.489':<8} {'54.8%':<10} {'96.3%':<10} {'-':<8} {'-':<8}")
    print(f"{'LSTM v1 (raw)':<25} {'0.737':<8} {'42.1%':<10} {'89.0%':<10} {'-0.371':<8} {'0.24':<8}")
    print(f"{'LSTM v2 (clinical)':<25} {'0.610':<8} {'49.2%':<10} {'93.5%':<10} {'0.001':<8} {'0.40':<8}")
    print(f"{'LSTM v3 (regression)':<25} {'0.547':<8} {'55.9%':<10} {'96.5%':<10} {'0.20':<8} {'0.46':<8}")
    print(f"{'LSTM v4 (top features)':<25} {results['mae']:.3f}{'':>5} {results['exact']:.1f}%{'':>5} {results['within1']:.1f}%{'':>5} {results['r2']:.2f}{'':>5} {results['pearson']:.2f}")


if __name__ == "__main__":
    main()
