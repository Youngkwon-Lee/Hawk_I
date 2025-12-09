"""
LSTM v2 - Improved with Clinical Features
Key improvements:
1. Use finger distance time series (clinical relevance)
2. Class balancing with weighted loss
3. Better feature engineering
4. Attention mechanism
"""
import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINED_MODELS_DIR, CACHE_DIR, ensure_dirs

# Paths (from centralized config)
OUTPUT_DIR = str(TRAINED_MODELS_DIR)
ensure_dirs()

def extract_clinical_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract clinically relevant time-series features"""
    # landmarks: (frames, 129) - 63 positions + 63 velocities + 3 speeds

    # Key indices in original landmarks
    # Thumb tip: landmarks 4 → indices 12:15 (x,y,z)
    # Index tip: landmarks 8 → indices 24:27 (x,y,z)

    frames = landmarks.shape[0]

    # Extract positions
    thumb_pos = landmarks[:, 12:15]  # (frames, 3)
    index_pos = landmarks[:, 24:27]  # (frames, 3)

    # 1. Finger distance (aperture) - most important for finger tapping
    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1)  # (frames,)

    # 2. Opening/closing velocity (derivative of distance)
    dist_velocity = np.gradient(finger_distance)

    # 3. Acceleration
    dist_accel = np.gradient(dist_velocity)

    # 4. Thumb velocity magnitude
    thumb_vel = landmarks[:, 63+12:63+15]  # velocity features
    thumb_speed = np.linalg.norm(thumb_vel, axis=1)

    # 5. Index velocity magnitude
    index_vel = landmarks[:, 63+24:63+27]
    index_speed = np.linalg.norm(index_vel, axis=1)

    # 6. Combined finger speed
    combined_speed = thumb_speed + index_speed

    # 7. Distance from wrist (stability measure)
    wrist_pos = landmarks[:, 0:3]  # landmark 0 is wrist
    thumb_from_wrist = np.linalg.norm(thumb_pos - wrist_pos, axis=1)
    index_from_wrist = np.linalg.norm(index_pos - wrist_pos, axis=1)

    # 8. Normalized distance (relative to hand size)
    hand_size = np.maximum(thumb_from_wrist, index_from_wrist) + 0.001
    normalized_distance = finger_distance / hand_size

    # Stack all features: (frames, 10)
    features = np.stack([
        finger_distance,        # 0: Aperture
        dist_velocity,          # 1: Opening/closing velocity
        dist_accel,             # 2: Acceleration
        thumb_speed,            # 3: Thumb speed
        index_speed,            # 4: Index speed
        combined_speed,         # 5: Combined speed
        thumb_from_wrist,       # 6: Thumb reach
        index_from_wrist,       # 7: Index reach
        normalized_distance,    # 8: Normalized aperture
        hand_size,              # 9: Hand size (normalization reference)
    ], axis=1)

    return features


def process_cached_data():
    """Convert cached raw landmarks to clinical features"""
    print("Loading cached data...")

    with open(f"{CACHE_DIR}/train_landmarks.pkl", 'rb') as f:
        X_train_raw, y_train = pickle.load(f)

    with open(f"{CACHE_DIR}/test_landmarks.pkl", 'rb') as f:
        X_test_raw, y_test = pickle.load(f)

    print(f"Raw train shape: {X_train_raw.shape}")
    print(f"Raw test shape: {X_test_raw.shape}")

    # Convert to clinical features
    print("Extracting clinical features...")
    X_train = np.array([extract_clinical_features(seq) for seq in X_train_raw])
    X_test = np.array([extract_clinical_features(seq) for seq in X_test_raw])

    print(f"Clinical train shape: {X_train.shape}")
    print(f"Clinical test shape: {X_test.shape}")

    # Normalize features (per-feature standardization)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


class AttentionLSTM(nn.Module):
    """LSTM with Attention for temporal importance"""
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

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)

        # Output
        output = self.classifier(context)
        return output.squeeze()


def calculate_class_weights(y):
    """Calculate balanced class weights"""
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)

    sample_weights = np.array([weights[int(label)] for label in y])
    return sample_weights, dict(zip(classes.astype(int), weights))


def main():
    print("=" * 60)
    print("LSTM v2 - Clinical Features + Attention")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load and process data
    X_train, y_train, X_test, y_test = process_cached_data()

    # Class distribution
    print("\nClass distribution (Train):")
    for score in range(5):
        count = (y_train == score).sum()
        print(f"  Score {score}: {count} ({count/len(y_train)*100:.1f}%)")

    # Calculate class weights for balanced training
    sample_weights, class_weights = calculate_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

    # Model
    input_size = X_train.shape[2]
    model = AttentionLSTM(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    # Weighted MSE loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    # Validation split
    val_size = int(len(X_train_t) * 0.15)
    indices = torch.randperm(len(X_train_t))
    val_indices = indices[:val_size]

    X_val = X_train_t[val_indices].to(device)
    y_val = y_train_t[val_indices].to(device)

    for epoch in range(100):
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
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/lstm_v2_finger_tapping.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100 - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/lstm_v2_finger_tapping.pth"))

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

    mae = np.mean(np.abs(y_test_np - y_pred))
    exact = np.mean(y_pred_rounded == y_test_np) * 100
    within1 = np.mean(np.abs(y_test_np - y_pred_rounded) <= 1) * 100

    print(f"LSTM v2 (Clinical Features + Attention):")
    print(f"  MAE: {mae:.3f}")
    print(f"  Exact Accuracy: {exact:.1f}%")
    print(f"  Within 1 Point: {within1:.1f}%")

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"RF:           MAE=0.489, Exact=54.8%, Within 1pt=96.3%")
    print(f"LSTM v1:      MAE=0.737, Exact=42.1%, Within 1pt=89.0%")
    print(f"LSTM v2:      MAE={mae:.3f}, Exact={exact:.1f}%, Within 1pt={within1:.1f}%")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for score in range(5):
        mask = y_test_np == score
        if mask.sum() > 0:
            acc = (y_pred_rounded[mask] == score).mean() * 100
            print(f"  Score {score}: {acc:.1f}% ({mask.sum()} samples)")


if __name__ == "__main__":
    main()
