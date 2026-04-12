"""
LSTM Production Model + RF Ensemble for Finger Tapping UPDRS Scoring
Trains final model on all data and creates RF+LSTM ensemble
"""
import os
import sys
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
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


def extract_statistical_features(sequence: np.ndarray) -> np.ndarray:
    """Extract statistical features from time-series for RF model"""
    features = []

    for feat_idx in range(sequence.shape[1]):
        feat = sequence[:, feat_idx]
        features.extend([
            np.mean(feat),
            np.std(feat),
            np.min(feat),
            np.max(feat),
            np.percentile(feat, 25),
            np.percentile(feat, 75),
            np.median(feat),
            stats.skew(feat) if len(feat) > 2 else 0,
            stats.kurtosis(feat) if len(feat) > 3 else 0,
        ])

    # Add temporal features
    finger_dist = sequence[:, 0]
    peaks = np.where((finger_dist[1:-1] > finger_dist[:-2]) &
                     (finger_dist[1:-1] > finger_dist[2:]))[0]
    features.append(len(peaks))  # Number of taps

    if len(peaks) > 1:
        intervals = np.diff(peaks)
        features.extend([
            np.mean(intervals),
            np.std(intervals),
            np.std(intervals) / (np.mean(intervals) + 0.001),  # CV
        ])
    else:
        features.extend([0, 0, 0])

    return np.array(features)


class AttentionLSTM(nn.Module):
    """LSTM with Attention - Production Model"""
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


class EnsembleModel:
    """RF + LSTM Ensemble for Finger Tapping Scoring"""

    def __init__(self, lstm_model, rf_model, norm_params, device='cpu'):
        self.lstm_model = lstm_model
        self.rf_model = rf_model
        self.norm_params = norm_params
        self.device = device
        self.lstm_weight = 0.6  # LSTM has better CV performance
        self.rf_weight = 0.4

    def predict(self, landmarks_sequence: np.ndarray) -> dict:
        """
        Predict UPDRS score from raw landmark sequence

        Args:
            landmarks_sequence: (frames, 129) raw landmark features

        Returns:
            dict with predictions and confidence
        """
        # Extract clinical features
        clinical_features = extract_clinical_features(landmarks_sequence)

        # LSTM prediction
        normalized = (clinical_features - self.norm_params['mean']) / self.norm_params['std']
        X_lstm = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)

        self.lstm_model.eval()
        with torch.no_grad():
            lstm_pred = self.lstm_model(X_lstm).cpu().numpy()[0]

        # RF prediction
        stat_features = extract_statistical_features(clinical_features)
        rf_pred = self.rf_model.predict([stat_features])[0]

        # Ensemble
        ensemble_pred = self.lstm_weight * lstm_pred + self.rf_weight * rf_pred
        ensemble_pred = np.clip(ensemble_pred, 0, 4)

        # Confidence based on model agreement
        agreement = 1 - abs(lstm_pred - rf_pred) / 4

        return {
            'score': float(np.round(ensemble_pred)),
            'raw_score': float(ensemble_pred),
            'lstm_score': float(np.clip(lstm_pred, 0, 4)),
            'rf_score': float(np.clip(rf_pred, 0, 4)),
            'confidence': float(agreement),
            'model': 'lstm_rf_ensemble'
        }


def train_production_models():
    print("=" * 70)
    print("Training Production Models (LSTM + RF Ensemble)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load all data
    print("\nLoading cached data...")
    with open(f"{CACHE_DIR}/train_landmarks.pkl", 'rb') as f:
        X_train_raw, y_train = pickle.load(f)

    with open(f"{CACHE_DIR}/test_landmarks.pkl", 'rb') as f:
        X_test_raw, y_test = pickle.load(f)

    # Combine all data for production model
    X_all_raw = np.vstack([X_train_raw, X_test_raw])
    y_all = np.concatenate([y_train, y_test])

    print(f"Total samples: {len(y_all)}")
    print(f"Class distribution:")
    for score in range(5):
        count = (y_all == score).sum()
        print(f"  Score {score}: {count} ({count/len(y_all)*100:.1f}%)")

    # Extract clinical features
    print("\nExtracting clinical features...")
    X_clinical = np.array([extract_clinical_features(seq) for seq in X_all_raw])
    print(f"Clinical feature shape: {X_clinical.shape}")

    # Normalize
    mean = X_clinical.mean(axis=(0, 1), keepdims=True)
    std = X_clinical.std(axis=(0, 1), keepdims=True) + 1e-8
    X_normalized = (X_clinical - mean) / std

    norm_params = {
        'mean': mean.squeeze(),
        'std': std.squeeze()
    }

    # ========== Train LSTM ==========
    print("\n" + "=" * 70)
    print("Training LSTM Production Model")
    print("=" * 70)

    input_size = X_clinical.shape[2]
    model = AttentionLSTM(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)

    # Use 10% for validation during training
    val_size = int(len(X_normalized) * 0.1)
    indices = torch.randperm(len(X_normalized))
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    X_train_t = torch.FloatTensor(X_normalized[train_idx])
    y_train_t = torch.FloatTensor(y_all[train_idx])
    X_val_t = torch.FloatTensor(X_normalized[val_idx]).to(device)
    y_val_t = torch.FloatTensor(y_all[val_idx]).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(100):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100 - Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # Save LSTM model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'norm_params': norm_params
    }, f"{OUTPUT_DIR}/lstm_finger_tapping_production.pth")
    print(f"LSTM model saved to {OUTPUT_DIR}/lstm_finger_tapping_production.pth")

    # ========== Train RF ==========
    print("\n" + "=" * 70)
    print("Training Random Forest Model")
    print("=" * 70)

    # Extract statistical features for RF
    X_rf = np.array([extract_statistical_features(seq) for seq in X_clinical])
    print(f"RF feature shape: {X_rf.shape}")

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_rf, y_all)

    # Save RF model
    with open(f"{OUTPUT_DIR}/rf_finger_tapping_production.pkl", 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"RF model saved to {OUTPUT_DIR}/rf_finger_tapping_production.pkl")

    # ========== Create Ensemble ==========
    print("\n" + "=" * 70)
    print("Creating Ensemble Model")
    print("=" * 70)

    ensemble = EnsembleModel(model, rf_model, norm_params, device)

    # Test ensemble on all data
    print("\nEvaluating ensemble on all data...")
    lstm_preds = []
    rf_preds = []
    ensemble_preds = []

    model.eval()
    with torch.no_grad():
        X_all_t = torch.FloatTensor(X_normalized).to(device)
        lstm_preds = model(X_all_t).cpu().numpy()
        lstm_preds = np.clip(lstm_preds, 0, 4)

    rf_preds = rf_model.predict(X_rf)
    rf_preds = np.clip(rf_preds, 0, 4)

    ensemble_preds = 0.6 * lstm_preds + 0.4 * rf_preds
    ensemble_preds = np.clip(ensemble_preds, 0, 4)

    # Metrics
    print("\n" + "=" * 70)
    print("Production Model Performance (All Data)")
    print("=" * 70)

    for name, preds in [("LSTM", lstm_preds), ("RF", rf_preds), ("Ensemble", ensemble_preds)]:
        mae = np.mean(np.abs(y_all - preds))
        preds_rounded = np.round(preds)
        exact = np.mean(preds_rounded == y_all) * 100
        within1 = np.mean(np.abs(y_all - preds_rounded) <= 1) * 100
        pearson, _ = stats.pearsonr(y_all, preds)

        print(f"\n{name}:")
        print(f"  MAE: {mae:.3f}")
        print(f"  Exact Accuracy: {exact:.1f}%")
        print(f"  Within 1 Point: {within1:.1f}%")
        print(f"  Pearson r: {pearson:.3f}")

    # Save ensemble configuration
    ensemble_config = {
        'lstm_weight': 0.6,
        'rf_weight': 0.4,
        'norm_params': norm_params,
        'lstm_path': f"{OUTPUT_DIR}/lstm_finger_tapping_production.pth",
        'rf_path': f"{OUTPUT_DIR}/rf_finger_tapping_production.pkl",
        'cv_metrics': {
            'mae': 0.381,
            'exact_accuracy': 70.6,
            'within_1': 98.6,
            'pearson': 0.706
        }
    }

    with open(f"{OUTPUT_DIR}/ensemble_finger_tapping_config.pkl", 'wb') as f:
        pickle.dump(ensemble_config, f)
    print(f"\nEnsemble config saved to {OUTPUT_DIR}/ensemble_finger_tapping_config.pkl")

    print("\n" + "=" * 70)
    print("Production Training Complete!")
    print("=" * 70)
    print(f"\nModels saved to {OUTPUT_DIR}/:")
    print("  - lstm_finger_tapping_production.pth")
    print("  - rf_finger_tapping_production.pkl")
    print("  - ensemble_finger_tapping_config.pkl")

    return model, rf_model, norm_params


if __name__ == "__main__":
    train_production_models()
