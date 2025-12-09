"""
Train LSTM Model for Finger Tapping UPDRS Scoring (PyTorch Version)
Uses time-series hand landmarks from PD4T videos
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PD4T_ANNOTATIONS, PD4T_VIDEOS, TRAINED_MODELS_DIR, CACHE_DIR, ensure_dirs

# Paths (from centralized config)
ANNOTATION_DIR = str(PD4T_ANNOTATIONS["finger_tapping"])
VIDEO_DIR = str(PD4T_VIDEOS["finger_tapping"])
OUTPUT_DIR = str(TRAINED_MODELS_DIR)

ensure_dirs()

# LSTM Config
SEQUENCE_LENGTH = 60  # frames per sequence
NUM_LANDMARKS = 21  # Hand landmarks
LANDMARK_DIM = 3  # x, y, z

def load_annotations():
    """Load PD4T annotations"""
    train_df = pd.read_csv(f"{ANNOTATION_DIR}/train.csv",
                           header=None, names=['video', 'frames', 'score'])
    test_df = pd.read_csv(f"{ANNOTATION_DIR}/test.csv",
                          header=None, names=['video', 'frames', 'score'])

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train score distribution:\n{train_df['score'].value_counts().sort_index()}")

    return train_df, test_df

def get_video_path(video_id: str) -> str:
    """Get video file path from annotation video ID"""
    parts = video_id.rsplit('_', 1)
    if len(parts) == 2:
        base_name, side = parts
        video_path = f"{VIDEO_DIR}/{side}/{base_name}.mp4"
        if os.path.exists(video_path):
            return video_path

    for side in ['l', 'r']:
        video_path = f"{VIDEO_DIR}/{side}/{video_id}.mp4"
        if os.path.exists(video_path):
            return video_path

    return None

def extract_landmarks_from_video(video_path: str, max_frames: int = 300):
    """Extract hand landmarks from video using MediaPipe"""
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            frame_landmarks = []
            for lm in hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(frame_landmarks)
        else:
            landmarks_list.append([0.0] * (NUM_LANDMARKS * LANDMARK_DIM))

        frame_count += 1

    cap.release()
    hands.close()

    return np.array(landmarks_list)

def add_velocity_features(landmarks: np.ndarray) -> np.ndarray:
    """Add velocity and acceleration features"""
    if len(landmarks) < 2:
        return landmarks

    velocity = np.diff(landmarks, axis=0)
    velocity = np.vstack([np.zeros((1, landmarks.shape[1])), velocity])

    thumb_vel = velocity[:, 12:15]
    index_vel = velocity[:, 24:27]
    thumb_speed = np.linalg.norm(thumb_vel, axis=1, keepdims=True)
    index_speed = np.linalg.norm(index_vel, axis=1, keepdims=True)

    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]
    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1, keepdims=True)

    features = np.hstack([landmarks, velocity, thumb_speed, index_speed, finger_distance])

    return features

def create_sequences(landmarks: np.ndarray, seq_length: int = SEQUENCE_LENGTH):
    """Create overlapping sequences from landmarks"""
    sequences = []

    if len(landmarks) < seq_length:
        padded = np.zeros((seq_length, landmarks.shape[1]))
        padded[:len(landmarks)] = landmarks
        sequences.append(padded)
    else:
        step = seq_length // 2
        for i in range(0, len(landmarks) - seq_length + 1, step):
            sequences.append(landmarks[i:i + seq_length])

    return np.array(sequences)

def extract_all_landmarks(df: pd.DataFrame, cache_name: str):
    """Extract landmarks from all videos with caching"""
    cache_path = f"{CACHE_DIR}/{cache_name}_landmarks.pkl"

    if os.path.exists(cache_path):
        print(f"Loading cached landmarks from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    all_sequences = []
    all_scores = []

    print(f"Extracting landmarks from {len(df)} videos...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['video']
        score = row['score']

        video_path = get_video_path(video_id)
        if video_path is None:
            continue

        try:
            landmarks = extract_landmarks_from_video(video_path)

            if len(landmarks) < 10:
                continue

            features = add_velocity_features(landmarks)
            sequences = create_sequences(features)

            for seq in sequences:
                all_sequences.append(seq)
                all_scores.append(score)

        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue

    X = np.array(all_sequences)
    y = np.array(all_scores)

    print(f"Caching {len(X)} sequences to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump((X, y), f)

    return X, y

def main():
    print("=" * 60)
    print("LSTM Training for Finger Tapping UPDRS Scoring (PyTorch)")
    print("=" * 60)

    # Check PyTorch
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        print(f"PyTorch version: {torch.__version__}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
    except ImportError:
        print("PyTorch not installed. Please install with: pip install torch")
        return

    # Load annotations
    train_df, test_df = load_annotations()

    # Extract landmarks
    print("\n" + "=" * 60)
    print("Extracting Training Landmarks")
    print("=" * 60)
    X_train, y_train = extract_all_landmarks(train_df, "train")

    print("\n" + "=" * 60)
    print("Extracting Test Landmarks")
    print("=" * 60)
    X_test, y_test = extract_all_landmarks(test_df, "test")

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Build model
    print("\n" + "=" * 60)
    print("Building PyTorch LSTM Model")
    print("=" * 60)

    input_size = X_train.shape[2]  # Feature dimension

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
            )
            self.bn1 = nn.BatchNorm1d(hidden_size * 2)
            self.fc1 = nn.Linear(hidden_size * 2, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            # Take the last time step
            out = lstm_out[:, -1, :]
            out = self.bn1(out)
            out = self.relu(self.fc1(out))
            out = self.bn2(out)
            out = self.dropout(out)
            out = self.relu(self.fc2(out))
            out = self.fc3(out)
            return out.squeeze()

    model = LSTMModel(input_size).to(device)
    print(model)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Train
    print("\n" + "=" * 60)
    print("Training LSTM Model")
    print("=" * 60)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Split for validation
    val_size = int(len(X_train_t) * 0.15)
    indices = torch.randperm(len(X_train_t))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    X_train_split = X_train_t[train_indices]
    y_train_split = y_train_t[train_indices]
    X_val = X_train_t[val_indices]
    y_val = y_train_t[val_indices]

    train_dataset = TensorDataset(X_train_split, y_train_split)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(100):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
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
            # Save best model
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/lstm_finger_tapping_pytorch.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/lstm_finger_tapping_pytorch.pth"))

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating LSTM Model")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()

    y_pred = np.clip(y_pred, 0, 4)
    y_test_np = y_test_t.cpu().numpy()

    mae = np.mean(np.abs(y_test_np - y_pred))
    y_pred_rounded = np.round(y_pred)
    exact = np.mean(y_pred_rounded == y_test_np) * 100
    within1 = np.mean(np.abs(y_test_np - y_pred_rounded) <= 1) * 100

    print(f"PyTorch LSTM Results:")
    print(f"  MAE: {mae:.3f}")
    print(f"  Exact Accuracy: {exact:.1f}%")
    print(f"  Within 1 Point: {within1:.1f}%")

    # Compare with RF
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"RF (from earlier):    MAE=0.489, Exact=54.8%, Within 1pt=96.3%")
    print(f"PyTorch LSTM:         MAE={mae:.3f}, Exact={exact:.1f}%, Within 1pt={within1:.1f}%")

    print(f"\nModel saved to {OUTPUT_DIR}/lstm_finger_tapping_pytorch.pth")

if __name__ == "__main__":
    main()
