"""
Gait LSTM with K-Fold Cross-Validation
Uses Pose landmarks (33 landmarks) for gait analysis
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from tqdm import tqdm

# Add scripts directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PD4T_ROOT, PD4T_ANNOTATIONS, PD4T_VIDEOS, TRAINED_MODELS_DIR, CACHE_DIR, ensure_dirs

# Paths (from centralized config)
ANNOTATION_DIR = str(PD4T_ANNOTATIONS["gait"])
VIDEO_DIR = str(PD4T_VIDEOS["gait"])
OUTPUT_DIR = str(TRAINED_MODELS_DIR)

ensure_dirs()

# Config
SEQUENCE_LENGTH = 90  # Gait needs longer sequences
NUM_POSE_LANDMARKS = 33


def load_annotations():
    """Load PD4T Gait annotations"""
    train_df = pd.read_csv(f"{ANNOTATION_DIR}/train.csv",
                           header=None, names=['video', 'frames', 'score'])
    test_df = pd.read_csv(f"{ANNOTATION_DIR}/test.csv",
                          header=None, names=['video', 'frames', 'score'])

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train score distribution:\n{train_df['score'].value_counts().sort_index()}")

    return train_df, test_df


def get_video_path(video_id: str) -> str:
    """Get video file path from annotation video ID"""
    # Format: "15-001760_009" -> find matching video
    parts = video_id.rsplit('_', 1)
    if len(parts) == 2:
        base_name, suffix = parts

        # Try different folder structures
        for folder in os.listdir(VIDEO_DIR):
            folder_path = f"{VIDEO_DIR}/{folder}"
            if os.path.isdir(folder_path):
                # Try direct match
                for ext in ['.mp4', '.avi', '.mov']:
                    video_path = f"{folder_path}/{video_id}{ext}"
                    if os.path.exists(video_path):
                        return video_path

                    # Try with base name
                    video_path = f"{folder_path}/{base_name}{ext}"
                    if os.path.exists(video_path):
                        return video_path

    return None


def extract_pose_landmarks(video_path: str, max_frames: int = 450):
    """Extract pose landmarks from video using MediaPipe"""
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
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
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            frame_landmarks = []
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            landmarks_list.append(frame_landmarks)
        else:
            # No pose detected
            landmarks_list.append([0.0] * (NUM_POSE_LANDMARKS * 4))

        frame_count += 1

    cap.release()
    pose.close()

    return np.array(landmarks_list)


def add_velocity_features(landmarks: np.ndarray) -> np.ndarray:
    """Add velocity features for pose landmarks"""
    if len(landmarks) < 2:
        return landmarks

    # Extract just x, y, z (ignore visibility for velocity)
    positions = []
    for i in range(NUM_POSE_LANDMARKS):
        positions.append(landmarks[:, i*4:i*4+3])

    positions = np.concatenate(positions, axis=1)  # (frames, 99)

    # Velocity
    velocity = np.diff(positions, axis=0)
    velocity = np.vstack([np.zeros((1, positions.shape[1])), velocity])

    # Key joint speeds (hip, knee, ankle)
    # Left hip: landmark 23, Right hip: 24
    # Left knee: 25, Right knee: 26
    # Left ankle: 27, Right ankle: 28
    hip_left_vel = velocity[:, 23*3:23*3+3]
    hip_right_vel = velocity[:, 24*3:24*3+3]
    knee_left_vel = velocity[:, 25*3:25*3+3]
    knee_right_vel = velocity[:, 26*3:26*3+3]
    ankle_left_vel = velocity[:, 27*3:27*3+3]
    ankle_right_vel = velocity[:, 28*3:28*3+3]

    hip_speed = np.linalg.norm(hip_left_vel, axis=1) + np.linalg.norm(hip_right_vel, axis=1)
    knee_speed = np.linalg.norm(knee_left_vel, axis=1) + np.linalg.norm(knee_right_vel, axis=1)
    ankle_speed = np.linalg.norm(ankle_left_vel, axis=1) + np.linalg.norm(ankle_right_vel, axis=1)

    # Combine
    features = np.hstack([
        landmarks,  # Original (with visibility)
        velocity,   # Velocity
        hip_speed.reshape(-1, 1),
        knee_speed.reshape(-1, 1),
        ankle_speed.reshape(-1, 1)
    ])

    return features


def extract_gait_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract clinically relevant gait features"""
    # Simplified feature extraction for time series
    num_frames = len(landmarks)

    # Key landmark indices (x, y, z per landmark, with 4 values each including visibility)
    # Hip center approximation
    hip_left_idx = 23 * 4
    hip_right_idx = 24 * 4
    knee_left_idx = 25 * 4
    knee_right_idx = 26 * 4
    ankle_left_idx = 27 * 4
    ankle_right_idx = 28 * 4
    shoulder_left_idx = 11 * 4
    shoulder_right_idx = 12 * 4

    # Extract positions (x, y only for 2D analysis)
    hip_left = landmarks[:, hip_left_idx:hip_left_idx+2]
    hip_right = landmarks[:, hip_right_idx:hip_right_idx+2]
    knee_left = landmarks[:, knee_left_idx:knee_left_idx+2]
    knee_right = landmarks[:, knee_right_idx:knee_right_idx+2]
    ankle_left = landmarks[:, ankle_left_idx:ankle_left_idx+2]
    ankle_right = landmarks[:, ankle_right_idx:ankle_right_idx+2]
    shoulder_left = landmarks[:, shoulder_left_idx:shoulder_left_idx+2]
    shoulder_right = landmarks[:, shoulder_right_idx:shoulder_right_idx+2]

    # Hip center
    hip_center = (hip_left + hip_right) / 2

    # Features
    # 1. Step width (lateral distance between ankles)
    step_width = np.abs(ankle_left[:, 0] - ankle_right[:, 0])

    # 2. Step height (vertical position of ankles)
    step_height_left = ankle_left[:, 1]
    step_height_right = ankle_right[:, 1]

    # 3. Knee flexion (angle approximation)
    hip_knee_left = knee_left - hip_left
    knee_ankle_left = ankle_left - knee_left
    knee_angle_left = np.arctan2(knee_ankle_left[:, 1], knee_ankle_left[:, 0]) - \
                      np.arctan2(hip_knee_left[:, 1], hip_knee_left[:, 0])

    hip_knee_right = knee_right - hip_right
    knee_ankle_right = ankle_right - knee_right
    knee_angle_right = np.arctan2(knee_ankle_right[:, 1], knee_ankle_right[:, 0]) - \
                       np.arctan2(hip_knee_right[:, 1], hip_knee_right[:, 0])

    # 4. Trunk angle (shoulder to hip)
    shoulder_center = (shoulder_left + shoulder_right) / 2
    trunk_vector = shoulder_center - hip_center
    trunk_angle = np.arctan2(trunk_vector[:, 0], trunk_vector[:, 1])

    # 5. Walking speed (hip center movement)
    hip_velocity = np.gradient(hip_center, axis=0)
    walking_speed = np.linalg.norm(hip_velocity, axis=1)

    # 6. Arm swing (shoulder movement)
    arm_left = np.linalg.norm(np.gradient(shoulder_left, axis=0), axis=1)
    arm_right = np.linalg.norm(np.gradient(shoulder_right, axis=0), axis=1)
    arm_swing = arm_left + arm_right

    # 7. Gait asymmetry
    left_ankle_movement = np.linalg.norm(np.gradient(ankle_left, axis=0), axis=1)
    right_ankle_movement = np.linalg.norm(np.gradient(ankle_right, axis=0), axis=1)
    asymmetry = np.abs(left_ankle_movement - right_ankle_movement) / (left_ankle_movement + right_ankle_movement + 0.001)

    # Stack features
    features = np.stack([
        step_width,
        step_height_left,
        step_height_right,
        knee_angle_left,
        knee_angle_right,
        trunk_angle,
        walking_speed,
        arm_swing,
        asymmetry,
        left_ankle_movement,
        right_ankle_movement,
        hip_center[:, 0],  # x position
        hip_center[:, 1],  # y position
    ], axis=1)

    return features


def create_sequences(features: np.ndarray, seq_length: int = SEQUENCE_LENGTH):
    """Create overlapping sequences"""
    sequences = []

    if len(features) < seq_length:
        padded = np.zeros((seq_length, features.shape[1]))
        padded[:len(features)] = features
        sequences.append(padded)
    else:
        step = seq_length // 2
        for i in range(0, len(features) - seq_length + 1, step):
            sequences.append(features[i:i + seq_length])

    return np.array(sequences)


def extract_all_landmarks(df: pd.DataFrame, cache_name: str):
    """Extract landmarks from all videos with caching"""
    cache_path = f"{CACHE_DIR}/{cache_name}_gait_landmarks.pkl"

    if os.path.exists(cache_path):
        print(f"Loading cached landmarks from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    all_sequences = []
    all_scores = []
    processed = 0
    skipped = 0

    print(f"Extracting landmarks from {len(df)} videos...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_id = row['video']
        score = row['score']

        video_path = get_video_path(video_id)
        if video_path is None:
            skipped += 1
            continue

        try:
            landmarks = extract_pose_landmarks(video_path)

            if len(landmarks) < 30:
                skipped += 1
                continue

            # Extract gait features
            features = extract_gait_features(landmarks)
            sequences = create_sequences(features)

            for seq in sequences:
                all_sequences.append(seq)
                all_scores.append(score)

            processed += 1

        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            skipped += 1
            continue

    print(f"Processed: {processed}, Skipped: {skipped}")

    X = np.array(all_sequences)
    y = np.array(all_scores)

    print(f"Caching {len(X)} sequences to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump((X, y), f)

    return X, y


class GaitAttentionLSTM(nn.Module):
    """LSTM with Attention for Gait Analysis"""
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
    print("Gait LSTM with Stratified K-Fold Cross-Validation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load annotations
    train_df, test_df = load_annotations()

    # Extract landmarks
    print("\nExtracting Training Landmarks...")
    X_train, y_train = extract_all_landmarks(train_df, "train")

    print("\nExtracting Test Landmarks...")
    X_test, y_test = extract_all_landmarks(test_df, "test")

    # Combine for CV
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    print(f"\nTotal samples: {len(y_all)}")
    print(f"Feature shape: {X_all.shape}")
    print(f"Class distribution:")
    for score in range(5):
        count = (y_all == score).sum()
        print(f"  Score {score}: {count} ({count/len(y_all)*100:.1f}%)")

    # Normalize
    mean = X_all.mean(axis=(0, 1), keepdims=True)
    std = X_all.std(axis=(0, 1), keepdims=True) + 1e-8
    X_normalized = (X_all - mean) / std

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

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_normalized, y_all)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train_fold = X_normalized[train_idx]
        y_train_fold = y_all[train_idx]
        X_val_fold = X_normalized[val_idx]
        y_val_fold = y_all[val_idx]

        X_train_t = torch.FloatTensor(X_train_fold)
        y_train_t = torch.FloatTensor(y_train_fold)
        X_val_t = torch.FloatTensor(X_val_fold).to(device)
        y_val_t = torch.FloatTensor(y_val_fold).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = GaitAttentionLSTM(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)
        model = train_fold(model, train_loader, X_val_t, y_val_t, device, epochs=80)

        results = evaluate_fold(model, X_val_fold, y_val_fold, device)
        fold_results.append(results)

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

    # Save production model
    print(f"\n{'=' * 70}")
    print("Training Production Model on All Data")
    print(f"{'=' * 70}")

    # Train final model
    X_train_t = torch.FloatTensor(X_normalized)
    y_train_t = torch.FloatTensor(y_all)

    # Split for validation
    val_size = int(len(X_normalized) * 0.1)
    indices = torch.randperm(len(X_normalized))

    train_dataset = TensorDataset(X_train_t[indices[val_size:]], y_train_t[indices[val_size:]])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    X_val_t = X_train_t[indices[:val_size]].to(device)
    y_val_t = y_train_t[indices[:val_size]].to(device)

    model = GaitAttentionLSTM(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)
    model = train_fold(model, train_loader, X_val_t, y_val_t, device, epochs=80)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'norm_params': {
            'mean': mean.squeeze(),
            'std': std.squeeze()
        },
        'cv_metrics': {
            'mae': np.mean(mae_scores),
            'exact_accuracy': np.mean(exact_scores),
            'within_1': np.mean(within1_scores),
            'pearson': np.mean(pearson_scores)
        }
    }, f"{OUTPUT_DIR}/lstm_gait_production.pth")

    print(f"\nModel saved to {OUTPUT_DIR}/lstm_gait_production.pth")


if __name__ == "__main__":
    main()
