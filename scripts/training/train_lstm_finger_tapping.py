"""
Train LSTM Model for Finger Tapping UPDRS Scoring
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
from config import PD4T_ANNOTATIONS, PD4T_VIDEOS, TRAINED_MODELS_DIR, CACHE_DIR, BACKEND_DIR, ensure_dirs

sys.path.insert(0, str(BACKEND_DIR))

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
    # Format: "15-005087_l" -> "Videos/Finger tapping/l/15-005087.mp4"
    parts = video_id.rsplit('_', 1)
    if len(parts) == 2:
        base_name, side = parts
        video_path = f"{VIDEO_DIR}/{side}/{base_name}.mp4"
        if os.path.exists(video_path):
            return video_path

    # Try direct match
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

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            frame_landmarks = []
            for lm in hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(frame_landmarks)
        else:
            # No hand detected - use zeros
            landmarks_list.append([0.0] * (NUM_LANDMARKS * LANDMARK_DIM))

        frame_count += 1

    cap.release()
    hands.close()

    return np.array(landmarks_list)

def add_velocity_features(landmarks: np.ndarray) -> np.ndarray:
    """Add velocity and acceleration features"""
    if len(landmarks) < 2:
        return landmarks

    # Velocity: difference between consecutive frames
    velocity = np.diff(landmarks, axis=0)
    velocity = np.vstack([np.zeros((1, landmarks.shape[1])), velocity])

    # Speed: magnitude of velocity for key landmarks (thumb tip, index tip)
    thumb_vel = velocity[:, 12:15]  # Landmark 4 (thumb tip)
    index_vel = velocity[:, 24:27]  # Landmark 8 (index tip)
    thumb_speed = np.linalg.norm(thumb_vel, axis=1, keepdims=True)
    index_speed = np.linalg.norm(index_vel, axis=1, keepdims=True)

    # Finger distance (thumb-index)
    thumb_pos = landmarks[:, 12:15]
    index_pos = landmarks[:, 24:27]
    finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1, keepdims=True)

    # Combine
    features = np.hstack([landmarks, velocity, thumb_speed, index_speed, finger_distance])

    return features

def create_sequences(landmarks: np.ndarray, seq_length: int = SEQUENCE_LENGTH):
    """Create overlapping sequences from landmarks"""
    sequences = []

    if len(landmarks) < seq_length:
        # Pad if too short
        padded = np.zeros((seq_length, landmarks.shape[1]))
        padded[:len(landmarks)] = landmarks
        sequences.append(padded)
    else:
        # Sliding window with 50% overlap
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
            # Extract landmarks
            landmarks = extract_landmarks_from_video(video_path)

            if len(landmarks) < 10:
                continue

            # Add velocity features
            features = add_velocity_features(landmarks)

            # Create sequences
            sequences = create_sequences(features)

            for seq in sequences:
                all_sequences.append(seq)
                all_scores.append(score)

        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue

    X = np.array(all_sequences)
    y = np.array(all_scores)

    # Cache results
    print(f"Caching {len(X)} sequences to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump((X, y), f)

    return X, y

def build_lstm_model(input_shape):
    """Build LSTM model"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(16, activation='relu'),

        Dense(1, activation='linear')  # Regression output
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model

def main():
    print("="*60)
    print("LSTM Training for Finger Tapping UPDRS Scoring")
    print("="*60)

    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("TensorFlow not installed. Please install with: pip install tensorflow")
        return

    # Load annotations
    train_df, test_df = load_annotations()

    # Extract landmarks (this takes time - uses caching)
    print("\n" + "="*60)
    print("Extracting Training Landmarks")
    print("="*60)
    X_train, y_train = extract_all_landmarks(train_df, "train")

    print("\n" + "="*60)
    print("Extracting Test Landmarks")
    print("="*60)
    X_test, y_test = extract_all_landmarks(test_df, "test")

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Build model
    print("\n" + "="*60)
    print("Building LSTM Model")
    print("="*60)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    model.summary()

    # Train
    print("\n" + "="*60)
    print("Training LSTM Model")
    print("="*60)

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(
            f"{OUTPUT_DIR}/lstm_finger_tapping_model.keras",
            monitor='val_loss',
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "="*60)
    print("Evaluating LSTM Model")
    print("="*60)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred.flatten(), 0, 4)

    mae = np.mean(np.abs(y_test - y_pred))
    y_pred_rounded = np.round(y_pred)
    exact = np.mean(y_pred_rounded == y_test) * 100
    within1 = np.mean(np.abs(y_test - y_pred_rounded) <= 1) * 100

    print(f"LSTM Results:")
    print(f"  MAE: {mae:.3f}")
    print(f"  Exact Accuracy: {exact:.1f}%")
    print(f"  Within 1 Point: {within1:.1f}%")

    # Compare with RF
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    print(f"RF (from earlier):  MAE=0.489, Exact=54.8%, Within 1pt=96.3%")
    print(f"LSTM:               MAE={mae:.3f}, Exact={exact:.1f}%, Within 1pt={within1:.1f}%")

    # Save model
    model.save(f"{OUTPUT_DIR}/lstm_finger_tapping_model.keras")
    print(f"\nModel saved to {OUTPUT_DIR}/lstm_finger_tapping_model.keras")

if __name__ == "__main__":
    main()
