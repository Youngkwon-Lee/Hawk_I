"""
LSTM Model for UPDRS Scoring

Time-series deep learning model for Parkinson's disease severity assessment.
Uses landmark trajectories as input sequences.

Architecture:
- Input: (batch_size, sequence_length, num_features)
- LSTM layers with dropout
- Dense layers for regression
- Output: UPDRS score (0-4)

Features per frame:
- Hand: 21 landmarks * 3 (x, y, z) = 63 features
- Pose: 33 landmarks * 3 (x, y, z) = 99 features
- Velocity and acceleration (derived)
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Check if TensorFlow/PyTorch is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization,
        Input, Bidirectional, Attention, Concatenate
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. LSTM model will use fallback.")


@dataclass
class LSTMConfig:
    """LSTM Model Configuration"""
    sequence_length: int = 60  # Number of frames per sequence
    lstm_units: List[int] = None  # LSTM layer sizes
    dense_units: List[int] = None  # Dense layer sizes
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10  # Early stopping patience

    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [128, 64]
        if self.dense_units is None:
            self.dense_units = [32, 16]


class LSTMScorer:
    """LSTM-based UPDRS Scorer for time-series analysis"""

    def __init__(self, task_type: str = 'finger_tapping', config: Optional[LSTMConfig] = None):
        self.task_type = task_type
        self.config = config or LSTMConfig()
        self.model = None
        self.scaler = None

        # Feature configuration based on task type
        if task_type in ['finger_tapping', 'hand_movement']:
            self.num_landmarks = 21  # Hand landmarks
            self.landmark_dim = 3  # x, y, z
        else:  # gait, leg_agility
            self.num_landmarks = 33  # Pose landmarks
            self.landmark_dim = 3

        # Additional derived features
        self.derived_features = 6  # velocity_x, velocity_y, velocity_z, speed, acc_x, acc_y

        self.num_features = (self.num_landmarks * self.landmark_dim) + self.derived_features

        # Model path
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', 'ml_models')
        self.model_path = os.path.join(self.model_dir, f'lstm_{task_type}_model.h5')

    def build_model(self) -> Optional['Model']:
        """Build LSTM model architecture"""
        if not TF_AVAILABLE:
            print("TensorFlow not available. Cannot build LSTM model.")
            return None

        inputs = Input(shape=(self.config.sequence_length, self.num_features))

        # Bidirectional LSTM layers
        x = inputs
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = (i < len(self.config.lstm_units) - 1)
            x = Bidirectional(
                LSTM(units, return_sequences=return_sequences)
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.config.dropout_rate)(x)

        # Dense layers
        for units in self.config.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.config.dropout_rate)(x)

        # Output layer (regression for UPDRS 0-4)
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def preprocess_landmarks(self, landmark_frames: List[Dict]) -> np.ndarray:
        """
        Preprocess landmark frames into sequences

        Args:
            landmark_frames: List of frame dictionaries with landmarks

        Returns:
            Numpy array of shape (num_sequences, sequence_length, num_features)
        """
        # Extract raw landmark coordinates
        raw_data = []

        for frame in landmark_frames:
            keypoints = frame.get('landmarks', frame.get('keypoints', []))

            if len(keypoints) < self.num_landmarks:
                # Pad with zeros if insufficient landmarks
                frame_features = np.zeros(self.num_landmarks * self.landmark_dim)
            else:
                frame_features = []
                for i in range(self.num_landmarks):
                    kp = keypoints[i] if i < len(keypoints) else {'x': 0, 'y': 0, 'z': 0}
                    frame_features.extend([
                        kp.get('x', 0),
                        kp.get('y', 0),
                        kp.get('z', 0)
                    ])
                frame_features = np.array(frame_features)

            raw_data.append(frame_features)

        raw_data = np.array(raw_data)

        # Calculate derived features (velocity, acceleration)
        if len(raw_data) > 1:
            # Velocity: difference between consecutive frames
            velocity = np.diff(raw_data, axis=0)
            velocity = np.vstack([np.zeros((1, raw_data.shape[1])), velocity])

            # Speed: magnitude of velocity (using first 3 coords as representative)
            speed = np.linalg.norm(velocity[:, :3], axis=1, keepdims=True)

            # Acceleration: difference of velocity
            acceleration = np.diff(velocity, axis=0)
            acceleration = np.vstack([np.zeros((1, velocity.shape[1])), acceleration])

            # Combine features
            # Use velocity of first landmark (x, y, z), speed, and acceleration (x, y)
            derived = np.hstack([
                velocity[:, :3],  # velocity x, y, z
                speed,            # speed
                acceleration[:, :2]  # acceleration x, y
            ])

            features = np.hstack([raw_data, derived])
        else:
            # Single frame - pad with zeros
            features = np.hstack([raw_data, np.zeros((len(raw_data), self.derived_features))])

        # Create sequences using sliding window
        sequences = []
        seq_len = self.config.sequence_length

        if len(features) >= seq_len:
            # Slide window to create multiple sequences
            step = seq_len // 2  # 50% overlap
            for i in range(0, len(features) - seq_len + 1, step):
                sequences.append(features[i:i + seq_len])
        else:
            # Pad to sequence length
            padded = np.zeros((seq_len, features.shape[1]))
            padded[:len(features)] = features
            sequences.append(padded)

        return np.array(sequences)

    def predict(self, landmark_frames: List[Dict]) -> Tuple[float, float]:
        """
        Predict UPDRS score from landmark sequence

        Args:
            landmark_frames: List of frame dictionaries

        Returns:
            Tuple of (score, confidence)
        """
        if not TF_AVAILABLE or self.model is None:
            # Fallback: use simple statistics-based prediction
            return self._fallback_predict(landmark_frames)

        try:
            # Preprocess
            sequences = self.preprocess_landmarks(landmark_frames)

            # Predict for each sequence
            predictions = self.model.predict(sequences, verbose=0)

            # Aggregate predictions (mean)
            mean_score = float(np.mean(predictions))

            # Clip to valid range
            mean_score = np.clip(mean_score, 0, 4)

            # Confidence based on prediction variance
            if len(predictions) > 1:
                variance = float(np.var(predictions))
                confidence = max(0.5, 1.0 - variance / 4.0)  # Higher variance = lower confidence
            else:
                confidence = 0.7

            return mean_score, confidence

        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return self._fallback_predict(landmark_frames)

    def _fallback_predict(self, landmark_frames: List[Dict]) -> Tuple[float, float]:
        """
        Fallback prediction using simple statistics

        Used when LSTM model is not available or fails
        """
        if len(landmark_frames) < 10:
            return 2.0, 0.5  # Default moderate score

        # Extract movement statistics
        try:
            # Get landmark positions
            positions = []
            for frame in landmark_frames:
                keypoints = frame.get('landmarks', frame.get('keypoints', []))
                if keypoints:
                    # Use first landmark as reference
                    kp = keypoints[0] if isinstance(keypoints[0], dict) else {'x': 0, 'y': 0}
                    positions.append([kp.get('x', 0), kp.get('y', 0)])

            positions = np.array(positions)

            if len(positions) < 5:
                return 2.0, 0.5

            # Calculate movement metrics
            velocities = np.diff(positions, axis=0)
            speeds = np.linalg.norm(velocities, axis=1)

            mean_speed = np.mean(speeds)
            speed_cv = np.std(speeds) / (mean_speed + 1e-6)

            # Heuristic scoring
            # Higher CV and lower speed = higher UPDRS
            if mean_speed < 0.01 and speed_cv > 0.5:
                score = 3.0
            elif mean_speed < 0.02 and speed_cv > 0.3:
                score = 2.0
            elif mean_speed < 0.03:
                score = 1.0
            else:
                score = 0.5

            return score, 0.6

        except Exception:
            return 2.0, 0.5

    def load_model(self) -> bool:
        """Load pre-trained LSTM model"""
        if not TF_AVAILABLE:
            return False

        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"Loaded LSTM model from {self.model_path}")
                return True
            except Exception as e:
                print(f"Failed to load LSTM model: {e}")
                return False
        return False

    def save_model(self) -> bool:
        """Save trained LSTM model"""
        if not TF_AVAILABLE or self.model is None:
            return False

        try:
            os.makedirs(self.model_dir, exist_ok=True)
            self.model.save(self.model_path)
            print(f"Saved LSTM model to {self.model_path}")
            return True
        except Exception as e:
            print(f"Failed to save LSTM model: {e}")
            return False

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2) -> Dict:
        """
        Train LSTM model

        Args:
            X: Training sequences (num_samples, sequence_length, num_features)
            y: Target UPDRS scores (num_samples,)
            validation_split: Fraction for validation

        Returns:
            Training history dictionary
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for training")

        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True
            )
        ]

        history = self.model.fit(
            X, y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'mae': history.history['mae'],
            'val_mae': history.history['val_mae']
        }


class LSTMScorerFactory:
    """Factory for creating task-specific LSTM scorers"""

    _instances: Dict[str, LSTMScorer] = {}

    @classmethod
    def get_scorer(cls, task_type: str) -> LSTMScorer:
        """Get or create LSTM scorer for task type"""
        if task_type not in cls._instances:
            scorer = LSTMScorer(task_type)
            # Try to load pre-trained model
            scorer.load_model()
            cls._instances[task_type] = scorer
        return cls._instances[task_type]


def create_training_data_from_videos(
    video_metadata: List[Dict],
    landmark_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset from labeled videos

    Args:
        video_metadata: List of {'video_id': str, 'updrs_score': float, 'task_type': str}
        landmark_dir: Directory containing landmark JSON files

    Returns:
        Tuple of (X, y) for training
    """
    X_sequences = []
    y_scores = []

    for meta in video_metadata:
        video_id = meta['video_id']
        score = meta['updrs_score']
        task_type = meta.get('task_type', 'finger_tapping')

        landmark_path = os.path.join(landmark_dir, f"{video_id}_landmarks.json")

        if not os.path.exists(landmark_path):
            continue

        with open(landmark_path, 'r') as f:
            landmarks = json.load(f)

        scorer = LSTMScorer(task_type)
        sequences = scorer.preprocess_landmarks(landmarks)

        for seq in sequences:
            X_sequences.append(seq)
            y_scores.append(score)

    return np.array(X_sequences), np.array(y_scores)


if __name__ == "__main__":
    # Test LSTM model
    print("Testing LSTM Model...")

    # Create dummy data
    config = LSTMConfig(sequence_length=30)
    scorer = LSTMScorer('finger_tapping', config)

    # Create fake landmarks
    fake_landmarks = []
    for i in range(60):
        keypoints = []
        for j in range(21):
            keypoints.append({
                'id': j,
                'x': 0.5 + 0.1 * np.sin(i * 0.1 + j * 0.1),
                'y': 0.5 + 0.1 * np.cos(i * 0.1 + j * 0.1),
                'z': 0.0
            })
        fake_landmarks.append({'landmarks': keypoints})

    # Test preprocessing
    sequences = scorer.preprocess_landmarks(fake_landmarks)
    print(f"Preprocessed shape: {sequences.shape}")

    # Test fallback prediction
    score, confidence = scorer.predict(fake_landmarks)
    print(f"Prediction: score={score:.2f}, confidence={confidence:.2f}")

    # Build model (if TF available)
    if TF_AVAILABLE:
        model = scorer.build_model()
        if model:
            print(f"Model built: {model.summary()}")
    else:
        print("TensorFlow not available - using fallback mode")
