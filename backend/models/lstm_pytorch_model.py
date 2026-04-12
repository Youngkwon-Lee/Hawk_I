"""
PyTorch LSTM Model for Finger Tapping UPDRS Scoring

Production model with RF+LSTM ensemble
Validated with 5-Fold Cross-Validation:
- MAE: 0.381 ± 0.024
- Exact Accuracy: 70.6% ± 2.5%
- Within 1 Point: 98.6% ± 0.4%
"""

import os
import numpy as np
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. LSTM model will use fallback.")

# Model paths (updated for new structure)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_backend_dir)
ML_MODEL_DIR = os.path.join(_project_root, "models", "trained")


@dataclass
class LSTMPrediction:
    """LSTM prediction result"""
    score: int
    confidence: float
    raw_score: float
    lstm_score: float
    rf_score: float
    model_type: str


class AttentionLSTM(nn.Module):
    """Bidirectional LSTM with Attention mechanism"""

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


class FingerTappingLSTMScorer:
    """
    PyTorch LSTM + RF Ensemble for Finger Tapping UPDRS Scoring

    Production model trained on PD4T dataset with 5-Fold CV validation:
    - MAE: 0.381, Exact: 70.6%, Within 1: 98.6%
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance._lstm_model = None
            cls._instance._rf_model = None
            cls._instance._norm_params = None
            # Auto-detect device
            cls._instance._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {cls._instance._device}")
        return cls._instance

    def load_models(self, model_dir: str = ML_MODEL_DIR) -> bool:
        """Load production LSTM and RF models"""
        if self._loaded:
            return True

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available")
            return False

        lstm_path = os.path.join(model_dir, "lstm_finger_tapping_production.pth")
        rf_path = os.path.join(model_dir, "rf_finger_tapping_production.pkl")

        # Load LSTM model
        if os.path.exists(lstm_path):
            try:
                # Load map_location based on device
                checkpoint = torch.load(lstm_path, map_location=self._device)

                self._lstm_model = AttentionLSTM(
                    input_size=checkpoint['input_size'],
                    hidden_size=checkpoint['hidden_size'],
                    num_layers=checkpoint['num_layers'],
                    dropout=checkpoint['dropout']
                )
                self._lstm_model.load_state_dict(checkpoint['model_state_dict'])
                self._lstm_model.to(self._device) # Move model to device
                self._lstm_model.eval()

                self._norm_params = checkpoint['norm_params']
                logger.info(f"Loaded LSTM model from {lstm_path} on {self._device}")

            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}")
                return False
        else:
            logger.warning(f"LSTM model not found at {lstm_path}")
            return False

        # Load RF model
        if os.path.exists(rf_path):
            try:
                with open(rf_path, 'rb') as f:
                    self._rf_model = pickle.load(f)
                logger.info(f"Loaded RF model from {rf_path}")
            except Exception as e:
                logger.error(f"Failed to load RF model: {e}")
                # RF is optional, LSTM alone can work

        self._loaded = True
        return True

    def is_loaded(self) -> bool:
        return self._loaded

    def _extract_clinical_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract clinically relevant time-series features from landmarks

        Args:
            landmarks: (frames, 129) - 63 positions + 63 velocities + 3 speeds
                      OR (frames, 63) - just positions

        Returns:
            (frames, 10) clinical features
        """
        # Handle different input formats
        if landmarks.shape[1] < 63:
            logger.warning(f"Unexpected landmark shape: {landmarks.shape}")
            return None

        # Extract positions
        thumb_pos = landmarks[:, 12:15]  # Thumb tip (landmark 4)
        index_pos = landmarks[:, 24:27]  # Index tip (landmark 8)
        wrist_pos = landmarks[:, 0:3]    # Wrist (landmark 0)

        # 1. Finger distance (aperture)
        finger_distance = np.linalg.norm(thumb_pos - index_pos, axis=1)

        # 2. Opening/closing velocity
        dist_velocity = np.gradient(finger_distance)

        # 3. Acceleration
        dist_accel = np.gradient(dist_velocity)

        # 4-5. Thumb and index speeds
        if landmarks.shape[1] >= 126:
            # Has velocity data
            thumb_vel = landmarks[:, 63+12:63+15]
            index_vel = landmarks[:, 63+24:63+27]
        else:
            # Calculate velocity from positions
            thumb_vel = np.gradient(thumb_pos, axis=0)
            index_vel = np.gradient(index_pos, axis=0)

        thumb_speed = np.linalg.norm(thumb_vel, axis=1)
        index_speed = np.linalg.norm(index_vel, axis=1)

        # 6. Combined finger speed
        combined_speed = thumb_speed + index_speed

        # 7-8. Distance from wrist
        thumb_from_wrist = np.linalg.norm(thumb_pos - wrist_pos, axis=1)
        index_from_wrist = np.linalg.norm(index_pos - wrist_pos, axis=1)

        # 9-10. Normalized features
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

    def _extract_statistical_features(self, clinical_features: np.ndarray) -> np.ndarray:
        """Extract statistical features for RF model"""
        from scipy import stats

        features = []

        for feat_idx in range(clinical_features.shape[1]):
            feat = clinical_features[:, feat_idx]
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

        # Temporal features
        finger_dist = clinical_features[:, 0]
        peaks = np.where((finger_dist[1:-1] > finger_dist[:-2]) &
                         (finger_dist[1:-1] > finger_dist[2:]))[0]
        features.append(len(peaks))

        if len(peaks) > 1:
            intervals = np.diff(peaks)
            features.extend([
                np.mean(intervals),
                np.std(intervals),
                np.std(intervals) / (np.mean(intervals) + 0.001),
            ])
        else:
            features.extend([0, 0, 0])

        return np.array(features)

    def _preprocess_landmarks_from_dict(self, landmark_frames: List[Dict]) -> np.ndarray:
        """
        Convert landmark dictionary format to numpy array

        Args:
            landmark_frames: List of frames with 'landmarks' or 'keypoints'

        Returns:
            (frames, 63) landmark positions
        """
        raw_data = []

        for frame in landmark_frames:
            keypoints = frame.get('landmarks', frame.get('keypoints', []))

            if len(keypoints) < 21:
                frame_features = np.zeros(63)
            else:
                frame_features = []
                for i in range(21):
                    kp = keypoints[i] if i < len(keypoints) else {'x': 0, 'y': 0, 'z': 0}
                    if isinstance(kp, dict):
                        frame_features.extend([
                            kp.get('x', 0),
                            kp.get('y', 0),
                            kp.get('z', 0)
                        ])
                    else:
                        frame_features.extend([0, 0, 0])
                frame_features = np.array(frame_features)

            raw_data.append(frame_features)

        return np.array(raw_data)

    def predict(self, landmarks: Any, sequence_length: int = 60) -> LSTMPrediction:
        """
        Predict UPDRS score from landmarks

        Args:
            landmarks: Either:
                - List[Dict]: Frame dictionaries with 'landmarks' or 'keypoints'
                - np.ndarray: (frames, features) array

        Returns:
            LSTMPrediction with score, confidence, and model details
        """
        if not self._loaded:
            self.load_models()

        if not self._loaded:
            # Fallback prediction
            return self._fallback_predict(landmarks)

        try:
            # Convert to numpy if needed
            if isinstance(landmarks, list):
                landmarks = self._preprocess_landmarks_from_dict(landmarks)

            if len(landmarks) < 10:
                return self._fallback_predict(landmarks)

            # Extract clinical features
            clinical_features = self._extract_clinical_features(landmarks)
            if clinical_features is None:
                return self._fallback_predict(landmarks)

            # Pad or truncate to sequence length
            if len(clinical_features) < sequence_length:
                padded = np.zeros((sequence_length, clinical_features.shape[1]))
                padded[:len(clinical_features)] = clinical_features
                clinical_features = padded
            elif len(clinical_features) > sequence_length:
                # Use sliding window and average
                step = sequence_length // 2
                predictions = []

                for i in range(0, len(clinical_features) - sequence_length + 1, step):
                    seq = clinical_features[i:i + sequence_length]
                    pred = self._predict_sequence(seq)
                    predictions.append(pred)

                # Average predictions
                lstm_pred = np.mean([p['lstm'] for p in predictions])
                rf_pred = np.mean([p['rf'] for p in predictions]) if predictions[0]['rf'] is not None else None

                return self._combine_predictions(lstm_pred, rf_pred)

            return self._predict_sequence(clinical_features)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_predict(landmarks)

    def _predict_sequence(self, clinical_features: np.ndarray) -> LSTMPrediction:
        """Predict from a single sequence"""
        # Normalize
        mean = self._norm_params['mean']
        std = self._norm_params['std']
        normalized = (clinical_features - mean) / std

        # LSTM prediction
        X_lstm = torch.FloatTensor(normalized).unsqueeze(0)

        with torch.no_grad():
            lstm_pred = self._lstm_model(X_lstm).numpy()[0]

        # RF prediction (if available)
        rf_pred = None
        if self._rf_model is not None:
            stat_features = self._extract_statistical_features(clinical_features)
            rf_pred = self._rf_model.predict([stat_features])[0]

        return self._combine_predictions(lstm_pred, rf_pred)

    def _combine_predictions(self, lstm_pred: float, rf_pred: Optional[float]) -> LSTMPrediction:
        """Combine LSTM and RF predictions"""
        lstm_pred = float(np.clip(lstm_pred, 0, 4))

        if rf_pred is not None:
            rf_pred = float(np.clip(rf_pred, 0, 4))
            # Ensemble: 60% LSTM + 40% RF
            ensemble_pred = 0.6 * lstm_pred + 0.4 * rf_pred
            agreement = 1 - abs(lstm_pred - rf_pred) / 4
        else:
            ensemble_pred = lstm_pred
            rf_pred = lstm_pred
            agreement = 0.7

        score = int(np.round(ensemble_pred))
        score = max(0, min(4, score))

        return LSTMPrediction(
            score=score,
            confidence=float(agreement),
            raw_score=float(ensemble_pred),
            lstm_score=float(lstm_pred),
            rf_score=float(rf_pred),
            model_type='lstm_rf_ensemble'
        )

    def _fallback_predict(self, landmarks: Any) -> LSTMPrediction:
        """Fallback prediction when model is not available"""
        return LSTMPrediction(
            score=2,
            confidence=0.5,
            raw_score=2.0,
            lstm_score=2.0,
            rf_score=2.0,
            model_type='fallback'
        )


def get_finger_tapping_lstm_scorer() -> FingerTappingLSTMScorer:
    """Get singleton instance of Finger Tapping LSTM scorer"""
    return FingerTappingLSTMScorer()


# For compatibility with existing code
class PyTorchLSTMFactory:
    """Factory for PyTorch LSTM scorers"""

    _scorers = {}

    @classmethod
    def get_scorer(cls, task_type: str):
        if task_type in ['finger_tapping', 'hand_movement']:
            if 'finger_tapping' not in cls._scorers:
                scorer = FingerTappingLSTMScorer()
                scorer.load_models()
                cls._scorers['finger_tapping'] = scorer
            return cls._scorers['finger_tapping']
        else:
            # Gait not yet implemented in PyTorch
            return None


if __name__ == "__main__":
    # Test the model
    print("Testing PyTorch LSTM Scorer...")

    scorer = get_finger_tapping_lstm_scorer()
    loaded = scorer.load_models()
    print(f"Model loaded: {loaded}")

    # Test with fake data
    fake_landmarks = []
    for i in range(60):
        keypoints = []
        for j in range(21):
            keypoints.append({
                'x': 0.5 + 0.1 * np.sin(i * 0.1 + j * 0.1),
                'y': 0.5 + 0.1 * np.cos(i * 0.1 + j * 0.1),
                'z': 0.0
            })
        fake_landmarks.append({'landmarks': keypoints})

    result = scorer.predict(fake_landmarks)
    print(f"Prediction: score={result.score}, confidence={result.confidence:.2f}")
    print(f"  LSTM: {result.lstm_score:.2f}, RF: {result.rf_score:.2f}")
    print(f"  Model: {result.model_type}")
