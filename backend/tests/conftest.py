"""
Pytest Configuration and Fixtures for Hawkeye Backend Tests
"""
import sys
import os
import pytest
import numpy as np
import math

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _IdentityScaler:
    def transform(self, values):
        return values


class _ConstantModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, values):
        return np.full(len(values), self.prediction, dtype=float)


@pytest.fixture(scope="session")
def test_ml_model_dir(tmp_path_factory):
    import joblib

    model_dir = tmp_path_factory.mktemp("hawkeye_ml_models")

    joblib.dump(_ConstantModel(2.0), model_dir / "rf_finger_tapping_scorer.pkl")
    joblib.dump(_ConstantModel(2.2), model_dir / "xgb_finger_tapping_scorer.pkl")
    joblib.dump(_IdentityScaler(), model_dir / "finger_tapping_scaler.pkl")

    joblib.dump(_ConstantModel(1.0), model_dir / "rf_gait_scorer.pkl")
    joblib.dump(_ConstantModel(1.1), model_dir / "xgb_gait_scorer.pkl")
    joblib.dump(_IdentityScaler(), model_dir / "gait_scaler.pkl")

    return model_dir


@pytest.fixture(autouse=True)
def isolated_ml_scorer(monkeypatch, test_ml_model_dir):
    monkeypatch.setenv("HAWKEYE_ML_MODEL_DIR", str(test_ml_model_dir))

    try:
        from services.ml_scorer import MLScorer
        MLScorer._instance = None
    except ImportError:
        pass

    yield

    try:
        from services.ml_scorer import MLScorer
        MLScorer._instance = None
    except ImportError:
        pass


@pytest.fixture
def mock_finger_tapping_landmarks():
    """Generate mock hand landmark data for finger tapping analysis"""
    frames = []
    for i in range(60):  # 2 seconds at 30fps
        phase = 2 * math.pi * i / 12
        gap = 0.025 + 0.075 * (0.5 + 0.5 * math.sin(phase))
        index_tip = {"id": 8, "x": 0.42 + gap, "y": 0.49, "z": 0.01 * math.sin(phase), "visibility": 1.0}

        landmarks = [
            {"id": 0, "x": 0.50, "y": 0.75, "z": 0.00, "visibility": 1.0},
            {"id": 1, "x": 0.46, "y": 0.67, "z": 0.00, "visibility": 1.0},
            {"id": 2, "x": 0.43, "y": 0.59, "z": 0.00, "visibility": 1.0},
            {"id": 3, "x": 0.42, "y": 0.53, "z": 0.00, "visibility": 1.0},
            {"id": 4, "x": 0.42, "y": 0.49, "z": 0.00, "visibility": 1.0},
            {"id": 5, "x": 0.52, "y": 0.56, "z": 0.00, "visibility": 1.0},
            {"id": 6, "x": 0.53, "y": 0.47, "z": 0.00, "visibility": 1.0},
            {"id": 7, "x": 0.535, "y": 0.40, "z": 0.00, "visibility": 1.0},
            index_tip,
            {"id": 9, "x": 0.57, "y": 0.58, "z": 0.00, "visibility": 1.0},
            {"id": 10, "x": 0.58, "y": 0.48, "z": 0.00, "visibility": 1.0},
            {"id": 11, "x": 0.585, "y": 0.40, "z": 0.00, "visibility": 1.0},
            {"id": 12, "x": 0.59, "y": 0.34, "z": 0.00, "visibility": 1.0},
            {"id": 13, "x": 0.61, "y": 0.60, "z": 0.00, "visibility": 1.0},
            {"id": 14, "x": 0.62, "y": 0.51, "z": 0.00, "visibility": 1.0},
            {"id": 15, "x": 0.625, "y": 0.44, "z": 0.00, "visibility": 1.0},
            {"id": 16, "x": 0.63, "y": 0.38, "z": 0.00, "visibility": 1.0},
            {"id": 17, "x": 0.65, "y": 0.63, "z": 0.00, "visibility": 1.0},
            {"id": 18, "x": 0.66, "y": 0.56, "z": 0.00, "visibility": 1.0},
            {"id": 19, "x": 0.665, "y": 0.50, "z": 0.00, "visibility": 1.0},
            {"id": 20, "x": 0.67, "y": 0.45, "z": 0.00, "visibility": 1.0},
        ]

        frame = {
            "frame_number": i,
            "timestamp": i / 30.0,
            "keypoints": landmarks,
            "landmarks": landmarks,
        }
        frames.append(frame)
    return frames


@pytest.fixture
def mock_gait_landmarks():
    """Generate mock pose landmark data for gait analysis"""
    frames = []
    for i in range(90):  # 3 seconds at 30fps
        # Simulate walking with lateral movement
        phase = i * 0.2
        left_offset = 0.05 * math.sin(phase)
        right_offset = 0.05 * math.sin(phase + math.pi)

        frame = {
            "frame_number": i,
            "timestamp": i / 30.0,
            "keypoints": [
                # Hips
                {"id": 23, "x": 0.45, "y": 0.5 + left_offset, "z": i * 0.01, "visibility": 1.0},
                {"id": 24, "x": 0.55, "y": 0.5 + right_offset, "z": i * 0.01, "visibility": 1.0},
                # Knees
                {"id": 25, "x": 0.45, "y": 0.7 + left_offset * 0.5, "z": i * 0.01, "visibility": 1.0},
                {"id": 26, "x": 0.55, "y": 0.7 + right_offset * 0.5, "z": i * 0.01, "visibility": 1.0},
                # Ankles
                {"id": 27, "x": 0.45 + left_offset, "y": 0.9, "z": i * 0.01, "visibility": 1.0},
                {"id": 28, "x": 0.55 + right_offset, "y": 0.9, "z": i * 0.01, "visibility": 1.0},
                # Wrists (for arm swing)
                {"id": 15, "x": 0.35 + left_offset * 2, "y": 0.6, "z": i * 0.01, "visibility": 1.0},
                {"id": 16, "x": 0.65 + right_offset * 2, "y": 0.6, "z": i * 0.01, "visibility": 1.0},
            ],
            "world_keypoints": [
                {"id": 23, "x": -0.1, "y": 0.0, "z": left_offset * 0.3, "visibility": 1.0},
                {"id": 24, "x": 0.1, "y": 0.0, "z": right_offset * 0.3, "visibility": 1.0},
                {"id": 25, "x": -0.1, "y": -0.4, "z": left_offset * 0.2, "visibility": 1.0},
                {"id": 26, "x": 0.1, "y": -0.4, "z": right_offset * 0.2, "visibility": 1.0},
                {"id": 27, "x": -0.1 + left_offset * 0.5, "y": -0.8, "z": left_offset * 0.1, "visibility": 1.0},
                {"id": 28, "x": 0.1 + right_offset * 0.5, "y": -0.8, "z": right_offset * 0.1, "visibility": 1.0},
                {"id": 15, "x": -0.2, "y": -0.2, "z": left_offset * 0.4, "visibility": 1.0},
                {"id": 16, "x": 0.2, "y": -0.2, "z": right_offset * 0.4, "visibility": 1.0},
            ]
        }
        frames.append(frame)
    return frames


@pytest.fixture
def mock_finger_tapping_metrics():
    """Mock metrics dict for ML scorer testing"""
    return {
        'tapping_speed': 2.5,
        'amplitude_mean': 0.15,
        'amplitude_std': 0.03,
        'amplitude_decrement': 10.0,
        'first_half_amplitude': 0.16,
        'second_half_amplitude': 0.14,
        'opening_velocity_mean': 1.2,
        'closing_velocity_mean': 1.1,
        'peak_velocity_mean': 1.5,
        'velocity_decrement': 8.0,
        'rhythm_variability': 15.0,
        'hesitation_count': 2,
        'halt_count': 1,
        'freeze_episodes': 0,
        'fatigue_rate': 5.0,
        'velocity_first_third': 1.6,
        'velocity_mid_third': 1.5,
        'velocity_last_third': 1.4,
        'amplitude_first_third': 0.16,
        'amplitude_mid_third': 0.15,
        'amplitude_last_third': 0.14,
        'velocity_slope': -0.5,
        'amplitude_slope': -0.3,
        'rhythm_slope': 0.2,
        'variability_first_half': 12.0,
        'variability_second_half': 18.0,
        'variability_change': 50.0
    }


@pytest.fixture
def analysis_context():
    """Create a basic AnalysisContext for testing"""
    from domain.context import AnalysisContext
    return AnalysisContext(video_path="test_video.mp4")
