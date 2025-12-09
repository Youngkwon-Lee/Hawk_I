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


@pytest.fixture
def mock_finger_tapping_landmarks():
    """Generate mock hand landmark data for finger tapping analysis"""
    frames = []
    for i in range(60):  # 2 seconds at 30fps
        # Simulate tapping motion with sine wave
        val = 0.1 + 0.08 * math.sin(i * 0.5)  # Oscillating distance

        frame = {
            "frame_number": i,
            "timestamp": i / 30.0,
            "keypoints": [
                {"id": 0, "x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0},  # Wrist
                {"id": 4, "x": 0.5, "y": 0.4, "z": 0.0, "visibility": 1.0},  # Thumb tip
                {"id": 5, "x": 0.5, "y": 0.45, "z": 0.0, "visibility": 1.0}, # Index MCP
                {"id": 6, "x": 0.5, "y": 0.42, "z": 0.0, "visibility": 1.0}, # Index PIP
                {"id": 7, "x": 0.5, "y": 0.38, "z": 0.0, "visibility": 1.0}, # Index DIP
                {"id": 8, "x": 0.5 + val, "y": 0.4 + val, "z": val, "visibility": 1.0},  # Index tip
            ]
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
