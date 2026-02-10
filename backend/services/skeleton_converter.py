"""
Skeleton Converter for CORAL Models

Converts MediaPipe landmarks to the format expected by CORAL models.
Each task requires specific keypoints and frame counts:
- Gait: 10 keypoints × 3 coords = 30 features, 300 frames
- Finger Tapping: 41 keypoints × 3 coords = 123 features, 150 frames
- Hand Movement: 21 keypoints × 3 coords = 63 features, 150 frames
- Leg Agility: 6 keypoints × 3 coords = 18 features, 150 frames
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.interpolate import interp1d

# Task-specific configurations
TASK_CONFIGS = {
    'gait': {
        'target_frames': 300,
        'expected_features': 30,
        'mode': 'pose',
        # Keypoints: hips (23,24), knees (25,26), ankles (27,28), shoulders (11,12), wrists (15,16)
        'keypoint_indices': [23, 24, 25, 26, 27, 28, 11, 12, 15, 16]
    },
    'finger_tapping': {
        'target_frames': 150,
        'expected_features': 123,
        'mode': 'hand',
        # All 21 hand landmarks + 20 derived features (velocity, etc.)
        # Actually uses all 41 features from training
        'keypoint_indices': None  # Use all
    },
    'hand_movement': {
        'target_frames': 150,
        'expected_features': 63,
        'mode': 'hand',
        # All 21 hand landmarks × 3 coords = 63
        'keypoint_indices': None  # Use all 21 hand landmarks
    },
    'leg_agility': {
        'target_frames': 150,
        'expected_features': 18,
        'mode': 'pose',
        # Hip, knee, ankle for both legs: 23,24,25,26,27,28
        'keypoint_indices': [23, 24, 25, 26, 27, 28]
    }
}


def convert_landmarks_to_array(
    landmark_frames: List[Dict],
    task_type: str
) -> Optional[np.ndarray]:
    """
    Convert MediaPipe landmark frames to numpy array for CORAL model

    Args:
        landmark_frames: List of landmark dicts from MediaPipe
            Each dict has: {frame_number, timestamp, landmarks: [{id, x, y, z, visibility}]}
        task_type: 'gait', 'finger_tapping', 'hand_movement', 'leg_agility'

    Returns:
        numpy array of shape (target_frames, n_features) or None if conversion fails
    """
    if task_type not in TASK_CONFIGS:
        return None

    config = TASK_CONFIGS[task_type]
    target_frames = config['target_frames']
    expected_features = config['expected_features']
    keypoint_indices = config['keypoint_indices']

    # Filter frames with valid landmarks
    valid_frames = []
    for frame in landmark_frames:
        landmarks = frame.get('landmarks', [])
        if landmarks and len(landmarks) > 0:
            valid_frames.append(landmarks)

    if len(valid_frames) < 10:
        return None

    # Extract coordinates
    sequences = []
    for landmarks in valid_frames:
        if keypoint_indices is None:
            # Use all landmarks
            coords = []
            for lm in landmarks:
                coords.extend([lm['x'], lm['y'], lm['z']])
        else:
            # Use specific keypoints
            coords = []
            for idx in keypoint_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    coords.extend([lm['x'], lm['y'], lm['z']])
                else:
                    coords.extend([0.0, 0.0, 0.0])
        sequences.append(coords)

    sequences = np.array(sequences, dtype=np.float32)

    # Handle feature count mismatch
    actual_features = sequences.shape[1]
    if actual_features != expected_features:
        if actual_features > expected_features:
            # Truncate to expected
            sequences = sequences[:, :expected_features]
        else:
            # Pad with zeros (rare case)
            padding = np.zeros((sequences.shape[0], expected_features - actual_features), dtype=np.float32)
            sequences = np.concatenate([sequences, padding], axis=1)

    # Resample to target frames
    if len(sequences) != target_frames:
        sequences = resample_sequence(sequences, target_frames)

    return sequences


def resample_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Resample sequence to target number of frames using linear interpolation

    Args:
        sequence: numpy array of shape (T, F)
        target_frames: target number of frames

    Returns:
        resampled array of shape (target_frames, F)
    """
    T, F = sequence.shape

    if T == target_frames:
        return sequence

    old_indices = np.linspace(0, 1, T)
    new_indices = np.linspace(0, 1, target_frames)

    resampled = np.zeros((target_frames, F), dtype=np.float32)
    for f in range(F):
        interp_func = interp1d(old_indices, sequence[:, f], kind='linear', fill_value='extrapolate')
        resampled[:, f] = interp_func(new_indices)

    return resampled


def get_task_type_for_coral(task_type: str) -> str:
    """
    Map frontend task types to CORAL model task types

    Args:
        task_type: task type from frontend/classification

    Returns:
        CORAL model task type
    """
    # Normalize task type
    task_type = task_type.lower().replace(' ', '_').replace('-', '_')

    # Map to CORAL task types
    task_mapping = {
        'gait': 'gait',
        'walking': 'gait',
        'finger_tapping': 'finger_tapping',
        'finger': 'finger_tapping',
        'tapping': 'finger_tapping',
        'hand_movement': 'hand_movement',
        'hand': 'hand_movement',
        'hand_movements': 'hand_movement',
        'pronation_supination': 'hand_movement',
        'leg_agility': 'leg_agility',
        'leg': 'leg_agility',
    }

    return task_mapping.get(task_type, task_type)
