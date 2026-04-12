"""
Hawkeye Project - Central Configuration
All paths are relative to project root (Hawkeye/)
"""
import os
from pathlib import Path

# ============================================
# Project Root Detection
# ============================================
def get_project_root() -> Path:
    """Get project root directory (Hawkeye/)"""
    # Start from this file's location and go up to find project root
    current = Path(__file__).resolve()

    # Go up until we find the project root markers
    for parent in [current] + list(current.parents):
        if (parent / "backend").exists() and (parent / "frontend").exists():
            return parent
        if (parent / "CLAUDE.md").exists() and (parent / "README.md").exists():
            return parent

    # Fallback: assume scripts/config.py -> go up 2 levels
    return current.parent.parent


PROJECT_ROOT = get_project_root()

# ============================================
# Data Paths
# ============================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Dataset paths
PD4T_ROOT = RAW_DATA_DIR / "PD4T" / "PD4T" / "PD4T"
TULIP_ROOT = RAW_DATA_DIR / "TULIP"
VIDEO_CLIPS_DIR = RAW_DATA_DIR / "video_clips"

# PD4T specific paths
PD4T_ANNOTATIONS = {
    "gait": PD4T_ROOT / "Annotations" / "Gait",
    "finger_tapping": PD4T_ROOT / "Annotations" / "Finger tapping",
    "hand_movement": PD4T_ROOT / "Annotations" / "Hand movement",
    "leg_agility": PD4T_ROOT / "Annotations" / "Leg agility",
}

PD4T_VIDEOS = {
    "gait": PD4T_ROOT / "Videos" / "Gait",
    "finger_tapping": PD4T_ROOT / "Videos" / "Finger tapping",
    "hand_movement": PD4T_ROOT / "Videos" / "Hand movement",
    "leg_agility": PD4T_ROOT / "Videos" / "Leg agility",
}

# Processed data
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
CACHE_DIR = PROCESSED_DATA_DIR / "cache"

# ============================================
# Model Paths
# ============================================
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# ============================================
# Experiment Paths
# ============================================
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# ============================================
# Other Paths
# ============================================
DEMO_VIDEOS_DIR = PROJECT_ROOT / "demo_videos"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# ============================================
# Backend Paths
# ============================================
BACKEND_DIR = PROJECT_ROOT / "backend"
BACKEND_UPLOADS_DIR = BACKEND_DIR / "uploads"

# ============================================
# Ensure directories exist
# ============================================
def ensure_dirs():
    """Create all necessary directories"""
    dirs_to_create = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        FEATURES_DIR, CACHE_DIR,
        TRAINED_MODELS_DIR, CHECKPOINTS_DIR,
        CONFIGS_DIR, RESULTS_DIR,
        NOTEBOOKS_DIR, UPLOADS_DIR
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)


# ============================================
# Legacy Path Compatibility (for migration)
# ============================================
# Old paths -> New paths mapping
LEGACY_PATHS = {
    "C:/Users/YK/tulip/PD4T/PD4T/PD4T": str(PD4T_ROOT),
    "C:/Users/YK/tulip/Hawkeye/ml_models": str(TRAINED_MODELS_DIR),
    "C:/Users/YK/tulip/Hawkeye/ml_features": str(FEATURES_DIR),
    "C:/Users/YK/tulip/Hawkeye/lstm_cache": str(CACHE_DIR),
}

# ============================================
# Utility Functions
# ============================================
def get_annotation_path(task: str, split: str = "train") -> Path:
    """Get annotation CSV path for a task

    Args:
        task: 'gait', 'finger_tapping', 'hand_movement', 'leg_agility'
        split: 'train' or 'test'

    Returns:
        Path to the annotation CSV file
    """
    task_key = task.lower().replace(" ", "_")
    if task_key not in PD4T_ANNOTATIONS:
        raise ValueError(f"Unknown task: {task}. Available: {list(PD4T_ANNOTATIONS.keys())}")
    return PD4T_ANNOTATIONS[task_key] / f"{split}.csv"


def get_video_dir(task: str) -> Path:
    """Get video directory for a task"""
    task_key = task.lower().replace(" ", "_")
    if task_key not in PD4T_VIDEOS:
        raise ValueError(f"Unknown task: {task}. Available: {list(PD4T_VIDEOS.keys())}")
    return PD4T_VIDEOS[task_key]


def get_model_path(model_name: str) -> Path:
    """Get path for a trained model file"""
    return TRAINED_MODELS_DIR / model_name


def get_feature_path(feature_name: str) -> Path:
    """Get path for a feature file"""
    return FEATURES_DIR / feature_name


# ============================================
# Print Configuration (for debugging)
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("Hawkeye Project Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"PD4T Root: {PD4T_ROOT}")
    print(f"  - Exists: {PD4T_ROOT.exists()}")
    print(f"Models Dir: {TRAINED_MODELS_DIR}")
    print(f"Features Dir: {FEATURES_DIR}")
    print(f"Cache Dir: {CACHE_DIR}")
    print("=" * 50)
