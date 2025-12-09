"""
Hawkeye Project - Environment-Aware Configuration
Supports: local, hpc, cloud environments
"""
import os
from pathlib import Path
from enum import Enum
from typing import Optional


class Environment(Enum):
    LOCAL = "local"      # Windows/Mac 로컬 개발
    HPC = "hpc"          # HPC Innovation Hub (GPU)
    CLOUD = "cloud"      # Cloud GPU (AWS, GCP, etc.)


def detect_environment() -> Environment:
    """Detect current execution environment"""
    # 환경변수로 명시적 지정
    env = os.getenv("HAWKEYE_ENV", "").lower()
    if env:
        return Environment(env)

    # 자동 감지
    if os.path.exists("/home") and os.path.exists("/scratch"):
        return Environment.HPC
    elif os.getenv("AWS_EXECUTION_ENV") or os.getenv("GOOGLE_CLOUD_PROJECT"):
        return Environment.CLOUD
    else:
        return Environment.LOCAL


# ============================================
# Environment Detection
# ============================================
CURRENT_ENV = detect_environment()


# ============================================
# Base Paths by Environment
# ============================================
def get_base_paths():
    """Get base paths for current environment"""

    if CURRENT_ENV == Environment.LOCAL:
        # Windows/Mac 로컬 환경
        project_root = Path(__file__).resolve().parent.parent
        data_root = project_root / "data"
        model_root = project_root / "models"

    elif CURRENT_ENV == Environment.HPC:
        # HPC 환경 - 환경변수 또는 기본 경로
        home = Path(os.getenv("HOME", "/home/user"))
        project_root = home / "hawkeye"
        # HPC는 보통 scratch 디렉토리에 대용량 데이터
        data_root = Path(os.getenv("SCRATCH", home)) / "hawkeye_data"
        model_root = project_root / "models"

    elif CURRENT_ENV == Environment.CLOUD:
        # Cloud 환경
        project_root = Path("/workspace/hawkeye")
        data_root = Path("/data/hawkeye")
        model_root = project_root / "models"

    return {
        "project_root": project_root,
        "data_root": data_root,
        "model_root": model_root,
    }


# ============================================
# Path Configuration
# ============================================
_paths = get_base_paths()
PROJECT_ROOT = _paths["project_root"]
DATA_ROOT = _paths["data_root"]
MODEL_ROOT = _paths["model_root"]

# Data paths
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
CACHE_DIR = PROCESSED_DATA_DIR / "cache"

# Dataset paths
PD4T_ROOT = RAW_DATA_DIR / "PD4T" / "PD4T" / "PD4T"
TULIP_ROOT = RAW_DATA_DIR / "TULIP"

# Model paths
TRAINED_MODELS_DIR = MODEL_ROOT / "trained"
CHECKPOINTS_DIR = MODEL_ROOT / "checkpoints"
VLM_MODELS_DIR = MODEL_ROOT / "vlm"

# Experiment paths
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"


# ============================================
# GPU Configuration
# ============================================
def get_gpu_config():
    """Get GPU configuration for current environment"""

    if CURRENT_ENV == Environment.LOCAL:
        return {
            "use_gpu": False,  # 로컬은 기본 CPU
            "device": "cpu",
            "batch_size": 16,
            "num_workers": 0,
            "mixed_precision": False,
        }

    elif CURRENT_ENV == Environment.HPC:
        return {
            "use_gpu": True,
            "device": "cuda",
            "batch_size": 64,      # V100 기준
            "num_workers": 4,
            "mixed_precision": True,  # AMP 사용
        }

    elif CURRENT_ENV == Environment.CLOUD:
        return {
            "use_gpu": True,
            "device": "cuda",
            "batch_size": 32,
            "num_workers": 2,
            "mixed_precision": True,
        }


GPU_CONFIG = get_gpu_config()


# ============================================
# VLM Configuration
# ============================================
VLM_CONFIG = {
    "local": {
        # 로컬: API 기반만 가능
        "supported_models": ["gpt-4-vision", "claude-3-opus"],
        "use_local_model": False,
    },
    "hpc": {
        # HPC: 로컬 VLM 가능
        "supported_models": [
            "qwen2-vl-7b",
            "llava-1.5-13b",
            "video-llama",
            "gpt-4-vision",  # API도 가능
        ],
        "use_local_model": True,
        "model_cache_dir": VLM_MODELS_DIR,
        "quantization": "4bit",  # 메모리 절약
    },
    "cloud": {
        "supported_models": [
            "qwen2-vl-72b",
            "llava-next-34b",
            "gpt-4-vision",
        ],
        "use_local_model": True,
        "model_cache_dir": VLM_MODELS_DIR,
        "quantization": None,  # 클라우드는 풀 정밀도
    },
}


# ============================================
# Utility Functions
# ============================================
def ensure_dirs():
    """Create all necessary directories"""
    dirs = [
        RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, CACHE_DIR,
        TRAINED_MODELS_DIR, CHECKPOINTS_DIR, VLM_MODELS_DIR,
        CONFIGS_DIR, RESULTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current configuration"""
    print("=" * 60)
    print(f"Hawkeye Environment Configuration")
    print("=" * 60)
    print(f"Environment: {CURRENT_ENV.value}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Model Root: {MODEL_ROOT}")
    print("-" * 60)
    print(f"GPU: {GPU_CONFIG['device']} (batch_size={GPU_CONFIG['batch_size']})")
    print(f"Mixed Precision: {GPU_CONFIG['mixed_precision']}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
