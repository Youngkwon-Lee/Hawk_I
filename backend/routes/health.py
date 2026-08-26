"""
Health Check Route
"""

from flask import Blueprint, jsonify
import cv2
import mediapipe as mp
import os


def _prediction_status():
    """Report configured model paths without loading a heavyweight model in health checks."""
    coral_files = {
        "gait": "gait_coral_raw_kfold_best.pth",
        "finger_tapping": "finger_coral_raw_kfold_best.pth",
        "hand_movement": "hand_coral_raw_kfold_best.pth",
        "leg_agility": "leg_coral_raw_kfold_best.pth",
    }
    try:
        from models.coral_scorer import MODEL_DIR, TORCH_AVAILABLE
        coral_tasks = [
            task for task, filename in coral_files.items()
            if TORCH_AVAILABLE and os.path.isfile(os.path.join(MODEL_DIR, filename))
        ]
    except Exception:
        coral_tasks = []

    try:
        from services.finetuned_vlm import get_config
        finetuned_config = get_config()
    except Exception:
        finetuned_config = None

    methods = []
    if coral_tasks:
        methods.append("coral")
    if finetuned_config:
        methods.append("finetuned_vlm")

    return {
        # This flag means a trained prediction model is actually configured,
        # not merely that rule-based scoring can produce a number.
        "updrs_prediction": bool(methods),
        "updrs_prediction_methods": methods,
        "coral_tasks": coral_tasks,
        "finetuned_vlm_configured": bool(finetuned_config),
        "finetuned_vlm_model": finetuned_config.model if finetuned_config else None,
        "finetuned_vlm_condition": finetuned_config.condition if finetuned_config else None,
    }

# PyTorch is optional (for UPDRS prediction)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

bp = Blueprint('health', __name__)


@bp.route('/health')
@bp.route('/api/health')
def health_check():
    """
    Health check endpoint
    Returns service status and dependency versions
    """
    try:
        # Check OpenCV
        opencv_version = cv2.__version__

        # Check MediaPipe
        mp_version = mp.__version__

        # Check PyTorch (optional)
        if TORCH_AVAILABLE:
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
        else:
            torch_version = "Not installed"
            cuda_available = False
            cuda_device = None

        return jsonify({
            "status": "healthy",
            "service": "HawkEye PD Backend",
            "dependencies": {
                "opencv": opencv_version,
                "mediapipe": mp_version,
                "pytorch": torch_version,
                "cuda_available": cuda_available,
                "cuda_device": cuda_device
            },
            "capabilities": {
                "roi_detection": True,
                "task_classification": True,
                "skeleton_extraction": True,
                "rule_based_scoring": True,
                **_prediction_status(),
            },
        }), 200

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
