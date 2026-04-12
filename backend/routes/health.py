"""
Health Check Route
"""

from flask import Blueprint, jsonify
import cv2
import mediapipe as mp

# PyTorch is optional (for UPDRS prediction)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

bp = Blueprint('health', __name__)


@bp.route('/health')
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
                "updrs_prediction": False  # Will be True after model integration
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
