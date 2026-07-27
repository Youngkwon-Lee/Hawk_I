"""
Analysis Progress Tracker
Tracks real-time progress of video analysis for frontend display
Persists to JSON file to survive Flask restarts in debug mode
"""

import json
import os
import sys
import threading
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from services.json_store import atomic_write_json

# Progress file path
# Use absolute path to ensure consistency
# Default to ./uploads/analysis_progress.json relative to CWD (where app.py runs)
UPLOAD_DIR = os.getenv('UPLOAD_FOLDER', './uploads')
if not os.path.isabs(UPLOAD_DIR):
    UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)
PROGRESS_FILE = Path(UPLOAD_DIR) / "analysis_progress.json"

# Global dictionary to track analysis progress
# Structure: {video_id: {status, steps: {step_name: {status, result_url}}}}
ANALYSIS_PROGRESS = {}
_LOCK = threading.RLock()


def _utc_now():
    """Return an ISO-8601 UTC timestamp for persisted recovery metadata."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _synchronized(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        with _LOCK:
            return function(*args, **kwargs)

    return wrapper


def _load_progress():
    """Load progress from file"""
    global ANALYSIS_PROGRESS
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                ANALYSIS_PROGRESS = json.load(f)
        except Exception as e:
            print(f"[Progress Tracker] Error loading progress file: {e}")
            ANALYSIS_PROGRESS = {}


def _save_progress():
    """Save progress to file"""
    try:
        os.makedirs(PROGRESS_FILE.parent, exist_ok=True)
        sys.stderr.write(f"[Progress Tracker] Saving to {PROGRESS_FILE}\n")
        atomic_write_json(PROGRESS_FILE, ANALYSIS_PROGRESS, indent=2)
    except Exception as e:
        sys.stderr.write(f"[Progress Tracker] Error saving progress file: {e}\n")


# Load existing progress on module import
_load_progress()


@_synchronized
def init_analysis(video_id: str, task_type: str = "gait"):
    """Initialize analysis progress tracking"""
    # Reload to ensure we have latest state
    _load_progress()
    
    ANALYSIS_PROGRESS[video_id] = {
        "status": "in_progress",
        "task_type": task_type,
        "steps": {
            "roi_detection": {"status": "pending", "result_url": None},
            "skeleton": {"status": "pending", "result_url": None},
            "heatmap": {"status": "pending", "result_url": None},
            "temporal_map": {"status": "pending", "result_url": None},
            "attention_map": {"status": "pending", "result_url": None},
            "overlay_video": {"status": "pending", "result_url": None},
            "metrics": {"status": "pending", "result_url": None},
            "updrs_calculation": {"status": "pending", "result_url": None},
            "gait_cycle": {"status": "pending", "result_url": None},
            "validation": {"status": "pending", "result_url": None},
            "ai_interpretation": {"status": "pending", "result_url": None}
        }
    }
    _save_progress()
    print(f"[Progress Tracker] Initialized tracking for video_id: {video_id}")


@_synchronized
def resume_analysis(video_id: str, task_type: str, resume_attempt: int):
    """Reset interrupted step progress before a persisted job is rerun."""
    init_analysis(video_id, task_type=task_type)
    ANALYSIS_PROGRESS[video_id]["resumed"] = True
    ANALYSIS_PROGRESS[video_id]["resume_attempt"] = resume_attempt
    ANALYSIS_PROGRESS[video_id]["resumed_at"] = _utc_now()
    _save_progress()


@_synchronized
def update_step(video_id: str, step_name: str, status: str, result_url: str = None):
    """
    Update progress for a specific step
    """
    # Reload to ensure we have latest state (in case updated by another thread/process)
    if video_id not in ANALYSIS_PROGRESS:
        _load_progress()

    if video_id not in ANALYSIS_PROGRESS:
        print(f"[Progress Tracker] Warning: video_id {video_id} not found. Initializing...")
        init_analysis(video_id)

    if step_name in ANALYSIS_PROGRESS[video_id]["steps"]:
        ANALYSIS_PROGRESS[video_id]["steps"][step_name]["status"] = status
        if result_url:
            ANALYSIS_PROGRESS[video_id]["steps"][step_name]["result_url"] = result_url
        _save_progress()
        print(f"[Progress Tracker] {video_id} - {step_name}: {status}")
    else:
        print(f"[Progress Tracker] Warning: Unknown step '{step_name}'")


@_synchronized
def complete_analysis(video_id: str):
    """Mark analysis as completed"""
    if video_id not in ANALYSIS_PROGRESS:
        _load_progress()
        
    if video_id in ANALYSIS_PROGRESS:
        progress = ANALYSIS_PROGRESS[video_id]
        progress["status"] = "completed"
        progress.pop("error", None)
        progress.pop("error_code", None)
        progress.pop("retryable", None)
        steps = progress.get("steps")
        if isinstance(steps, dict):
            for step in steps.values():
                if isinstance(step, dict) and step.get("status") == "in_progress":
                    step["status"] = "completed"
        _save_progress()
        print(f"[Progress Tracker] Analysis completed for video_id: {video_id}")


@_synchronized
def fail_analysis(
    video_id: str,
    error_message: str,
    error_code: str | None = None,
    retryable: bool = False,
):
    """Mark analysis as failed"""
    if video_id not in ANALYSIS_PROGRESS:
        _load_progress()

    if video_id not in ANALYSIS_PROGRESS:
        ANALYSIS_PROGRESS[video_id] = {"status": "error", "steps": {}}

    ANALYSIS_PROGRESS[video_id]["status"] = "error"
    ANALYSIS_PROGRESS[video_id]["error"] = error_message
    ANALYSIS_PROGRESS[video_id]["retryable"] = retryable
    if error_code:
        ANALYSIS_PROGRESS[video_id]["error_code"] = error_code
    _save_progress()
    print(f"[Progress Tracker] Analysis failed for video_id: {video_id} - {error_message}")


@_synchronized
def recover_interrupted_analyses(exclude_video_ids=None):
    """Resolve analyses left in progress by a previous single-worker process.

    A complete, atomically-written result wins over stale progress. Otherwise the
    interrupted analysis becomes a retryable error so clients stop polling and
    can ask the user to submit the video again.
    """
    _load_progress()

    excluded = set(exclude_video_ids or ())
    recovered = {"completed": [], "interrupted": []}
    recovered_at = _utc_now()

    for video_id, progress in ANALYSIS_PROGRESS.items():
        if video_id in excluded:
            continue
        if not isinstance(progress, dict) or progress.get("status") != "in_progress":
            continue

        result_path = PROGRESS_FILE.parent / f"{video_id}_result.json"
        result_is_valid = False
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as result_file:
                    result_is_valid = isinstance(json.load(result_file), dict)
            except (OSError, json.JSONDecodeError):
                result_is_valid = False

        if result_is_valid:
            progress["status"] = "completed"
            progress["recovered_at"] = recovered_at
            progress.pop("error", None)
            progress.pop("error_code", None)
            progress.pop("retryable", None)
            steps = progress.get("steps")
            if isinstance(steps, dict):
                for step in steps.values():
                    if isinstance(step, dict) and step.get("status") == "in_progress":
                        step["status"] = "completed"
            recovered["completed"].append(video_id)
            continue

        progress["status"] = "error"
        progress["error"] = (
            "Analysis was interrupted because the backend restarted. "
            "Please submit the video again."
        )
        progress["error_code"] = "service_restarted"
        progress["retryable"] = True
        progress["recovered_at"] = recovered_at

        steps = progress.get("steps")
        if isinstance(steps, dict):
            for step in steps.values():
                if isinstance(step, dict) and step.get("status") == "in_progress":
                    step["status"] = "error"

        recovered["interrupted"].append(video_id)

    if recovered["completed"] or recovered["interrupted"]:
        _save_progress()

    return recovered


@_synchronized
def get_progress(video_id: str):
    """Get progress for a specific video"""
    if video_id not in ANALYSIS_PROGRESS:
        _load_progress()
        
    return ANALYSIS_PROGRESS.get(video_id, {
        "status": "not_found",
        "message": "Analysis not found for this video ID",
        "steps": {}
    })


@_synchronized
def cleanup_old_progress(max_entries: int = 100):
    """Clean up old progress entries to prevent memory leak"""
    if len(ANALYSIS_PROGRESS) > max_entries:
        # Keep only the most recent entries
        keys = list(ANALYSIS_PROGRESS.keys())
        for key in keys[:-max_entries]:
            del ANALYSIS_PROGRESS[key]
        print(f"[Progress Tracker] Cleaned up {len(keys) - max_entries} old entries")
