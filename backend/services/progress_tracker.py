"""
Analysis Progress Tracker
Tracks real-time progress of video analysis for frontend display
Persists to JSON file to survive Flask restarts in debug mode
"""

import json
import os
import sys
from pathlib import Path

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
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(ANALYSIS_PROGRESS, f, indent=2)
    except Exception as e:
        sys.stderr.write(f"[Progress Tracker] Error saving progress file: {e}\n")


# Load existing progress on module import
_load_progress()


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


def complete_analysis(video_id: str):
    """Mark analysis as completed"""
    if video_id not in ANALYSIS_PROGRESS:
        _load_progress()
        
    if video_id in ANALYSIS_PROGRESS:
        ANALYSIS_PROGRESS[video_id]["status"] = "completed"
        _save_progress()
        print(f"[Progress Tracker] Analysis completed for video_id: {video_id}")


def fail_analysis(video_id: str, error_message: str):
    """Mark analysis as failed"""
    if video_id not in ANALYSIS_PROGRESS:
        _load_progress()

    if video_id in ANALYSIS_PROGRESS:
        ANALYSIS_PROGRESS[video_id]["status"] = "error"
        ANALYSIS_PROGRESS[video_id]["error"] = error_message
        _save_progress()
        print(f"[Progress Tracker] Analysis failed for video_id: {video_id} - {error_message}")


def get_progress(video_id: str):
    """Get progress for a specific video"""
    if video_id not in ANALYSIS_PROGRESS:
        _load_progress()
        
    return ANALYSIS_PROGRESS.get(video_id, {
        "status": "not_found",
        "message": "Analysis not found for this video ID",
        "steps": {}
    })


def cleanup_old_progress(max_entries: int = 100):
    """Clean up old progress entries to prevent memory leak"""
    if len(ANALYSIS_PROGRESS) > max_entries:
        # Keep only the most recent entries
        keys = list(ANALYSIS_PROGRESS.keys())
        for key in keys[:-max_entries]:
            del ANALYSIS_PROGRESS[key]
        print(f"[Progress Tracker] Cleaned up {len(keys) - max_entries} old entries")
