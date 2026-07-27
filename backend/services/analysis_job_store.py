"""Persistent job metadata for restart-resumable Hawk I analyses."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.json_store import atomic_write_json


UPLOAD_DIR = os.getenv("UPLOAD_FOLDER", "./uploads")
if not os.path.isabs(UPLOAD_DIR):
    UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)

JOB_FILE = Path(UPLOAD_DIR) / "analysis_jobs.json"
MAX_RESUME_ATTEMPTS = 3
_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_jobs() -> dict[str, dict[str, Any]]:
    if not JOB_FILE.exists():
        return {}
    try:
        with open(JOB_FILE, "r", encoding="utf-8") as job_file:
            loaded = json.load(job_file)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_jobs(jobs: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(JOB_FILE, jobs, ensure_ascii=False, indent=2)


def create_job(video_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist a queued job before its background thread is started."""
    with _LOCK:
        jobs = _load_jobs()
        if video_id in jobs:
            raise ValueError(f"analysis job already exists: {video_id}")

        now = _utc_now()
        job = {
            "video_id": video_id,
            "status": "queued",
            "attempts": 0,
            "resume_count": 0,
            "created_at": now,
            "updated_at": now,
            "payload": payload,
        }
        jobs[video_id] = job
        _save_jobs(jobs)
        return dict(job)


def get_job(video_id: str) -> dict[str, Any] | None:
    with _LOCK:
        job = _load_jobs().get(video_id)
        return dict(job) if isinstance(job, dict) else None


def list_resumable_jobs() -> list[dict[str, Any]]:
    """Return queued or abruptly interrupted jobs in creation order."""
    with _LOCK:
        jobs = [
            dict(job)
            for job in _load_jobs().values()
            if isinstance(job, dict) and job.get("status") in {"queued", "running"}
        ]
    return sorted(jobs, key=lambda job: str(job.get("created_at", "")))


def claim_job(video_id: str) -> dict[str, Any] | None:
    """Atomically transition a queued job to running and count the attempt."""
    with _LOCK:
        jobs = _load_jobs()
        job = jobs.get(video_id)
        if not isinstance(job, dict) or job.get("status") != "queued":
            return None

        now = _utc_now()
        job["status"] = "running"
        job["attempts"] = int(job.get("attempts", 0)) + 1
        job["started_at"] = now
        job["updated_at"] = now
        job.pop("last_error", None)
        _save_jobs(jobs)
        return dict(job)


def requeue_interrupted_job(video_id: str) -> dict[str, Any] | None:
    """Move a previously running job back to the queue for startup recovery."""
    with _LOCK:
        jobs = _load_jobs()
        job = jobs.get(video_id)
        if not isinstance(job, dict) or job.get("status") != "running":
            return None

        job["status"] = "queued"
        job["resume_count"] = int(job.get("resume_count", 0)) + 1
        job["updated_at"] = _utc_now()
        _save_jobs(jobs)
        return dict(job)


def mark_job_completed(video_id: str) -> None:
    with _LOCK:
        jobs = _load_jobs()
        job = jobs.get(video_id)
        if not isinstance(job, dict):
            return
        now = _utc_now()
        job["status"] = "completed"
        job["completed_at"] = now
        job["updated_at"] = now
        job.pop("last_error", None)
        _save_jobs(jobs)


def mark_job_failed(video_id: str, error_message: str, error_code: str) -> None:
    with _LOCK:
        jobs = _load_jobs()
        job = jobs.get(video_id)
        if not isinstance(job, dict):
            return
        now = _utc_now()
        job["status"] = "failed"
        job["last_error"] = error_message
        job["error_code"] = error_code
        job["failed_at"] = now
        job["updated_at"] = now
        _save_jobs(jobs)
