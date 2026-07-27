import importlib
import json

import pytest


def _load_store(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_FOLDER", str(tmp_path))

    import services.analysis_job_store as analysis_job_store

    return importlib.reload(analysis_job_store)


def test_persisted_job_lifecycle(monkeypatch, tmp_path):
    store = _load_store(monkeypatch, tmp_path)
    payload = {
        "video_path": str(tmp_path / "finger.mp4"),
        "patient_id": "deidentified-subject",
        "medication_context": {"available": False},
    }

    created = store.create_job("analysis-1", payload)
    claimed = store.claim_job("analysis-1")
    store.mark_job_completed("analysis-1")
    completed = store.get_job("analysis-1")

    assert created["status"] == "queued"
    assert claimed["status"] == "running"
    assert claimed["attempts"] == 1
    assert completed["status"] == "completed"
    assert completed["payload"] == payload
    assert completed["completed_at"].endswith("Z")


def test_interrupted_job_can_be_requeued_and_claimed_again(monkeypatch, tmp_path):
    store = _load_store(monkeypatch, tmp_path)
    store.create_job("analysis-2", {"video_path": str(tmp_path / "gait.mp4")})
    store.claim_job("analysis-2")

    requeued = store.requeue_interrupted_job("analysis-2")
    claimed_again = store.claim_job("analysis-2")

    assert requeued["status"] == "queued"
    assert requeued["resume_count"] == 1
    assert claimed_again["status"] == "running"
    assert claimed_again["attempts"] == 2


def test_only_queued_and_running_jobs_are_resumable(monkeypatch, tmp_path):
    store = _load_store(monkeypatch, tmp_path)
    store.create_job("queued", {"video_path": "queued.mp4"})
    store.create_job("completed", {"video_path": "completed.mp4"})
    store.claim_job("completed")
    store.mark_job_completed("completed")
    store.create_job("failed", {"video_path": "failed.mp4"})
    store.claim_job("failed")
    store.mark_job_failed("failed", "bad video", "analysis_failed")
    store.create_job("running", {"video_path": "running.mp4"})
    store.claim_job("running")

    resumable_ids = [job["video_id"] for job in store.list_resumable_jobs()]

    assert resumable_ids == ["queued", "running"]


def test_failed_job_serialization_preserves_existing_queue(monkeypatch, tmp_path):
    store = _load_store(monkeypatch, tmp_path)
    store.create_job("safe", {"video_path": "safe.mp4"})

    with pytest.raises(TypeError):
        store.create_job("invalid", {"not_json": object()})

    persisted = json.loads(store.JOB_FILE.read_text())
    assert list(persisted) == ["safe"]
    assert list(tmp_path.glob(".*.tmp")) == []
