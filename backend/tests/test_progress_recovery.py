import importlib
import json


def _load_tracker(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_FOLDER", str(tmp_path))

    import services.progress_tracker as progress_tracker

    return importlib.reload(progress_tracker)


def test_recovery_completes_valid_result_and_fails_unfinished_job(monkeypatch, tmp_path):
    tracker = _load_tracker(monkeypatch, tmp_path)
    progress = {
        "has-result": {
            "status": "in_progress",
            "steps": {"metrics": {"status": "in_progress", "result_url": None}},
        },
        "needs-retry": {
            "status": "in_progress",
            "steps": {"skeleton": {"status": "in_progress", "result_url": None}},
        },
        "already-complete": {"status": "completed", "steps": {}},
    }
    tracker.atomic_write_json(tracker.PROGRESS_FILE, progress)
    tracker.atomic_write_json(tmp_path / "has-result_result.json", {"id": "has-result"})

    recovered = tracker.recover_interrupted_analyses()

    assert recovered == {
        "completed": ["has-result"],
        "interrupted": ["needs-retry"],
    }

    persisted = json.loads(tracker.PROGRESS_FILE.read_text())
    assert persisted["has-result"]["status"] == "completed"
    assert persisted["has-result"]["recovered_at"].endswith("Z")
    assert persisted["has-result"]["steps"]["metrics"]["status"] == "completed"
    assert persisted["needs-retry"]["status"] == "error"
    assert persisted["needs-retry"]["error_code"] == "service_restarted"
    assert persisted["needs-retry"]["retryable"] is True
    assert persisted["needs-retry"]["steps"]["skeleton"]["status"] == "error"
    assert persisted["already-complete"] == {"status": "completed", "steps": {}}


def test_recovery_is_idempotent(monkeypatch, tmp_path):
    tracker = _load_tracker(monkeypatch, tmp_path)
    tracker.atomic_write_json(
        tracker.PROGRESS_FILE,
        {"needs-retry": {"status": "in_progress", "steps": {}}},
    )

    first = tracker.recover_interrupted_analyses()
    persisted_after_first = tracker.PROGRESS_FILE.read_text()
    second = tracker.recover_interrupted_analyses()

    assert first == {"completed": [], "interrupted": ["needs-retry"]}
    assert second == {"completed": [], "interrupted": []}
    assert tracker.PROGRESS_FILE.read_text() == persisted_after_first


def test_recovery_rejects_invalid_result_json(monkeypatch, tmp_path):
    tracker = _load_tracker(monkeypatch, tmp_path)
    tracker.atomic_write_json(
        tracker.PROGRESS_FILE,
        {"invalid-result": {"status": "in_progress", "steps": {}}},
    )
    (tmp_path / "invalid-result_result.json").write_text("not-json")

    recovered = tracker.recover_interrupted_analyses()

    assert recovered == {"completed": [], "interrupted": ["invalid-result"]}
    assert tracker.get_progress("invalid-result")["error_code"] == "service_restarted"


def test_resume_analysis_resets_steps_and_records_attempt(monkeypatch, tmp_path):
    tracker = _load_tracker(monkeypatch, tmp_path)
    tracker.atomic_write_json(
        tracker.PROGRESS_FILE,
        {
            "resume-me": {
                "status": "in_progress",
                "steps": {"skeleton": {"status": "in_progress"}},
            }
        },
    )

    tracker.resume_analysis("resume-me", "finger_tapping", 2)

    progress = tracker.get_progress("resume-me")
    assert progress["status"] == "in_progress"
    assert progress["task_type"] == "finger_tapping"
    assert progress["resumed"] is True
    assert progress["resume_attempt"] == 2
    assert progress["steps"]["skeleton"]["status"] == "pending"


def test_complete_analysis_normalizes_active_step(monkeypatch, tmp_path):
    tracker = _load_tracker(monkeypatch, tmp_path)
    tracker.init_analysis("completed-after-restart", "finger_tapping")
    tracker.update_step("completed-after-restart", "metrics", "in_progress")

    tracker.complete_analysis("completed-after-restart")

    progress = tracker.get_progress("completed-after-restart")
    assert progress["status"] == "completed"
    assert progress["steps"]["metrics"]["status"] == "completed"


def test_fail_analysis_creates_missing_progress_entry(monkeypatch, tmp_path):
    tracker = _load_tracker(monkeypatch, tmp_path)

    tracker.fail_analysis(
        "missing-video",
        "Saved video is unavailable",
        "saved_video_unavailable",
        retryable=True,
    )

    progress = tracker.get_progress("missing-video")
    assert progress["status"] == "error"
    assert progress["error_code"] == "saved_video_unavailable"
    assert progress["retryable"] is True
