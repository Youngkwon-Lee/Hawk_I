import importlib


def test_prediction_status_is_false_without_trained_model_or_endpoint(monkeypatch):
    health = importlib.import_module("routes.health")
    coral = importlib.import_module("models.coral_scorer")

    monkeypatch.setattr(coral, "MODEL_DIR", "/tmp/hawkeye-no-models")
    monkeypatch.delenv("HAWKEYE_VLM_BASE_URL", raising=False)
    monkeypatch.delenv("HAWKEYE_VLM_MODEL", raising=False)

    status = health._prediction_status()

    assert status["updrs_prediction"] is False
    assert status["updrs_prediction_methods"] == []
    assert status["finetuned_vlm_configured"] is False


def test_prediction_status_reports_configured_finetuned_endpoint(monkeypatch, tmp_path):
    health = importlib.import_module("routes.health")
    coral = importlib.import_module("models.coral_scorer")

    monkeypatch.setattr(coral, "MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("HAWKEYE_VLM_BASE_URL", "https://gpu.example/v1")
    monkeypatch.setenv("HAWKEYE_VLM_MODEL", "hawkeye-c3be")

    status = health._prediction_status()

    assert status["updrs_prediction"] is True
    assert status["updrs_prediction_methods"] == ["finetuned_vlm"]
    assert status["finetuned_vlm_model"] == "hawkeye-c3be"
    assert status["finetuned_vlm_condition"] == "C0B"
