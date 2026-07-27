import numpy as np
import pytest


torch = pytest.importorskip("torch")


def test_coral_scorer_loads_only_requested_model(tmp_path, monkeypatch):
    from models.coral_scorer import CORALScorer, MambaCoralModel

    config = {
        "input_size": 123,
        "hidden_size": 8,
        "num_layers": 1,
        "num_classes": 5,
        "dropout": 0.0,
    }
    model = MambaCoralModel(**config)
    torch.save(
        {"config": config, "model_state_dict": model.state_dict()},
        tmp_path / "finger_coral_raw_kfold_best.pth",
    )

    monkeypatch.setenv("HAWKEYE_CORAL_MODEL_DIR", str(tmp_path))
    CORALScorer._instance = None
    scorer = CORALScorer()

    prediction = scorer.predict(np.zeros((20, 123), dtype=np.float32), "finger_tapping")

    assert prediction is not None
    assert scorer.get_available_tasks() == ["finger_tapping"]
    assert scorer.get_load_error("finger_tapping") is None


def test_coral_scorer_reports_missing_checkpoint(tmp_path, monkeypatch):
    from models.coral_scorer import CORALScorer

    monkeypatch.setenv("HAWKEYE_CORAL_MODEL_DIR", str(tmp_path))
    CORALScorer._instance = None
    scorer = CORALScorer()

    prediction = scorer.predict(np.zeros((20, 123), dtype=np.float32), "finger_tapping")

    assert prediction is None
    assert scorer.get_load_error("finger_tapping") == "model_not_found"
