import json

from flask import Flask

from routes import pipeline
from services import pipeline_status
from services.finetuned_vlm import FinetunedVLMConfig
from services.handoff_manifest import build_handoff_manifest, write_handoff_manifest


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"id": "hawkeye-c0b-seed42"}]}


def _manifest(tmp_path, binding=None):
    (tmp_path / "train.jsonl").write_text(json.dumps({"clip_id": "safe"}) + "\n")
    manifest = build_handoff_manifest(
        tmp_path,
        {"train": {"clips": 1, "patients": 1}},
        task="gait",
    )
    if binding is not None:
        manifest["model_binding"] = binding(manifest)
    return write_handoff_manifest(tmp_path, manifest)


def _configure(monkeypatch):
    monkeypatch.setattr(
        pipeline_status,
        "get_config",
        lambda: FinetunedVLMConfig(
            base_url="http://model.invalid/v1",
            model="hawkeye-c0b-seed42",
            condition="C0B",
        ),
    )
    monkeypatch.setattr(pipeline_status.requests, "get", lambda *args, **kwargs: _Response())


def test_connected_model_does_not_claim_current_export_was_trained(monkeypatch, tmp_path):
    path = _manifest(tmp_path)
    _configure(monkeypatch)

    status = pipeline_status.get_pipeline_status(path)

    assert status["overall"] == "connected_unbound"
    assert status["handoff"]["verified"] is True
    assert status["model"]["model_present"] is True
    assert status["training_binding"]["verified"] is False
    assert status["inference"]["ready"] is True
    assert status["inference"]["uses_verified_handoff"] is False


def test_matching_training_binding_makes_pipeline_operational(monkeypatch, tmp_path):
    path = _manifest(
        tmp_path,
        lambda manifest: {
            "status": "verified",
            "dataset_sha256": manifest["dataset_sha256"],
            "model": "hawkeye-c0b-seed42",
        },
    )
    _configure(monkeypatch)

    status = pipeline_status.get_pipeline_status(path)

    assert status["overall"] == "operational"
    assert status["training_binding"]["verified"] is True


def test_route_returns_safe_aggregate_status(monkeypatch, tmp_path):
    path = _manifest(tmp_path)
    monkeypatch.setattr(pipeline, "get_pipeline_status", lambda: pipeline_status.get_pipeline_status(path))
    _configure(monkeypatch)
    app = Flask(__name__)
    app.register_blueprint(pipeline.bp)

    response = app.test_client().get("/api/pipeline/status")

    assert response.status_code == 200
    assert response.get_json()["handoff"]["total_records"] == 1
    assert "media_path" not in response.get_data(as_text=True)

