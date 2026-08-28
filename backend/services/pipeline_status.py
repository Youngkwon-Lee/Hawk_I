"""Read back the real labeling-to-inference pipeline state."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from services.finetuned_vlm import FinetunedVLMConfig, get_config
from services.handoff_manifest import verify_handoff_manifest


def _manifest_path() -> Path:
    configured = (os.getenv("HAWKEYE_LABEL_HANDOFF_MANIFEST") or "").strip()
    return Path(configured).expanduser() if configured else Path.home() / "gait_export" / "handoff-manifest.json"


def _model_ids(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    entries = payload.get("data") or payload.get("models") or []
    if not isinstance(entries, list):
        return set()
    model_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        value = entry.get("id") or entry.get("name") or entry.get("model")
        if isinstance(value, str) and value:
            model_ids.add(value)
    return model_ids


def probe_model(config: FinetunedVLMConfig | None, timeout_seconds: float = 4.0) -> dict[str, Any]:
    """Check the actual OpenAI-compatible model list without exposing its URL."""
    if config is None:
        return {
            "configured": False,
            "reachable": False,
            "model_present": False,
            "model": None,
            "condition": None,
            "error": "model endpoint is not configured",
        }
    try:
        response = requests.get(
            f"{config.base_url}/models",
            headers={"Authorization": f"Bearer {config.api_key}"} if config.api_key else {},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        models = _model_ids(response.json())
        present = config.model in models
        return {
            "configured": True,
            "reachable": True,
            "model_present": present,
            "model": config.model,
            "condition": config.condition,
            "error": None if present else "configured model was not listed by the endpoint",
        }
    except (requests.RequestException, ValueError) as exc:
        return {
            "configured": True,
            "reachable": False,
            "model_present": False,
            "model": config.model,
            "condition": config.condition,
            "error": f"model probe failed: {type(exc).__name__}",
        }


def get_pipeline_status(manifest_path: Path | None = None) -> dict[str, Any]:
    handoff = verify_handoff_manifest(manifest_path or _manifest_path())
    model = probe_model(get_config())

    binding = handoff.get("model_binding")
    if not isinstance(binding, dict):
        binding = {}
    binding_verified = (
        handoff.get("valid") is True
        and binding.get("status") == "verified"
        and binding.get("dataset_sha256") == handoff.get("dataset_sha256")
        and binding.get("model") == model.get("model")
    )
    model_ready = model["reachable"] and model["model_present"]
    handoff_ready = handoff.get("valid") is True

    if handoff_ready and model_ready and binding_verified:
        overall = "operational"
    elif handoff_ready and model_ready:
        overall = "connected_unbound"
    elif handoff_ready:
        overall = "handoff_ready"
    else:
        overall = "incomplete"

    splits = handoff.get("splits") if isinstance(handoff.get("splits"), dict) else {}
    total_records = sum(
        details.get("records", 0)
        for details in splits.values()
        if isinstance(details, dict) and isinstance(details.get("records"), int)
    )
    return {
        "success": True,
        "overall": overall,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "handoff": {
            "verified": handoff_ready,
            "handoff_id": handoff.get("handoff_id"),
            "created_at": handoff.get("created_at"),
            "dataset_sha256": handoff.get("dataset_sha256"),
            "total_records": total_records,
            "splits": splits,
            "errors": handoff.get("errors", []),
        },
        "training_binding": {
            "verified": binding_verified,
            "status": "verified" if binding_verified else "unverified",
            "reason": None if binding_verified else binding.get(
                "reason", "The serving model has not declared this export digest."
            ),
        },
        "model": model,
        "inference": {
            "ready": model_ready,
            "uses_verified_handoff": binding_verified,
        },
    }

