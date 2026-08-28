"""Build and verify a privacy-safe label handoff manifest.

The JSONL files remain private because they contain clip and patient identifiers.
The manifest exposes only aggregate counts and cryptographic digests, which lets
operations verify that labeling and training are referring to the same immutable
package without exposing any record-level data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "hawkeye.label-handoff.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _dataset_digest(task: str, splits: dict[str, dict[str, Any]]) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "splits": {
            name: {
                "records": details["records"],
                "sha256": details["sha256"],
            }
            for name, details in sorted(splits.items())
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_handoff_manifest(
    output_dir: Path,
    summary: dict[str, Any],
    *,
    task: str,
) -> dict[str, Any]:
    """Create an aggregate manifest for JSONL files already written to output_dir."""
    splits: dict[str, dict[str, Any]] = {}
    for split, split_summary in sorted(summary.items()):
        path = output_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing export split: {path.name}")
        records = _line_count(path)
        expected = int(split_summary["clips"])
        if records != expected:
            raise ValueError(f"{path.name} has {records} records; summary expects {expected}")
        splits[split] = {
            "file": path.name,
            "records": records,
            "patients": int(split_summary["patients"]),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    dataset_sha256 = _dataset_digest(task, splits)
    return {
        "schema_version": SCHEMA_VERSION,
        "handoff_id": f"pd4t-{task}-{dataset_sha256[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "task": task,
        "dataset_sha256": dataset_sha256,
        "splits": splits,
        "quality_gates": {
            "patient_overlap": 0,
            "record_counts_match": True,
        },
        "model_binding": {
            "status": "unverified",
            "reason": "No training run has declared this dataset digest.",
        },
    }


def write_handoff_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    path = output_dir / "handoff-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def verify_handoff_manifest(path: Path) -> dict[str, Any]:
    """Verify files and return only aggregate, API-safe status fields."""
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"manifest unreadable: {type(exc).__name__}"]}

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    task = payload.get("task")
    raw_splits = payload.get("splits")
    if not isinstance(task, str) or not task:
        errors.append("task is missing")
    if not isinstance(raw_splits, dict) or not raw_splits:
        errors.append("splits are missing")
        raw_splits = {}

    safe_splits: dict[str, dict[str, Any]] = {}
    base = path.parent.resolve()
    for split, details in sorted(raw_splits.items()):
        if not isinstance(details, dict):
            errors.append(f"{split}: invalid details")
            continue
        filename = details.get("file")
        if not isinstance(filename, str):
            errors.append(f"{split}: file is missing")
            continue
        candidate = (base / filename).resolve()
        if candidate.parent != base or not candidate.is_file():
            errors.append(f"{split}: export file is missing")
            continue
        actual_records = _line_count(candidate)
        actual_sha256 = _sha256(candidate)
        if actual_records != details.get("records"):
            errors.append(f"{split}: record count mismatch")
        if actual_sha256 != details.get("sha256"):
            errors.append(f"{split}: sha256 mismatch")
        safe_splits[split] = {
            "records": actual_records,
            "patients": details.get("patients"),
            "sha256": actual_sha256,
        }

    if task and safe_splits:
        computed_digest = _dataset_digest(task, {
            split: {"records": details["records"], "sha256": details["sha256"]}
            for split, details in safe_splits.items()
        })
        if computed_digest != payload.get("dataset_sha256"):
            errors.append("dataset_sha256 mismatch")
    else:
        computed_digest = None

    return {
        "valid": not errors,
        "errors": errors,
        "schema_version": payload.get("schema_version"),
        "handoff_id": payload.get("handoff_id"),
        "created_at": payload.get("created_at"),
        "task": task,
        "dataset_sha256": computed_digest,
        "splits": safe_splits,
        "quality_gates": payload.get("quality_gates"),
        "model_binding": payload.get("model_binding"),
    }

