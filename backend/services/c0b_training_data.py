"""Validate and stage the immutable gait-label handoff for C0B training.

The source export contains patient and clip identifiers plus local media paths.
Training only needs a video, split, and clinician score, so staging replaces
identifiers and filenames with dataset-bound opaque IDs before data leaves the
labeling host.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .handoff_manifest import verify_handoff_manifest
from .vlm_training_contract import (
    GAIT_ANCHOR,
    GAIT_GLOSSARY,
    GAIT_QUESTION,
    SYSTEM_ANCHOR,
)


STAGING_SCHEMA_VERSION = "hawkeye.c0b-training-stage.v1"
EXPECTED_SPLITS = ("train", "validation", "test")
WINDOWS_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


class C0BDataError(ValueError):
    """Raised when a handoff cannot safely be used for model training."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prompt_contract_sha256() -> str:
    """Fingerprint the exact prompt order and text used by the C0B adapter."""
    return canonical_sha256(
        {
            "order": ["SYSTEM", "ANCHOR", "GLOSSARY", "QUESTION", "VIDEO"],
            "system": SYSTEM_ANCHOR,
            "anchor": GAIT_ANCHOR,
            "glossary": GAIT_GLOSSARY,
            "question": GAIT_QUESTION,
            "target": "answer: <integer 0-4>",
        }
    )


def resolve_media_path(raw_path: str, *, drive_root: Path = Path("/mnt")) -> Path:
    """Resolve a native path or a Windows drive path from WSL/Linux."""
    match = WINDOWS_PATH.match(raw_path.strip())
    if match and os.name != "nt":
        drive, remainder = match.groups()
        return drive_root / drive.lower() / Path(remainder.replace("\\", "/"))
    return Path(raw_path).expanduser()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise C0BDataError(f"{path.name}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise C0BDataError(f"{path.name}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _validate_score(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
        raise C0BDataError(f"{context}: updrs_3_10 must be an integer from 0 to 4")
    return value


def validate_handoff(
    export_dir: Path,
    *,
    expected_dataset_sha256: str | None = None,
    require_media: bool = True,
    drive_root: Path = Path("/mnt"),
) -> dict[str, Any]:
    """Verify digests, split isolation, row schema, labels, and media access."""
    manifest_path = export_dir / "handoff-manifest.json"
    verified = verify_handoff_manifest(manifest_path)
    if not verified.get("valid"):
        errors = "; ".join(verified.get("errors") or ["unknown manifest error"])
        raise C0BDataError(f"handoff manifest verification failed: {errors}")

    dataset_sha256 = verified.get("dataset_sha256")
    if expected_dataset_sha256 and dataset_sha256 != expected_dataset_sha256:
        raise C0BDataError(
            f"dataset SHA mismatch: expected {expected_dataset_sha256}, got {dataset_sha256}"
        )

    patients: dict[str, set[str]] = {}
    clip_ids: set[str] = set()
    split_summary: dict[str, Any] = {}
    media_bytes = 0

    for split in EXPECTED_SPLITS:
        path = export_dir / f"{split}.jsonl"
        if not path.is_file():
            raise C0BDataError(f"missing split: {path.name}")
        rows = load_jsonl(path)
        split_patients: set[str] = set()
        score_counts: Counter[int] = Counter()
        missing_media = 0

        for index, row in enumerate(rows):
            context = f"{path.name}:{index + 1}"
            if row.get("split") != split:
                raise C0BDataError(f"{context}: split field does not match filename")
            if row.get("task") != "gait":
                raise C0BDataError(f"{context}: only gait rows are supported")
            clip_id = str(row.get("clip_id") or "").strip()
            patient_id = str(row.get("patient_id") or "").strip()
            media_path = str(row.get("media_path") or "").strip()
            if not clip_id or not patient_id or not media_path:
                raise C0BDataError(f"{context}: clip_id, patient_id, and media_path are required")
            if clip_id in clip_ids:
                raise C0BDataError(f"{context}: duplicate clip_id")
            clip_ids.add(clip_id)
            split_patients.add(patient_id)
            score_counts[_validate_score(row.get("updrs_3_10"), context=context)] += 1

            resolved = resolve_media_path(media_path, drive_root=drive_root)
            if resolved.is_file():
                media_bytes += resolved.stat().st_size
            else:
                missing_media += 1

        if require_media and missing_media:
            raise C0BDataError(f"{path.name}: {missing_media} media files are inaccessible")
        patients[split] = split_patients
        split_summary[split] = {
            "records": len(rows),
            "patients": len(split_patients),
            "score_counts": {str(k): score_counts[k] for k in sorted(score_counts)},
            "missing_media": missing_media,
            "sha256": sha256_file(path),
        }

    for index, left in enumerate(EXPECTED_SPLITS):
        for right in EXPECTED_SPLITS[index + 1 :]:
            if patients[left] & patients[right]:
                raise C0BDataError(f"patient overlap detected between {left} and {right}")

    return {
        "dataset_sha256": dataset_sha256,
        "handoff_id": verified.get("handoff_id"),
        "task": "gait",
        "splits": split_summary,
        "media_bytes": media_bytes,
        "patient_overlap": 0,
        "prompt_contract_sha256": prompt_contract_sha256(),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stage_handoff(
    export_dir: Path,
    output_dir: Path,
    *,
    expected_dataset_sha256: str | None = None,
    drive_root: Path = Path("/mnt"),
    copy_media: bool = True,
) -> dict[str, Any]:
    """Create a minimal, opaque, dataset-bound package for GPU training."""
    validation = validate_handoff(
        export_dir,
        expected_dataset_sha256=expected_dataset_sha256,
        require_media=True,
        drive_root=drive_root,
    )
    dataset_sha256 = validation["dataset_sha256"]
    existing_manifest = output_dir / "training-stage-manifest.json"
    if existing_manifest.is_file():
        return verify_training_stage(
            output_dir,
            expected_dataset_sha256=dataset_sha256,
            require_media=copy_media,
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise C0BDataError(
            "staging output is not empty and has no verifiable manifest; use a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    if copy_media:
        media_dir.mkdir(parents=True, exist_ok=True)

    staged_splits: dict[str, Any] = {}
    for split in EXPECTED_SPLITS:
        source_rows = load_jsonl(export_dir / f"{split}.jsonl")
        staged_rows: list[dict[str, Any]] = []
        media_identity: list[dict[str, Any]] = []
        for row in source_rows:
            opaque_id = hashlib.sha256(
                f"{dataset_sha256}:{split}:{row['clip_id']}".encode("utf-8")
            ).hexdigest()[:24]
            source_media = resolve_media_path(str(row["media_path"]), drive_root=drive_root)
            suffix = source_media.suffix.lower() or ".mp4"
            relative_media = Path("media") / f"{opaque_id}{suffix}"
            destination = output_dir / relative_media
            source_media_sha256 = sha256_file(source_media)
            source_media_bytes = source_media.stat().st_size
            if copy_media:
                if destination.exists() and sha256_file(destination) != source_media_sha256:
                    raise C0BDataError(f"staging collision for opaque sample {opaque_id}")
                if not destination.exists():
                    shutil.copy2(source_media, destination)
            media_record = {
                "sample_id": opaque_id,
                "sha256": source_media_sha256,
                "bytes": source_media_bytes,
            }
            media_identity.append(media_record)
            staged_rows.append({
                "sample_id": opaque_id,
                "split": split,
                "task": "gait",
                "score": int(row["updrs_3_10"]),
                "media_path": relative_media.as_posix(),
                "media_sha256": source_media_sha256,
                "media_bytes": source_media_bytes,
            })

        staged_path = output_dir / f"{split}.jsonl"
        _write_jsonl(staged_path, staged_rows)
        staged_splits[split] = {
            "file": staged_path.name,
            "records": len(staged_rows),
            "sha256": sha256_file(staged_path),
            "score_counts": validation["splits"][split]["score_counts"],
            "media_identity_sha256": canonical_sha256(media_identity),
            "media_bytes": sum(item["bytes"] for item in media_identity),
        }

    manifest = {
        "schema_version": STAGING_SCHEMA_VERSION,
        "source_handoff_id": validation["handoff_id"],
        "source_dataset_sha256": dataset_sha256,
        "prompt_contract_sha256": validation["prompt_contract_sha256"],
        "task": "gait",
        "target": "clinician_updrs_3_10_0_4",
        "privacy": {
            "patient_identifiers_included": False,
            "source_clip_identifiers_included": False,
            "source_paths_included": False,
        },
        "splits": staged_splits,
        "test_policy": "locked_until_validation_model_selection_is_frozen",
    }
    manifest["stage_sha256"] = canonical_sha256(manifest)
    manifest_path = output_dir / "training-stage-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_training_stage(
    data_dir: Path,
    *,
    expected_dataset_sha256: str | None = None,
    require_media: bool = True,
) -> dict[str, Any]:
    """Verify a staged package without exposing record-level content."""
    path = data_dir / "training-stage-manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C0BDataError("training stage manifest is unreadable") from exc
    if manifest.get("schema_version") != STAGING_SCHEMA_VERSION:
        raise C0BDataError("unsupported training stage schema")
    if expected_dataset_sha256 and manifest.get("source_dataset_sha256") != expected_dataset_sha256:
        raise C0BDataError("staged dataset SHA does not match the requested handoff")
    if manifest.get("prompt_contract_sha256") != prompt_contract_sha256():
        raise C0BDataError("staged prompt contract does not match this code revision")
    expected_stage_sha = manifest.get("stage_sha256")
    unsigned = dict(manifest)
    unsigned.pop("stage_sha256", None)
    if canonical_sha256(unsigned) != expected_stage_sha:
        raise C0BDataError("training stage manifest digest mismatch")

    seen: set[str] = set()
    for split in EXPECTED_SPLITS:
        details = (manifest.get("splits") or {}).get(split)
        if not isinstance(details, dict):
            raise C0BDataError(f"staged split metadata missing: {split}")
        split_path = data_dir / str(details.get("file"))
        if not split_path.is_file() or sha256_file(split_path) != details.get("sha256"):
            raise C0BDataError(f"staged split digest mismatch: {split}")
        rows = load_jsonl(split_path)
        if len(rows) != details.get("records"):
            raise C0BDataError(f"staged split record count mismatch: {split}")
        media_identity: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            context = f"{split_path.name}:{index + 1}"
            if row.get("split") != split or row.get("task") != "gait":
                raise C0BDataError(f"{context}: invalid staged split/task")
            sample_id = str(row.get("sample_id") or "")
            if not re.fullmatch(r"[0-9a-f]{24}", sample_id) or sample_id in seen:
                raise C0BDataError(f"{context}: invalid or duplicate opaque sample_id")
            seen.add(sample_id)
            _validate_score(row.get("score"), context=context)
            media_path = Path(str(row.get("media_path") or ""))
            if media_path.is_absolute() or ".." in media_path.parts:
                raise C0BDataError(f"{context}: media_path must remain inside the stage")
            expected_media_sha256 = str(row.get("media_sha256") or "")
            expected_media_bytes = row.get("media_bytes")
            if not re.fullmatch(r"[0-9a-f]{64}", expected_media_sha256):
                raise C0BDataError(f"{context}: invalid media SHA")
            if isinstance(expected_media_bytes, bool) or not isinstance(expected_media_bytes, int):
                raise C0BDataError(f"{context}: invalid media byte count")
            media_identity.append({
                "sample_id": sample_id,
                "sha256": expected_media_sha256,
                "bytes": expected_media_bytes,
            })
            if require_media:
                candidate = data_dir / media_path
                if not candidate.is_file():
                    raise C0BDataError(f"{context}: staged media is missing")
                if candidate.stat().st_size != expected_media_bytes:
                    raise C0BDataError(f"{context}: staged media byte count mismatch")
                if sha256_file(candidate) != expected_media_sha256:
                    raise C0BDataError(f"{context}: staged media digest mismatch")
        if canonical_sha256(media_identity) != details.get("media_identity_sha256"):
            raise C0BDataError(f"staged media identity mismatch: {split}")
        if sum(item["bytes"] for item in media_identity) != details.get("media_bytes"):
            raise C0BDataError(f"staged media bytes mismatch: {split}")
    return manifest
