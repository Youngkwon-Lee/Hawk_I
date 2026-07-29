"""Resolve analysis-owned media without trusting browser-provided file paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit


MEDIA_ASSET_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "skeleton_video": (("skeleton_data", "skeleton_video_url"),),
    "original_video": (("skeleton_data", "original_video_url"),),
    "heatmap": (
        ("visualization_urls", "heatmap"),
        ("visualization_maps", "heatmap_url"),
    ),
    "temporal_map": (
        ("visualization_urls", "temporal_map"),
        ("visualization_maps", "temporal_map_url"),
    ),
    "attention_map": (
        ("visualization_urls", "attention_map"),
        ("visualization_maps", "attention_map_url"),
    ),
    "overlay_video": (("visualization_maps", "overlay_video_url"),),
}
MEDIA_ASSETS = frozenset(MEDIA_ASSET_FIELDS)
ANALYSIS_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$")


@dataclass(frozen=True)
class FileAccessDecision:
    internal: bool
    protected_result: dict[str, Any] | None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_analysis_result(upload_folder: str, analysis_id: str) -> dict[str, Any] | None:
    if not ANALYSIS_ID_PATTERN.fullmatch(analysis_id) or ".." in analysis_id:
        return None
    return _load_json(Path(upload_folder).resolve() / f"{analysis_id}_result.json")


def _field_value(result: dict[str, Any], section: str, field: str) -> str | None:
    payload = result.get(section)
    payload = payload if isinstance(payload, dict) else {}
    value = payload.get(field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _safe_generated_filename(value: str) -> str | None:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return None

    decoded_path = unquote(parsed.path)
    prefix = next(
        (candidate for candidate in ("/files/", "/uploads/") if decoded_path.startswith(candidate)),
        None,
    )
    if prefix is None:
        return None

    filename = decoded_path[len(prefix):]
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or os.path.basename(filename) != filename
    ):
        return None
    return filename


def media_filename(result: dict[str, Any], asset: str) -> str | None:
    for section, field in MEDIA_ASSET_FIELDS.get(asset, ()):
        value = _field_value(result, section, field)
        if value:
            filename = _safe_generated_filename(value)
            if filename:
                return filename
    return None


def media_filenames(result: dict[str, Any]) -> set[str]:
    return {
        filename
        for asset in MEDIA_ASSETS
        if (filename := media_filename(result, asset)) is not None
    }


def resolve_media_path(
    upload_folder: str,
    result: dict[str, Any],
    asset: str,
) -> Path | None:
    filename = media_filename(result, asset)
    if not filename:
        return None

    folder = Path(upload_folder).resolve()
    path = (folder / filename).resolve()
    try:
        path.relative_to(folder)
    except ValueError:
        return None
    return path if path.is_file() else None


def write_analysis_access_record(
    upload_folder: str,
    analysis_id: str,
    physio_context: dict[str, Any] | None,
) -> None:
    """Write protection metadata before patient media becomes reachable."""
    context = physio_context if isinstance(physio_context, dict) else {}
    subject_id = context.get("subject_person_id")
    if not isinstance(subject_id, str) or not subject_id.strip():
        return

    folder = Path(upload_folder).resolve()
    access_path = folder / f"{analysis_id}_access.json"
    payload = {
        "analysis_id": analysis_id,
        "physio_context": {
            "subject_person_id": subject_id.strip(),
            "organization_id": context.get("organization_id"),
        },
    }
    access_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    access_path.chmod(0o600)


def _is_patient_linked(result: dict[str, Any]) -> bool:
    context = result.get("physio_context")
    context = context if isinstance(context, dict) else {}
    subject_id = context.get("subject_person_id")
    return isinstance(subject_id, str) and bool(subject_id.strip())


def classify_direct_file_access(upload_folder: str, filename: str) -> FileAccessDecision:
    """Protect result/media files that belong to a patient-linked analysis.

    Access sidecars are never web resources. New analyses are protected by the
    sidecar from the moment the original upload is written; result-file scans
    preserve protection for analyses created before sidecars existed.
    """
    if os.path.basename(filename) != filename or "/" in filename or "\\" in filename:
        return FileAccessDecision(internal=True, protected_result=None)
    if filename.endswith("_access.json"):
        return FileAccessDecision(internal=True, protected_result=None)

    folder = Path(upload_folder).resolve()
    for access_path in folder.glob("*_access.json"):
        access = _load_json(access_path)
        if not access or not _is_patient_linked(access):
            continue
        analysis_id = access.get("analysis_id")
        if (
            not isinstance(analysis_id, str)
            or not ANALYSIS_ID_PATTERN.fullmatch(analysis_id)
            or ".." in analysis_id
        ):
            continue
        owns_file = filename == f"{analysis_id}_result.json" or filename.startswith(
            f"{analysis_id}_"
        )
        if owns_file:
            result = load_analysis_result(upload_folder, analysis_id) or access
            return FileAccessDecision(internal=False, protected_result=result)

    for result_path in folder.glob("*_result.json"):
        result = _load_json(result_path)
        if not result or not _is_patient_linked(result):
            continue
        if filename == result_path.name or filename in media_filenames(result):
            return FileAccessDecision(internal=False, protected_result=result)

    return FileAccessDecision(internal=False, protected_result=None)
