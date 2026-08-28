import json
from pathlib import Path

import pytest

from services.c0b_training_data import (
    C0BDataError,
    prompt_contract_sha256,
    resolve_media_path,
    stage_handoff,
    verify_training_stage,
)
from services.handoff_manifest import build_handoff_manifest, write_handoff_manifest


def _write_export(root: Path, *, overlap: bool = False) -> str:
    summary = {}
    patients = {"train": "p1", "validation": "p1" if overlap else "p2", "test": "p3"}
    for index, split in enumerate(("train", "validation", "test")):
        media = root / f"source-{split}.mp4"
        media.write_bytes(f"video-{split}".encode())
        row = {
            "clip_id": f"clip-{split}",
            "patient_id": patients[split],
            "split": split,
            "task": "gait",
            "media_path": str(media),
            "updrs_3_10": index,
        }
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        summary[split] = {"clips": 1, "patients": 1}
    manifest = build_handoff_manifest(root, summary, task="gait")
    write_handoff_manifest(root, manifest)
    return manifest["dataset_sha256"]


def test_windows_media_path_resolves_for_wsl():
    assert resolve_media_path("D:/PD4T/gait/a.mp4") == Path("/mnt/d/PD4T/gait/a.mp4")


def test_stage_removes_source_identifiers_and_verifies(tmp_path: Path):
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    source.mkdir()
    dataset_sha = _write_export(source)

    manifest = stage_handoff(
        source,
        staged,
        expected_dataset_sha256=dataset_sha,
    )
    verified = verify_training_stage(staged, expected_dataset_sha256=dataset_sha)

    assert manifest == verified
    assert manifest["prompt_contract_sha256"] == prompt_contract_sha256()
    row = json.loads((staged / "train.jsonl").read_text())
    assert set(row) == {
        "sample_id",
        "split",
        "task",
        "score",
        "media_path",
        "media_sha256",
        "media_bytes",
    }
    assert "clip-train" not in (staged / "train.jsonl").read_text()
    assert "p1" not in (staged / "train.jsonl").read_text()
    assert (staged / row["media_path"]).read_bytes() == b"video-train"


def test_stage_refuses_patient_overlap(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_export(source, overlap=True)

    with pytest.raises(C0BDataError, match="patient overlap"):
        stage_handoff(source, tmp_path / "staged")


def test_stage_refuses_wrong_dataset_sha(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _write_export(source)

    with pytest.raises(C0BDataError, match="dataset SHA mismatch"):
        stage_handoff(source, tmp_path / "staged", expected_dataset_sha256="0" * 64)


def test_stage_digest_detects_split_tampering(tmp_path: Path):
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    source.mkdir()
    _write_export(source)
    stage_handoff(source, staged)
    with (staged / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(C0BDataError, match="split digest mismatch"):
        verify_training_stage(staged)


def test_stage_is_idempotent_only_for_a_verified_existing_stage(tmp_path: Path):
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    source.mkdir()
    dataset_sha = _write_export(source)
    first = stage_handoff(source, staged, expected_dataset_sha256=dataset_sha)
    second = stage_handoff(source, staged, expected_dataset_sha256=dataset_sha)
    assert first == second

    unverified = tmp_path / "unverified"
    unverified.mkdir()
    (unverified / "stale.txt").write_text("old")
    with pytest.raises(C0BDataError, match="not empty"):
        stage_handoff(source, unverified, expected_dataset_sha256=dataset_sha)
