import json

from services.handoff_manifest import (
    SCHEMA_VERSION,
    build_handoff_manifest,
    verify_handoff_manifest,
    write_handoff_manifest,
)


def _write_split(root, name, records):
    path = root / f"{name}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_manifest_is_hash_addressed_and_verifiable(tmp_path):
    _write_split(tmp_path, "train", [{"clip_id": "a"}, {"clip_id": "b"}])
    _write_split(tmp_path, "validation", [{"clip_id": "c"}])
    summary = {
        "train": {"clips": 2, "patients": 1},
        "validation": {"clips": 1, "patients": 1},
    }

    manifest = build_handoff_manifest(tmp_path, summary, task="gait")
    path = write_handoff_manifest(tmp_path, manifest)
    verified = verify_handoff_manifest(path)

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["handoff_id"].endswith(manifest["dataset_sha256"][:12])
    assert verified["valid"] is True
    assert verified["splits"]["train"]["records"] == 2
    assert verified["model_binding"]["status"] == "unverified"


def test_manifest_detects_export_tampering(tmp_path):
    _write_split(tmp_path, "train", [{"clip_id": "a"}])
    manifest = build_handoff_manifest(
        tmp_path,
        {"train": {"clips": 1, "patients": 1}},
        task="gait",
    )
    path = write_handoff_manifest(tmp_path, manifest)
    with (tmp_path / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"clip_id":"changed"}\n')

    verified = verify_handoff_manifest(path)

    assert verified["valid"] is False
    assert "train: record count mismatch" in verified["errors"]
    assert "train: sha256 mismatch" in verified["errors"]


def test_manifest_rejects_summary_count_mismatch(tmp_path):
    _write_split(tmp_path, "train", [{"clip_id": "a"}])

    try:
        build_handoff_manifest(
            tmp_path,
            {"train": {"clips": 2, "patients": 1}},
            task="gait",
        )
    except ValueError as exc:
        assert "summary expects 2" in str(exc)
    else:
        raise AssertionError("count mismatch should fail")

