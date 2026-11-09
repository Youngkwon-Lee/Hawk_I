"""Synthetic-only validation of the private C0B Qwen3-VL training path."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from scripts.vlm import c0b_retrain as c0b  # noqa: E402
from services.vlm_training_contract import (  # noqa: E402
    SYSTEM_ANCHOR, GAIT_ANCHOR, GAIT_GLOSSARY, GAIT_QUESTION,
)


@pytest.fixture
def case(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    rows = []
    for index, (subject, split, score) in enumerate(
        (("T", "train", 0), ("V", "validation", 2), ("E", "test", 4))
    ):
        clip = f"synthetic_{index}"
        (media / f"{clip}.mp4").write_bytes(b"synthetic only" + bytes([index]))
        rows.append({
            "clip_id": clip, "subject_id": subject, "split": split,
            "dataset": "PD4T", "task": "gait", "score": score,
            "video_path": f"{clip}.mp4",
        })
    manifest = tmp_path / "study.jsonl"
    write_rows(manifest, rows)
    return {
        "rows": rows,
        "manifest": manifest,
        "media": media,
        "root": tmp_path / "private_runs",
        "out": tmp_path / "private_runs" / "synthetic-20261109",
    }


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def prepare_case(case):
    return c0b.prepare(
        manifest=case["manifest"], media_root=case["media"], run_root=case["root"],
        run_id="synthetic-20261109",
        expected_sha256=c0b.sha256_file(case["manifest"]),
    )


def test_c0b_uses_existing_prompt_contract_without_target_in_question(case):
    row = {**case["rows"][0], "video_path": str(case["media"] / "synthetic_0.mp4")}
    sft = c0b.build_sft_record(row)
    assert sft["messages"][0] == {"role": "system", "content": SYSTEM_ANCHOR}
    content = sft["messages"][1]["content"]
    assert content.index(GAIT_ANCHOR) < content.index(GAIT_GLOSSARY) < content.index(GAIT_QUESTION)
    assert content.count("<video>") == 1
    assert "answer: 0" not in content
    assert sft["messages"][2]["content"] == "answer: 0"
    assert sft["videos"] == [row["video_path"]]


def test_prepare_freezes_data_and_never_gives_test_to_trainer(case):
    target = prepare_case(case)
    summary = json.loads((target / "run.json").read_text())
    assert summary["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert summary["subject_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert summary["training_binding_verified"] is False
    assert summary["held_out_verified"] is False
    assert summary["status"] == "prepared_unverified"
    assert (target / "train.jsonl").read_text().count("\n") == 1
    assert (target / "validation.jsonl").read_text().count("\n") == 1
    train = json.loads((target / "train.jsonl").read_text().strip())
    assert "synthetic_2" not in json.dumps(train)
    assert len((target / "private_media_binding.jsonl").read_text().splitlines()) == 3
    assert c0b._verify_prepared(target)["source_manifest_sha256"] == c0b.sha256_file(case["manifest"])
    assert target.stat().st_mode & 0o077 == 0
    assert (target / "train.jsonl").stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("bad_split", ["training", "val", None])
def test_unknown_split_rejected(case, bad_split):
    case["rows"][0]["split"] = bad_split
    write_rows(case["manifest"], case["rows"])
    with pytest.raises(c0b.PreflightError, match="explicit"):
        prepare_case(case)


def test_subject_overlap_rejected(case):
    case["rows"][1]["subject_id"] = case["rows"][0]["subject_id"]
    write_rows(case["manifest"], case["rows"])
    with pytest.raises(c0b.PreflightError, match="leakage"):
        prepare_case(case)


def test_duplicate_clip_rejected(case):
    case["rows"][1]["clip_id"] = case["rows"][0]["clip_id"]
    write_rows(case["manifest"], case["rows"])
    with pytest.raises(c0b.PreflightError, match="Duplicate"):
        prepare_case(case)


def test_duplicate_media_rejected(case):
    case["rows"][1]["video_path"] = case["rows"][0]["video_path"]
    write_rows(case["manifest"], case["rows"])
    with pytest.raises(c0b.PreflightError, match="media path"):
        prepare_case(case)


def test_out_of_root_media_rejected(case):
    (case["media"].parent / "outside.mp4").write_bytes(b"bad")
    case["rows"][0]["video_path"] = "../outside.mp4"
    write_rows(case["manifest"], case["rows"])
    with pytest.raises(c0b.PreflightError, match="escapes"):
        prepare_case(case)


def test_score_rejected_if_bool_or_string(case):
    for score in (True, "2", -1, 5):
        case["rows"][0]["score"] = score
        write_rows(case["manifest"], case["rows"])
        with pytest.raises(c0b.PreflightError, match="integer"):
            prepare_case(case)


def test_missing_test_rejected(case):
    write_rows(case["manifest"], case["rows"][:2])
    with pytest.raises(c0b.PreflightError, match="nonempty"):
        prepare_case(case)


def test_manifest_hash_must_be_correct(case):
    with pytest.raises(c0b.PreflightError, match="mismatch"):
        c0b.prepare(manifest=case["manifest"], media_root=case["media"],
                    run_root=case["root"], run_id="synthetic-20261109",
                    expected_sha256="0" * 64)


def test_source_media_mutation_fails_sealed_check(case):
    target = prepare_case(case)
    (case["media"] / "synthetic_0.mp4").write_bytes(b"changed")
    with pytest.raises(c0b.PreflightError, match="Media bytes changed"):
        c0b._verify_prepared(target)


def test_training_data_mutation_fails_sealed_check(case):
    target = prepare_case(case)
    with (target / "train.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(c0b.PreflightError, match="Prepared data changed"):
        c0b._verify_prepared(target)


def test_same_run_id_is_not_reused(case):
    prepare_case(case)
    with pytest.raises(c0b.PreflightError, match="already exists"):
        prepare_case(case)


def test_no_private_output_inside_git_checkout(case):
    with pytest.raises(c0b.PreflightError, match="Git checkout"):
        c0b.prepare(manifest=case["manifest"], media_root=case["media"],
                    run_root=ROOT / "experiments" / "results",
                    run_id="synthetic-20261109",
                    expected_sha256=c0b.sha256_file(case["manifest"]))


def test_invalid_approval_is_not_sufficient(case):
    receipt = case["media"].parent / "owner-approval.json"
    receipt.write_text('{"dataset":"PD4T","approved":true,"purpose":"local_research_retraining"}')
    with pytest.raises(c0b.PreflightError, match="approval"):
        c0b.validate_approval(receipt)


def test_command_reuses_frozen_train_val_only_and_never_autosplits(case):
    command = c0b.build_swift_command(case["out"], case["media"], case["root"] / "output")
    assert command[:2] == ["swift", "sft"]
    assert command[command.index("--split_dataset_ratio") + 1] == "0"
    assert command[command.index("--tuner_type") + 1] == "lora"
    assert command[command.index("--freeze_vit") + 1] == "true"
    assert command[command.index("--model") + 1] == str(case["media"])
    assert not any("test.jsonl" in x for x in command)


def test_training_default_does_not_run_expensive_gpu_work(case, capsys):
    target = prepare_case(case)
    c0b.train(run_dir=target, model_dir=case["media"],
              approval_path=case["media"] / "unapproved.json",
              gpu_index=0, execute=False)
    assert "not executed" in capsys.readouterr().out
    assert not (target / "execution.json").exists()


def test_prompt_contract_mutation_detected(case, monkeypatch):
    target = prepare_case(case)
    monkeypatch.setattr(c0b, "PROMPT", "changed")
    with pytest.raises(c0b.PreflightError, match="prompt/model"):
        c0b._verify_prepared(target)


def test_execute_requires_separate_compute_owner_approval(case):
    target = prepare_case(case)
    with pytest.raises(c0b.PreflightError, match="GPU owner"):
        c0b.train(run_dir=target, model_dir=case["media"],
                  approval_path=case["media"] / "unapproved.json",
                  gpu_index=0, execute=True, compute_approved=False)
    assert not (target / "execution.json").exists()
