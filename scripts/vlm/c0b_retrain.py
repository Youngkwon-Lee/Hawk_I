#!/usr/bin/env python3
"""Private, fail-closed PD4T gait C0B Qwen3-VL LoRA preparation and training.

This is a NEW research run, not an attestation of the 2026-08-17 archive.
All patient-level files, model artifacts and logs stay outside this repo.
The public training-binding status is intentionally never upgraded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from services.vlm_training_contract import (  # noqa: E402
    GAIT_ANCHOR, GAIT_GLOSSARY, GAIT_QUESTION, SYSTEM_ANCHOR,
)

SPLITS = ("train", "validation", "test")
VALIDATION_POLICIES = ("declared", "none_paper_protocol")
PROMPT = "\n\n".join((GAIT_ANCHOR, GAIT_GLOSSARY, GAIT_QUESTION, "<video>"))
MODEL = "Qwen/Qwen3-VL-4B-Instruct"


class PreflightError(ValueError):
    """Refuse to use an ambiguous or unsafe study input."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _private_path(path: Path) -> Path:
    if not path.is_absolute():
        raise PreflightError("Use absolute paths for private study inputs and outputs.")
    resolved = path.resolve()
    if _inside(resolved, ROOT):
        raise PreflightError("Private study data/output must not live in the Git checkout.")
    return resolved


def _resolve_video(raw: Any, media_root: Path) -> Path:
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise PreflightError("A non-empty video_path is required.")
    candidate = Path(raw)
    candidate = (candidate if candidate.is_absolute() else media_root / candidate).resolve()
    if not _inside(candidate, media_root):
        raise PreflightError("video_path escapes the declared media root.")
    if not candidate.is_file() or candidate.suffix.lower() != ".mp4" or candidate.stat().st_size <= 0:
        raise PreflightError("Expected a nonempty local .mp4 within the media root.")
    return candidate


def _load_rows(
    manifest: Path, media_root: Path, validation_policy: str = "declared",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    subjects: dict[str, str] = {}
    clips: set[str] = set()
    videos: set[str] = set()
    counts: Counter[str] = Counter()
    with manifest.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise PreflightError(f"Invalid JSONL line {line_no}.") from exc
            if not isinstance(row, dict):
                raise PreflightError(f"Line {line_no} is not an object.")
            if row.get("dataset") != "PD4T" or row.get("task") != "gait":
                raise PreflightError(f"Line {line_no} is not PD4T gait.")
            subject = row.get("subject_id")
            clip = row.get("clip_id")
            split = row.get("split")
            score = row.get("score")
            if not isinstance(subject, str) or not subject.strip() or len(subject) > 120:
                raise PreflightError(f"Line {line_no} has no valid explicit subject_id.")
            if not isinstance(clip, str) or not clip.strip() or len(clip) > 200:
                raise PreflightError(f"Line {line_no} has no valid clip_id.")
            if split not in SPLITS:
                raise PreflightError(f"Line {line_no} needs an explicit train/validation/test split.")
            if type(score) is not int or not 0 <= score <= 4:
                raise PreflightError(f"Line {line_no} needs an integer MDS-UPDRS gait score 0..4.")
            # No implicit subject assignment from a clip filename: file naming can be misleading.
            if subject in subjects and subjects[subject] != split:
                raise PreflightError("Subject leakage between splits.")
            if clip in clips:
                raise PreflightError("Duplicate clip_id in study manifest.")
            video = _resolve_video(row.get("video_path"), media_root)
            if str(video) in videos:
                raise PreflightError("Same media path appears in more than one row.")
            subjects[subject] = split
            clips.add(clip)
            videos.add(str(video))
            counts[split] += 1
            rows.append({"clip_id": clip, "subject_id": subject, "split": split, "score": score,
                         "video_path": str(video)})
    if validation_policy == "none_paper_protocol":
        # PD4T/PECoP protocol: all 22 official train subjects train, 8 test subjects stay sealed.
        if counts["validation"]:
            raise PreflightError("Paper protocol forbids validation rows; use the declared policy instead.")
        if counts["train"] == 0 or counts["test"] == 0:
            raise PreflightError("Train and sealed test must both be nonempty.")
    elif any(counts[s] == 0 for s in SPLITS):
        raise PreflightError("Train, validation and sealed test must all be nonempty.")
    return rows, {split: counts.get(split, 0) for split in SPLITS}


def _open_private(path: Path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(fd, "w", encoding="utf-8")


def _write_json(path: Path, content: Any) -> None:
    with _open_private(path) as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with _open_private(path) as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_sft_record(row: dict[str, Any]) -> dict[str, Any]:
    """C0B only: no clinician observations/kinematics, no GT in the user prompt."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_ANCHOR},
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": f"answer: {row['score']}"},
        ],
        "videos": [row["video_path"]],
    }


def prepare(
    *, manifest: Path, media_root: Path, run_root: Path,
    run_id: str, expected_sha256: str, validation_policy: str = "declared",
) -> Path:
    if validation_policy not in VALIDATION_POLICIES:
        raise PreflightError("Unknown validation policy.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}", run_id):
        raise PreflightError("Invalid run ID.")
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise PreflightError("An exact lowercase SHA-256 of the private manifest is required.")
    manifest = _private_path(manifest)
    media_root = _private_path(media_root)
    run_root = _private_path(run_root)
    if not media_root.is_dir() or not manifest.is_file():
        raise PreflightError("Private manifest/media root unavailable.")
    actual = sha256_file(manifest)
    if actual != expected_sha256:
        raise PreflightError("Frozen private manifest SHA-256 mismatch.")
    rows, counts = _load_rows(manifest, media_root, validation_policy)
    destination = run_root / run_id
    if destination.exists():
        raise PreflightError("The immutable run ID already exists; never overwrite runs.")

    # Hash media content, not just its filename; this does NOT attest independently
    # that the file was the source of the archived/previous model.
    media_binding = [
        {**row, "media_sha256": sha256_file(Path(row["video_path"]))}
        for row in rows
    ]
    training = [build_sft_record(row) for row in rows if row["split"] == "train"]
    validation = [build_sft_record(row) for row in rows if row["split"] == "validation"]
    if not run_root.exists():
        run_root.mkdir(parents=True, mode=0o700)
    if run_root.stat().st_mode & 0o077:
        raise PreflightError("Private run root must not be group/world accessible.")
    destination.mkdir(mode=0o700)
    try:
        _write_jsonl(destination / "train.jsonl", training)
        _write_jsonl(destination / "validation.jsonl", validation)
        _write_jsonl(destination / "private_media_binding.jsonl", media_binding)
        _write_json(destination / "run.json", {
            "schema": "hawkeye-c0b-qwen3vl-retrain-v1",
            "run_id": run_id,
            "dataset": "PD4T",
            "task": "gait",
            "condition": "C0B",
            "model_family": MODEL,
            "source_manifest_sha256": actual,
            "prompt_sha256": hashlib.sha256(
                (SYSTEM_ANCHOR + "\n" + PROMPT).encode("utf-8")
            ).hexdigest(),
            "source_media_binding_sha256": sha256_file(destination / "private_media_binding.jsonl"),
            "train_jsonl_sha256": sha256_file(destination / "train.jsonl"),
            "validation_jsonl_sha256": sha256_file(destination / "validation.jsonl"),
            "counts": counts,
            "subject_counts": {split: len({r["subject_id"] for r in rows if r["split"] == split})
                               for split in SPLITS},
            "split_check": "declared_subject_disjoint_only",
            "validation_policy": validation_policy,
            "held_out_verified": False,
            "training_binding_verified": False,
            "clinical_validity_verified": False,
            "status": "prepared_unverified",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    except BaseException:
        # Never silently overwrite partially created runs; inspect a failed run privately.
        raise
    return destination


def _verify_prepared(run_dir: Path) -> dict[str, Any]:
    manifest = run_dir / "run.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreflightError("Prepared private run.json is missing or invalid.") from exc
    if data.get("schema") != "hawkeye-c0b-qwen3vl-retrain-v1" or data.get("status") != "prepared_unverified":
        raise PreflightError("Not a prepared, unused C0B run.")
    expected_prompt_sha = hashlib.sha256(
        (SYSTEM_ANCHOR + "\n" + PROMPT).encode("utf-8")
    ).hexdigest()
    if data.get("prompt_sha256") != expected_prompt_sha or data.get("model_family") != MODEL:
        raise PreflightError("The frozen C0B prompt/model contract changed.")
    for filename, field in (
        ("train.jsonl", "train_jsonl_sha256"),
        ("validation.jsonl", "validation_jsonl_sha256"),
        ("private_media_binding.jsonl", "source_media_binding_sha256"),
    ):
        if sha256_file(run_dir / filename) != data[field]:
            raise PreflightError("Prepared data changed after freezing.")
    for line in (run_dir / "private_media_binding.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if sha256_file(Path(record["video_path"])) != record["media_sha256"]:
            raise PreflightError("Media bytes changed after freezing.")
    return data


def validate_approval(path: Path) -> str:
    """Owner-issued research permission; a placeholder/self-claim is not a grant."""
    approval = json.loads(_private_path(path).read_text(encoding="utf-8"))
    if not isinstance(approval, dict) or approval.get("dataset") != "PD4T":
        raise PreflightError("Approval must explicitly cover PD4T.")
    if approval.get("approved") is not True or approval.get("purpose") != "local_research_retraining":
        raise PreflightError("Approval does not authorize local research retraining.")
    for field in ("issued_by", "approval_reference", "approved_at"):
        if not isinstance(approval.get(field), str) or not approval[field].strip():
            raise PreflightError("A non-placeholder data-steward approval record is required.")
    return sha256_file(path)


def build_swift_command(
    run_dir: Path, model_dir: Path, output_dir: Path, *, with_validation: bool = True,
) -> list[str]:
    """Hardware-constrained Qwen3-VL QLoRA SFT pilot; never re-split patients.

    This FP16 4-bit 8-frame pilot is NOT equivalent to the archived August
    experiment or the upstream 2x21GiB BF16 LoRA reference.
    """
    evaluation = (
        ["--val_dataset", str(run_dir / "validation.jsonl")] if with_validation
        else ["--eval_strategy", "no"]
    )
    command = [
        "swift", "sft",
        "--model", str(model_dir),
        "--dataset", str(run_dir / "train.jsonl"),
        *evaluation,
        "--split_dataset_ratio", "0",
        "--tuner_type", "lora",
        "--quant_method", "bnb",
        "--quant_bits", "4",
        "--bnb_4bit_compute_dtype", "float16",
        "--bnb_4bit_quant_type", "nf4",
        "--bnb_4bit_use_double_quant", "true",
        "--torch_dtype", "float16",
        "--attn_impl", "sdpa",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--learning_rate", "0.0001",
        "--lora_rank", "8",
        "--lora_alpha", "32",
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--gradient_checkpointing", "true",
        "--gradient_accumulation_steps", "2",
        "--max_length", "3072",
        "--logging_steps", "5",
        "--save_steps", "100",
        "--save_total_limit", "2",
        "--output_dir", str(output_dir),
    ]
    if with_validation:
        command[command.index("--output_dir"):command.index("--output_dir")] = ["--eval_steps", "100"]
    return command


def verify_gpu_capacity(gpu_index: int) -> dict[str, int]:
    """Refuse training unless NVIDIA reports one idle GPU with memory headroom.

    GPU contention and ownership can change immediately after this check.
    The operator must also obtain resource-owner approval before --execute.
    """
    if gpu_index < 0:
        raise PreflightError("An authorized CUDA GPU index is required.")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightError("GPU memory inventory unavailable.") from exc
    if result.returncode != 0:
        raise PreflightError("GPU memory inventory unavailable.")
    rows: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        try:
            values = [int(part.strip()) for part in line.split(",")]
            if len(values) != 3:
                raise ValueError
            idx, total, free = values
            if idx < 0 or total <= 0 or not 0 <= free <= total or idx in rows:
                raise ValueError
            rows[idx] = (total, free)
        except ValueError as exc:
            raise PreflightError("Unexpected GPU capacity output.") from exc
    if gpu_index not in rows:
        raise PreflightError("Requested GPU index is missing from NVIDIA inventory.")
    total, free = rows[gpu_index]
    if total < 15_000 or free < 14_500:
        raise PreflightError(
            "ICT QLoRA pilot requires at least 15000 MiB total and 14500 MiB free GPU memory."
        )
    return {"gpu_index": gpu_index, "total_mib": total, "free_mib_at_check": free}


def train(
    *, run_dir: Path, model_dir: Path, approval_path: Path,
    gpu_index: int, execute: bool, compute_approved: bool = False,
) -> None:
    run_dir = _private_path(run_dir)
    model_dir = _private_path(model_dir)
    prepared = _verify_prepared(run_dir)
    with_validation = prepared.get("validation_policy", "declared") != "none_paper_protocol"
    if not execute:
        print("Preflight only: training not executed.")
        return
    if not compute_approved:
        raise PreflightError("The current GPU owner must authorize training compute.")
    if gpu_index < 0:
        raise PreflightError("An authorized CUDA GPU index is required.")
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise PreflightError("A preinstalled local Hugging Face model directory is required.")
    if not list(model_dir.glob("*.safetensors")):
        raise PreflightError("No local Hugging Face safetensors weights; GGUF is inference-only.")
    if shutil.which("swift") is None or shutil.which("nvidia-smi") is None:
        raise PreflightError("ms-swift or NVIDIA training environment unavailable.")
    approval_digest = validate_approval(approval_path)
    gpu_memory = verify_gpu_capacity(gpu_index)

    # Existing compute ownership and permissions must be established by the
    # caller; this check only detects obvious contention.
    result = subprocess.run(
        ["nvidia-smi", "--id", str(gpu_index), "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=15,
    )
    if result.returncode or any(line.strip() and line.strip() != "No running processes found"
                                for line in result.stdout.splitlines()):
        raise PreflightError("GPU compute ownership/availability not established.")

    weights = sorted(model_dir.glob("*.safetensors"))
    weight_hashes = [{"file": f.name, "sha256": sha256_file(f)} for f in weights]
    evidence = {"approval_sha256": approval_digest, "model_config_sha256":
                sha256_file(model_dir / "config.json"), "model_weights": weight_hashes,
                "status": "running_unverified",
                "hardware_check": gpu_memory,
                "training_recipe": "qwen3-vl-4b-bnb4-fp16-8frame-sdpa-v1"}
    _write_json(run_dir / "execution.json", evidence)  # x mode: cannot restart/overwrite
    command = build_swift_command(run_dir, model_dir, run_dir / "checkpoints",
                                  with_validation=with_validation)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_DISABLED": "true",
        "FPS_MAX_FRAMES": "8",
        "IMAGE_MAX_TOKEN_NUM": "512",
        "VIDEO_MAX_TOKEN_NUM": "128",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    # No video paths/IDs/logs leave the private study directory.
    with (run_dir / "train.stdout.log").open("x", encoding="utf-8") as stdout, (
        run_dir / "train.stderr.log"
    ).open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=str(run_dir), env=env,
                                   stdout=stdout, stderr=stderr, check=False)
    evidence["exit_code"] = completed.returncode
    checkpoints = sorted((run_dir / "checkpoints").rglob("adapter_model.safetensors"))
    if completed.returncode == 0 and checkpoints:
        evidence["status"] = "trained_unverified"
        evidence["adapter_digests"] = [
            {"sha256": sha256_file(f), "path_relative": str(f.relative_to(run_dir))}
            for f in checkpoints
        ]
    else:
        evidence["status"] = "failed_unverified"
    (run_dir / "execution.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if evidence["status"] != "trained_unverified":
        raise PreflightError("Training failed; inspect the private logs. No success claimed.")
    print("Local Qwen3-VL LoRA training completed; provenance and clinical validity UNVERIFIED.")


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze private source/media and prepare SFT JSONL")
    prep.add_argument("--manifest", required=True, type=Path)
    prep.add_argument("--media-root", required=True, type=Path)
    prep.add_argument("--run-root", required=True, type=Path)
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--expected-manifest-sha256", required=True)
    prep.add_argument("--validation-policy", choices=VALIDATION_POLICIES, default="declared",
                      help="none_paper_protocol: PD4T/PECoP 22-train/8-test protocol without validation")
    run = sub.add_parser("train", help="Run offline ms-swift on approved private CUDA hardware")
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--base-model-dir", required=True, type=Path)
    run.add_argument("--approval-json", required=True, type=Path)
    run.add_argument("--gpu-index", required=True, type=int)
    run.add_argument("--compute-approved", action="store_true",
                     help="Confirm assigned GPU owner explicitly authorized this training run")
    run.add_argument("--execute", action="store_true", help="Actually train; otherwise no work")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            destination = prepare(
                manifest=args.manifest, media_root=args.media_root,
                run_root=args.run_root, run_id=args.run_id,
                expected_sha256=args.expected_manifest_sha256,
                validation_policy=args.validation_policy,
            )
            result = json.loads((destination / "run.json").read_text(encoding="utf-8"))
            print(json.dumps({"status": result["status"], "counts": result["counts"],
                              "subject_counts": result["subject_counts"],
                              "training_binding_verified": False}))
        else:
            train(run_dir=args.run_dir, model_dir=args.base_model_dir,
                  approval_path=args.approval_json, gpu_index=args.gpu_index,
                  execute=args.execute, compute_approved=args.compute_approved)
    except (PreflightError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        # Never echo source identifiers, personal paths, or content in an error.
        message = str(exc) if isinstance(exc, PreflightError) else "private input or process error"
        print(f"preflight rejected: {type(exc).__name__}: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
