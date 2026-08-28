#!/usr/bin/env python3
"""Train or evaluate the provenance-bound Qwen3-VL C0B clinician candidate.

The default path evaluates validation only. Test generation requires the
explicit ``--unlock-test`` flag so model selection cannot accidentally inspect
the held-out split.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from services.c0b_training_data import (  # noqa: E402
    C0BDataError,
    load_jsonl,
    prompt_contract_sha256,
    sha256_file,
    verify_training_stage,
)
from services.c0b_training_eval import parse_c0b_answer, score_predictions  # noqa: E402
from services.vlm_training_contract import build_c0b_messages  # noqa: E402


LOGGER = logging.getLogger("hawkeye.c0b.train")
BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "decord", "opencv-python-headless")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StagedGaitDataset:
    def __init__(self, data_dir: Path, split: str, *, fps: float, frame_width: int):
        self.data_dir = data_dir
        self.rows = load_jsonl(data_dir / f"{split}.jsonl")
        self.fps = fps
        self.frame_width = frame_width

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from decord import VideoReader

        row = self.rows[index]
        video_path = self.data_dir / row["media_path"]
        reader = VideoReader(str(video_path))
        native_fps = float(reader.get_avg_fps())
        duration = len(reader) / native_fps
        n_frames = max(4, int(round(duration * self.fps)))
        n_frames -= n_frames % 2
        frame_indices = np.linspace(0, len(reader) - 1, n_frames).round().astype(int)
        frames = reader.get_batch(frame_indices).asnumpy()

        import cv2

        height, width = frames.shape[1:3]
        new_width = self.frame_width - self.frame_width % 32
        new_height = max(32, int(round(height * new_width / width)) // 32 * 32)
        frames = np.stack([cv2.resize(frame, (new_width, new_height)) for frame in frames])
        return {
            **row,
            "frames": list(frames),
            "duration": duration,
            "native_fps": native_fps,
            "frame_indices": frame_indices.tolist(),
        }


class C0BCollator:
    def __init__(self, processor: Any, *, max_length: int):
        self.processor = processor
        self.max_length = max_length
        self.pad_id = processor.tokenizer.pad_token_id
        self.processor.video_processor.do_sample_frames = False

    @staticmethod
    def messages_with_video() -> list[dict[str, Any]]:
        messages = build_c0b_messages()
        messages[1]["content"].append({"type": "video"})
        return messages

    @staticmethod
    def metadata(sample: dict[str, Any]) -> list[dict[str, Any]]:
        return [{
            "fps": sample["native_fps"],
            "duration": sample["duration"],
            "total_num_frames": int(round(sample["duration"] * sample["native_fps"])),
            "frames_indices": sample["frame_indices"],
        }]

    def encode_prompt(self, sample: dict[str, Any]) -> dict[str, Any]:
        prompt = self.processor.apply_chat_template(
            self.messages_with_video(), tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            text=[prompt],
            videos=[sample["frames"]],
            video_metadata=self.metadata(sample),
            cap_pixels_per_frame=True,
            return_tensors="pt",
        )

    def _encode_training_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        prompt = self.processor.apply_chat_template(
            self.messages_with_video(), tokenize=False, add_generation_prompt=True
        )
        target = f"answer: {sample['score']}{self.processor.tokenizer.eos_token}"
        common = {
            "videos": [sample["frames"]],
            "video_metadata": self.metadata(sample),
            "cap_pixels_per_frame": True,
            "return_tensors": "pt",
        }
        prompt_inputs = self.processor(text=[prompt], **common)
        full_inputs = self.processor(text=[prompt + target], **common)
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100
        full_inputs["labels"] = labels
        return full_inputs

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if len(samples) != 1:
            raise ValueError("Qwen3 C0B training requires per-device batch size 1")
        encoded = self._encode_training_sample(samples[0])
        width = int(encoded["input_ids"].shape[1])
        if width > self.max_length:
            raise ValueError(
                f"encoded length {width} exceeds max_length {self.max_length}; "
                "lower fps/frame width instead of truncating video tokens"
            )
        return encoded


def load_transformers_stack() -> dict[str, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments

    try:
        from transformers import AutoModelForMultimodalLM as AutoVLM
    except ImportError:  # Transformers versions used by the original 2026-08 bundle.
        from transformers import AutoModelForImageTextToText as AutoVLM
    return locals()


def load_base_and_processor(args: argparse.Namespace, stack: dict[str, Any], *, training: bool):
    torch = stack["torch"]
    processor = stack["AutoProcessor"].from_pretrained(
        args.base_model, revision=args.base_revision, trust_remote_code=True
    )
    processor.video_processor.do_sample_frames = False
    model_kwargs: dict[str, Any] = {
        "revision": args.base_revision,
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    if training and args.quantization == "4bit":
        model_kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif not training:
        model_kwargs["device_map"] = {"": 0}
    model = stack["AutoVLM"].from_pretrained(args.base_model, **model_kwargs)
    return model, processor


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolved_model_revision(model: Any, fallback: str) -> str:
    return str(getattr(model.config, "_commit_hash", None) or fallback)


def evaluate(
    *,
    model: Any,
    processor: Any,
    data_dir: Path,
    run_dir: Path,
    split: str,
    fps: float,
    frame_width: int,
    max_length: int,
    max_new_tokens: int,
    dataset_sha256: str,
) -> dict[str, Any]:
    import torch

    dataset = StagedGaitDataset(data_dir, split, fps=fps, frame_width=frame_width)
    collator = C0BCollator(processor, max_length=max_length)
    truth: list[int] = []
    parsed: list[int | None] = []
    rows: list[dict[str, Any]] = []
    model.eval()
    for index in range(len(dataset)):
        sample = dataset[index]
        inputs = collator.encode_prompt(sample)
        if int(inputs["input_ids"].shape[1]) > max_length:
            raise ValueError(f"validation sample {index} exceeds max_length")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        text = processor.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        prediction = parse_c0b_answer(text)
        truth.append(int(sample["score"]))
        parsed.append(prediction)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "score": int(sample["score"]),
                "prediction": prediction,
                "response": text,
            }
        )
        if (index + 1) % 10 == 0:
            LOGGER.info("evaluation %s: %d/%d", split, index + 1, len(dataset))

    predictions_path = run_dir / f"predictions-{split}.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = score_predictions(truth, parsed, fallback=0)
    payload = {
        "schema_version": "hawkeye.c0b-evaluation.v1",
        "created_at": utc_now(),
        "split": split,
        "source_dataset_sha256": dataset_sha256,
        "prompt_contract_sha256": prompt_contract_sha256(),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics": metrics,
    }
    write_json(run_dir / f"metrics-{split}.json", payload)
    return payload


def train(args: argparse.Namespace, stage_manifest: dict[str, Any]) -> int:
    stack = load_transformers_stack()
    set_reproducible_seed(args.seed)
    model, processor = load_base_and_processor(args, stack, training=True)
    if args.quantization == "4bit":
        model = stack["prepare_model_for_kbit_training"](
            model, use_gradient_checkpointing=True
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora = stack["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        exclude_modules=r".*visual.*",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = stack["get_peft_model"](model, lora)
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = {
        "schema_version": "hawkeye.c0b-training-run.v1",
        "status": "running",
        "started_at": utc_now(),
        "candidate_name": args.candidate_name,
        "source_dataset_sha256": stage_manifest["source_dataset_sha256"],
        "stage_sha256": stage_manifest["stage_sha256"],
        "prompt_contract_sha256": prompt_contract_sha256(),
        "code_git_revision": git_revision(),
        "base_model": args.base_model,
        "requested_base_revision": args.base_revision,
        "resolved_base_revision": resolved_model_revision(model, args.base_revision),
        "seed": args.seed,
        "data": {
            "fps": args.fps,
            "frame_width": args.frame_width,
            "processor_do_sample_frames": False,
            "cap_pixels_per_frame": True,
            "max_length": args.max_length,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": TARGET_MODULES,
            "exclude_modules": ".*visual.*",
            "quantization": args.quantization,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "per_device_batch_size": 1,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
        },
        "versions": package_versions(),
    }
    write_json(args.output_dir / "training-run-manifest.json", started)

    train_dataset = StagedGaitDataset(args.data_dir, "train", fps=args.fps, frame_width=args.frame_width)
    validation_dataset = StagedGaitDataset(
        args.data_dir, "validation", fps=args.fps, frame_width=args.frame_width
    )
    training_args = stack["TrainingArguments"](
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = stack["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=C0BCollator(processor, max_length=args.max_length),
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir / "processor"))
    model.config.use_cache = True

    validation = evaluate(
        model=model,
        processor=processor,
        data_dir=args.data_dir,
        run_dir=args.output_dir,
        split="validation",
        fps=args.fps,
        frame_width=args.frame_width,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        dataset_sha256=stage_manifest["source_dataset_sha256"],
    )
    adapter_path = args.output_dir / "adapter_model.safetensors"
    started.update(
        {
            "status": "candidate",
            "finished_at": utc_now(),
            "train_metrics": result.metrics,
            "validation_metrics": validation["metrics"],
            "adapter_sha256": sha256_file(adapter_path) if adapter_path.is_file() else None,
            "promotion": {
                "status": "not_promoted",
                "reason": "A candidate must be reviewed before test evaluation or serving binding.",
            },
        }
    )
    write_json(args.output_dir / "training-run-manifest.json", started)
    LOGGER.info("candidate complete: %s", args.output_dir)
    return 0


def smoke(args: argparse.Namespace, stage_manifest: dict[str, Any]) -> int:
    """Run one real video through the exact training collator and model loss."""
    import torch

    stack = load_transformers_stack()
    set_reproducible_seed(args.seed)
    model, processor = load_base_and_processor(args, stack, training=True)
    if args.quantization == "4bit":
        model = stack["prepare_model_for_kbit_training"](
            model, use_gradient_checkpointing=True
        )
    lora = stack["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        exclude_modules=r".*visual.*",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = stack["get_peft_model"](model, lora)
    dataset = StagedGaitDataset(args.data_dir, "train", fps=args.fps, frame_width=args.frame_width)
    sample = dataset[0]
    batch = C0BCollator(processor, max_length=args.max_length)([sample]).to(model.device)
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    loss = float(output.loss.detach().cpu())
    if not np.isfinite(loss):
        raise RuntimeError("smoke forward produced a non-finite loss")
    payload = {
        "valid": True,
        "source_dataset_sha256": stage_manifest["source_dataset_sha256"],
        "prompt_contract_sha256": prompt_contract_sha256(),
        "base_model": args.base_model,
        "resolved_base_revision": resolved_model_revision(model, args.base_revision),
        "quantization": args.quantization,
        "encoded_tokens": int(batch["input_ids"].shape[1]),
        "sampled_frames": len(sample["frames"]),
        "cap_pixels_per_frame": True,
        "loss": loss,
        "gpu": torch.cuda.get_device_name(0),
        "versions": package_versions(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def evaluate_existing(args: argparse.Namespace, stage_manifest: dict[str, Any]) -> int:
    if args.split == "test" and not args.unlock_test:
        raise SystemExit("Refusing test evaluation without --unlock-test after model selection is frozen.")
    stack = load_transformers_stack()
    model, processor = load_base_and_processor(args, stack, training=False)
    model = stack["PeftModel"].from_pretrained(model, str(args.run_dir)).eval()
    payload = evaluate(
        model=model,
        processor=processor,
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        split=args.split,
        fps=args.fps,
        frame_width=args.frame_width,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        dataset_sha256=stage_manifest["source_dataset_sha256"],
    )
    print(json.dumps(payload["metrics"], indent=2))
    return 0


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--frame-width", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument("--max-new-tokens", type=int, default=16)


def lora_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantization", choices=("4bit", "bf16"), default="4bit")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    common_arguments(validate_parser)

    smoke_parser = subparsers.add_parser("smoke")
    common_arguments(smoke_parser)
    lora_arguments(smoke_parser)

    train_parser = subparsers.add_parser("train")
    common_arguments(train_parser)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--candidate-name", default="hawkeye-c0b-clinician-v2-seed42")
    lora_arguments(train_parser)
    train_parser.add_argument("--epochs", type=float, default=3.0)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.05)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--resume-from-checkpoint", default="")

    evaluate_parser = subparsers.add_parser("evaluate")
    common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate_parser.add_argument("--unlock-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        stage_manifest = verify_training_stage(
            args.data_dir,
            expected_dataset_sha256=args.expected_dataset_sha256,
            require_media=True,
        )
    except C0BDataError as exc:
        LOGGER.error("training stage rejected: %s", exc)
        return 1
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "valid": True,
                    "source_dataset_sha256": stage_manifest["source_dataset_sha256"],
                    "stage_sha256": stage_manifest["stage_sha256"],
                    "prompt_contract_sha256": stage_manifest["prompt_contract_sha256"],
                    "splits": {
                        name: details["records"] for name, details in stage_manifest["splits"].items()
                    },
                },
                indent=2,
            )
        )
        return 0
    if args.command == "smoke":
        return smoke(args, stage_manifest)
    if args.command == "train":
        return train(args, stage_manifest)
    return evaluate_existing(args, stage_manifest)


if __name__ == "__main__":
    raise SystemExit(main())
