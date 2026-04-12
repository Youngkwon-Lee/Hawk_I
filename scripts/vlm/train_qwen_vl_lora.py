"""
Qwen2.5-VL LoRA training entry skeleton for MotorExam-VQA.

This script currently focuses on:
- config loading
- dataset loading / conversion to SFT records
- task filtering
- dry-run validation
- optional model + LoRA bootstrap
- trainer preparation skeleton

It is intentionally conservative: use `--dry-run` first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

try:
    from data.export_motorexam_qwen_sft import build_sft_record, load_jsonl as load_raw_jsonl
except ModuleNotFoundError:
    from scripts.data.export_motorexam_qwen_sft import build_sft_record, load_jsonl as load_raw_jsonl

DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "vlm" / "qwen_vl_lora_motorexam_v0_1.yaml"
TEMP_FRAME_DIR = Path(tempfile.gettempdir()) / "hawkeye_qwen_lora_train_frames"
TEMP_FRAME_DIR.mkdir(parents=True, exist_ok=True)

TASK_PHASE_POSITIONS = {
    "gait": [
        0.00, 0.06, 0.12, 0.20, 0.28, 0.36, 0.44, 0.52,
        0.60, 0.68, 0.76, 0.84, 0.90, 0.94, 0.98,
    ],
    "finger_tapping": [
        0.00, 0.03, 0.06, 0.09, 0.14, 0.20, 0.28, 0.36,
        0.44, 0.52, 0.60, 0.68, 0.76, 0.84, 0.90, 0.95, 0.99,
    ],
    "hand_movement": [
        0.00, 0.03, 0.06, 0.10, 0.16, 0.24, 0.32, 0.40,
        0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.94, 0.98,
    ],
    "leg_agility": [
        0.00, 0.03, 0.06, 0.10, 0.16, 0.24, 0.32, 0.40,
        0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.94, 0.98,
    ],
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset only")
    parser.add_argument("--limit-train", type=int, default=0, help="Optional cap on train examples for debugging")
    parser.add_argument("--limit-val", type=int, default=0, help="Optional cap on val examples for debugging")
    parser.add_argument(
        "--include-rationale",
        action="store_true",
        help="Include notes/rationale when converting full records to SFT format",
    )
    parser.add_argument(
        "--bootstrap-model",
        action="store_true",
        help="Try to load the base model and attach LoRA adapters without starting training",
    )
    parser.add_argument(
        "--prepare-trainer",
        action="store_true",
        help="Prepare a minimal trainer skeleton after bootstrap without calling train()",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the actual trainer loop (recommended only after bootstrap works on a GPU machine).",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def is_sft_record(record: dict[str, Any]) -> bool:
    return "messages" in record


def maybe_limit(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit and limit > 0:
        return rows[:limit]
    return rows


def maybe_filter_tasks(rows: list[dict[str, Any]], tasks: list[str]) -> list[dict[str, Any]]:
    if not tasks:
        return rows

    filtered: list[dict[str, Any]] = []
    for row in rows:
        task = row.get("task")
        if task is None and "metadata" in row:
            task = row.get("task")
        if task in tasks:
            filtered.append(row)
    return filtered


def maybe_filter_question_types(rows: list[dict[str, Any]], question_types: list[str]) -> list[dict[str, Any]]:
    if not question_types:
        return rows

    allowed = set(question_types)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata", {})
        question_type = metadata.get("question_type", "unknown")
        if question_type in allowed:
            filtered.append(row)
    return filtered


def convert_records_to_sft(rows: list[dict[str, Any]], include_rationale: bool) -> list[dict[str, Any]]:
    if not rows:
        return []
    if is_sft_record(rows[0]):
        return rows
    return [build_sft_record(row, include_rationale=include_rationale) for row in rows]


def summarize_split(name: str, rows: list[dict[str, Any]]) -> None:
    task_counter = Counter()
    qtype_counter = Counter()
    for row in rows:
        task_counter[row.get("task", "unknown")] += 1
        metadata = row.get("metadata", {})
        qtype_counter[metadata.get("question_type", "unknown")] += 1

    logger.info("%s examples: %d", name, len(rows))
    logger.info("%s tasks: %s", name, dict(task_counter))
    logger.info("%s question types: %s", name, dict(qtype_counter))


def load_split(
    path: Path,
    *,
    tasks: list[str],
    question_types: list[str],
    include_rationale: bool,
    limit: int,
) -> list[dict[str, Any]]:
    rows = load_raw_jsonl(path)
    rows = maybe_filter_tasks(rows, tasks)
    rows = convert_records_to_sft(rows, include_rationale=include_rationale)
    rows = maybe_filter_question_types(rows, question_types)
    rows = maybe_limit(rows, limit)
    return rows


def count_trainable_params(model: Any) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return trainable, total


def uniform_indices(total_frames: int, n_frames: int) -> list[int]:
    if n_frames <= 0:
        return []
    if total_frames <= 1:
        return [0]
    return sorted({int(i * (total_frames - 1) / max(n_frames - 1, 1)) for i in range(n_frames)})


def task_aware_indices(total_frames: int, n_frames: int, task: str) -> list[int]:
    if n_frames <= 0:
        return []
    if total_frames <= 1:
        return [0]

    positions = TASK_PHASE_POSITIONS.get(task)
    if not positions:
        return uniform_indices(total_frames, n_frames)

    anchor_indices = sorted({int(pos * (total_frames - 1)) for pos in positions})
    if len(anchor_indices) > n_frames:
        reduced = []
        for idx in uniform_indices(len(anchor_indices), n_frames):
            reduced.append(anchor_indices[idx])
        return sorted(set(reduced))

    selected = set(anchor_indices)
    if len(selected) < n_frames:
        for idx in uniform_indices(total_frames, n_frames):
            selected.add(idx)
            if len(selected) >= n_frames:
                break
    return sorted(selected)


def sample_video_to_frame_paths(video_path: Path, *, task: str, frame_budget: int, frame_width: int) -> list[str]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("opencv-python-headless is required for training-time frame sampling") from exc

    digest = hashlib.md5(f"{video_path}|{task}|{frame_budget}|{frame_width}".encode("utf-8")).hexdigest()[:12]
    cache_dir = TEMP_FRAME_DIR / digest
    if cache_dir.exists():
        cached = sorted(cache_dir.glob("*.jpg"))
        if cached:
            return [str(path) for path in cached]

    cache_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for training: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames found in video: {video_path}")

    indices = task_aware_indices(total_frames=total_frames, n_frames=frame_budget, task=task)
    saved_paths: list[str] = []
    for order, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        if frame_width > 0 and frame.shape[1] > frame_width:
            scale = frame_width / frame.shape[1]
            resized_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (frame_width, resized_height), interpolation=cv2.INTER_AREA)
        out_path = cache_dir / f"{order:03d}_{frame_index}.jpg"
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        out_path.write_bytes(buffer.tobytes())
        saved_paths.append(str(out_path))

    cap.release()

    if not saved_paths:
        raise RuntimeError(f"Failed to sample frames for training: {video_path}")
    return saved_paths


class MotorExamQwenDataCollator:
    def __init__(self, processor: Any, config: dict[str, Any]):
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Pillow is required for training-time fallback images") from exc

        self.Image = Image
        self.ImageDraw = ImageDraw
        self.processor = processor
        sampling_policy = config["data"].get("sampling_policy", {})
        self.frame_width = 512
        self.frame_budget_by_task = {k: int(v) for k, v in sampling_policy.items()}
        self.allow_missing_videos = bool(config["data"].get("allow_missing_videos_as_blank_images", False))

    def _placeholder_frames(self, *, task: str, frame_budget: int) -> list[str]:
        digest = hashlib.md5(f"placeholder|{task}|{frame_budget}".encode("utf-8")).hexdigest()[:12]
        cache_dir = TEMP_FRAME_DIR / f"placeholder_{digest}"
        if cache_dir.exists():
            cached = sorted(cache_dir.glob("*.jpg"))
            if cached:
                return [str(path) for path in cached]

        cache_dir.mkdir(parents=True, exist_ok=True)
        width = self.frame_width
        height = max(256, int(width * 0.75))
        saved: list[str] = []
        for i in range(frame_budget):
            image = self.Image.new("RGB", (width, height), color=(18, 24, 34))
            draw = self.ImageDraw.Draw(image)
            draw.rectangle((16, 16, width - 16, height - 16), outline=(90, 130, 180), width=3)
            draw.text((32, 40), f"Hawkeye toy placeholder", fill=(235, 240, 245))
            draw.text((32, 80), f"task={task}", fill=(235, 240, 245))
            draw.text((32, 120), f"frame={i+1}/{frame_budget}", fill=(235, 240, 245))
            out_path = cache_dir / f"{i:03d}.jpg"
            image.save(out_path, format="JPEG", quality=90)
            saved.append(str(out_path))
        return saved

    def _expand_user_content(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        task = row["task"]
        frame_budget = int(self.frame_budget_by_task.get(task, 8))
        expanded: list[dict[str, Any]] = []
        for item in row["messages"][0]["content"]:
            if item.get("type") == "text":
                expanded.append(item)
            elif item.get("type") == "video":
                video_path = ROOT / item["video"]
                if video_path.exists():
                    frame_paths = sample_video_to_frame_paths(
                        video_path,
                        task=task,
                        frame_budget=frame_budget,
                        frame_width=self.frame_width,
                    )
                elif self.allow_missing_videos:
                    frame_paths = self._placeholder_frames(task=task, frame_budget=frame_budget)
                else:
                    raise FileNotFoundError(f"Video path not found for training example: {video_path}")
                for frame_path in frame_paths:
                    expanded.append({"type": "image", "image": frame_path})
        return expanded

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("This MVP collator currently supports only micro-batch size 1.")

        row = features[0]
        user_content = self._expand_user_content(row)
        assistant_text = row["messages"][1]["content"]

        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
        prompt_messages = [
            {"role": "user", "content": user_content},
        ]

        full_inputs = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        labels = full_inputs["input_ids"].clone()
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        prompt_len = min(prompt_len, int(labels.shape[1]))
        labels[:, :prompt_len] = -100
        full_inputs["labels"] = labels
        return full_inputs


def bootstrap_lora_model(config: dict[str, Any]) -> None:
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing training dependencies. Install at least: pip install transformers peft bitsandbytes torch"
        ) from exc

    model_cfg = config["model"]
    lora_cfg = model_cfg["lora"]
    base_model = model_cfg["base_model"]
    quantization = str(model_cfg.get("quantization", "none")).lower()

    logger.info("Loading processor: %s", base_model)
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    logger.info("Loading model: %s", base_model)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4bit QLoRA bootstrap.")
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8bit bootstrap.")
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model, **model_kwargs)
    if quantization in {"4bit", "8bit"}:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable, total = count_trainable_params(model)

    logger.info("Processor loaded: %s", type(processor).__name__)
    logger.info("LoRA attached. Trainable params: %s / %s", trainable, total)
    return processor, model


def prepare_trainer_skeleton(
    *,
    processor: Any,
    model: Any,
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> None:
    try:
        from torch.utils.data import Dataset
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing trainer dependencies. Install at least: pip install transformers"
        ) from exc

    training_cfg = config["training"]
    output_dir = ROOT / "experiments" / "results" / "qwen_lora" / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    class _SimpleDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self.rows[idx]

    train_dataset = _SimpleDataset(train_rows)
    val_dataset = _SimpleDataset(val_rows)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg["epochs"],
        per_device_train_batch_size=training_cfg["micro_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        bf16=training_cfg.get("bf16", False),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )

    data_collator = MotorExamQwenDataCollator(processor=processor, config=config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_rows) > 0 else None,
        data_collator=data_collator,
    )

    logger.info("Trainer skeleton prepared.")
    logger.info("Processor type: %s", type(processor).__name__)
    logger.info("Model type: %s", type(model).__name__)
    logger.info("Train dataset rows: %d", len(train_rows))
    logger.info("Val dataset rows: %d", len(val_rows))
    logger.info("Output dir: %s", output_dir)
    logger.info("Training args: %s", training_args.to_dict())
    logger.info("Trainer type: %s", type(trainer).__name__)
    return trainer


def main() -> int:
    args = parse_args()
    args.config = resolve_path(args.config)
    config = load_yaml(args.config)

    train_path = resolve_path(Path(config["data"]["train_jsonl"]))
    val_path = resolve_path(Path(config["data"]["val_jsonl"]))
    tasks = config["data"].get("tasks", [])
    question_types = config["data"].get("question_types", [])

    logger.info("Config: %s", args.config)
    logger.info("Train source: %s", train_path)
    logger.info("Val source: %s", val_path)

    train_rows = load_split(
        train_path,
        tasks=tasks,
        question_types=question_types,
        include_rationale=args.include_rationale,
        limit=args.limit_train,
    )
    val_rows = load_split(
        val_path,
        tasks=tasks,
        question_types=question_types,
        include_rationale=args.include_rationale,
        limit=args.limit_val,
    )

    summarize_split("train", train_rows)
    summarize_split("val", val_rows)

    if train_rows:
        logger.info("Sample train record:\n%s", json.dumps(train_rows[0], ensure_ascii=False, indent=2)[:2000])

    processor = None
    model = None
    trainer = None
    if args.bootstrap_model or args.prepare_trainer:
        processor, model = bootstrap_lora_model(config)

    if args.prepare_trainer:
        trainer = prepare_trainer_skeleton(
            processor=processor,
            model=model,
            config=config,
            train_rows=train_rows,
            val_rows=val_rows,
        )

    if args.train:
        if trainer is None:
            raise SystemExit("Use --prepare-trainer together with --train.")
        logger.info("Starting training loop...")
        trainer.train()
        trainer.save_model()
        logger.info("Training finished and model saved.")

    if args.dry_run or not (args.bootstrap_model or args.prepare_trainer):
        logger.info("Dry-run complete.")
        return 0

    if not args.train:
        logger.info("Bootstrap / trainer preparation finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
