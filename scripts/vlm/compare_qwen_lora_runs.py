"""
Compare base Qwen2.5-VL against LoRA adapters on a small set of MotorExam-VQA samples.

This script is evaluation-oriented:
- load base model
- optionally attach one or more LoRA adapters
- run sampled-frame inference on the same examples
- emit a compact comparison JSONL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import gc
import sys
import tempfile
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
    from data.export_motorexam_qwen_sft import load_jsonl as load_sft_jsonl
except ModuleNotFoundError:
    from scripts.data.export_motorexam_qwen_sft import load_jsonl as load_sft_jsonl


DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "vlm" / "qwen_vl_lora_motorexam_small_v0_1.yaml"
DEFAULT_INPUT = ROOT / "experiments" / "data" / "motorexam_qwen_sft_small_val_v0_1.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "qwen_lora" / "qwen_comparison_v0_1.jsonl"
TEMP_FRAME_DIR = Path(tempfile.gettempdir()) / "hawkeye_qwen_compare_frames"
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
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--small-adapter", type=Path, default=ROOT / "experiments" / "results" / "qwen_lora" / "qwen_vl_lora_motorexam_small_v0_1")
    parser.add_argument("--overnight-adapter", type=Path, default=ROOT / "experiments" / "results" / "qwen_lora" / "qwen_vl_lora_motorexam_overnight_v0_1")
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


def uniform_indices(total_frames: int, n_frames: int) -> list[int]:
    if n_frames <= 0:
        return []
    if total_frames <= 1:
        return [0]
    return sorted({int(i * (total_frames - 1) / max(n_frames - 1, 1)) for i in range(n_frames)})


def task_aware_indices(total_frames: int, n_frames: int, task: str) -> list[int]:
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
    import cv2

    digest = hashlib.md5(f"{video_path}|{task}|{frame_budget}|{frame_width}".encode("utf-8")).hexdigest()[:12]
    cache_dir = TEMP_FRAME_DIR / digest
    if cache_dir.exists():
        cached = sorted(cache_dir.glob("*.jpg"))
        if cached:
            return [str(path) for path in cached]

    cache_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for comparison: {video_path}")
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
    return saved_paths


def placeholder_frame_paths(*, task: str, frame_budget: int, frame_width: int) -> list[str]:
    from PIL import Image, ImageDraw

    digest = hashlib.md5(f"placeholder|{task}|{frame_budget}|{frame_width}".encode("utf-8")).hexdigest()[:12]
    cache_dir = TEMP_FRAME_DIR / f"placeholder_{digest}"
    if cache_dir.exists():
        cached = sorted(cache_dir.glob("*.jpg"))
        if cached:
            return [str(path) for path in cached]

    cache_dir.mkdir(parents=True, exist_ok=True)
    width = frame_width
    height = max(256, int(width * 0.75))
    saved_paths: list[str] = []
    for i in range(frame_budget):
        image = Image.new("RGB", (width, height), color=(18, 24, 34))
        draw = ImageDraw.Draw(image)
        draw.rectangle((16, 16, width - 16, height - 16), outline=(90, 130, 180), width=3)
        draw.text((32, 40), "Hawkeye compare placeholder", fill=(235, 240, 245))
        draw.text((32, 80), f"task={task}", fill=(235, 240, 245))
        draw.text((32, 120), f"frame={i+1}/{frame_budget}", fill=(235, 240, 245))
        out_path = cache_dir / f"{i:03d}.jpg"
        image.save(out_path, format="JPEG", quality=90)
        saved_paths.append(str(out_path))
    return saved_paths


def load_model_bundle(base_model: str, quantization: str, adapter_path: Path | None = None):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    q = quantization.lower()
    if q == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif q == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model, **model_kwargs)

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return processor, model


def render_messages(
    record: dict[str, Any],
    frame_budget: int,
    frame_width: int,
    *,
    allow_missing_videos_as_blank_images: bool,
) -> list[dict[str, Any]]:
    task = record["task"]
    user = record["messages"][0]
    content: list[dict[str, Any]] = []
    for item in user["content"]:
        if item.get("type") == "text":
            content.append(item)
        elif item.get("type") == "video":
            video_path = ROOT / item["video"]
            if video_path.exists():
                frame_paths = sample_video_to_frame_paths(
                    video_path,
                    task=task,
                    frame_budget=frame_budget,
                    frame_width=frame_width,
                )
            elif allow_missing_videos_as_blank_images:
                frame_paths = placeholder_frame_paths(
                    task=task,
                    frame_budget=frame_budget,
                    frame_width=frame_width,
                )
            else:
                raise FileNotFoundError(f"Comparison video path not found: {video_path}")
            for frame_path in frame_paths:
                content.append({"type": "image", "image": frame_path})
    return [{"role": "user", "content": content}]


def run_inference(
    processor: Any,
    model: Any,
    record: dict[str, Any],
    frame_budget: int,
    frame_width: int,
    *,
    allow_missing_videos_as_blank_images: bool,
) -> str:
    import torch

    messages = render_messages(
        record,
        frame_budget=frame_budget,
        frame_width=frame_width,
        allow_missing_videos_as_blank_images=allow_missing_videos_as_blank_images,
    )
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    prompt_len = int(inputs["input_ids"].shape[1])
    decoded = processor.batch_decode(
        [outputs[0][prompt_len:]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return decoded


def main() -> int:
    args = parse_args()
    args.config = resolve_path(args.config)
    args.input = resolve_path(args.input)
    args.output = resolve_path(args.output)
    args.small_adapter = resolve_path(args.small_adapter)
    args.overnight_adapter = resolve_path(args.overnight_adapter)

    config = load_yaml(args.config)
    rows = load_sft_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    base_model = config["model"]["base_model"]
    quantization = str(config["model"].get("quantization", "none"))
    frame_width = 512
    frame_budget_by_task = {k: int(v) for k, v in config["data"].get("sampling_policy", {}).items()}
    allow_missing_videos = bool(config["data"].get("allow_missing_videos_as_blank_images", False))

    results = [
        {
            "sample_id": record["sample_id"],
            "task": record["task"],
            "question_id": record["question_id"],
            "target": record["messages"][1]["content"],
        }
        for record in rows
    ]

    model_specs: list[tuple[str, Path | None]] = [("base", None)]
    if not args.base_only and args.small_adapter.exists():
        model_specs.append(("small", args.small_adapter))
    if not args.base_only and args.overnight_adapter.exists():
        model_specs.append(("overnight", args.overnight_adapter))

    for name, adapter_path in model_specs:
        logger.info("Loading %s bundle...", name)
        processor, model = load_model_bundle(base_model, quantization, adapter_path)
        try:
            for idx, record in enumerate(rows):
                task = record["task"]
                frame_budget = int(frame_budget_by_task.get(task, 8))
                logger.info("Running %s on %s", name, record["sample_id"])
                results[idx][name] = run_inference(
                    processor,
                    model,
                    record,
                    frame_budget=frame_budget,
                    frame_width=frame_width,
                    allow_missing_videos_as_blank_images=allow_missing_videos,
                )
        finally:
            del model
            del processor
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Wrote %d comparison rows to %s", len(results), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
