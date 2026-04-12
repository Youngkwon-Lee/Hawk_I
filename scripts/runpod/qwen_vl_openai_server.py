#!/usr/bin/env python3
"""OpenAI-compatible Qwen2.5-VL server with lazy model loading."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

try:
    import torch
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - best effort diagnostics
    torch = None
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    import cv2
    CV2_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    cv2 = None
    CV2_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


MODEL_ID = os.getenv("QWEN_SERVE_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
HOST = os.getenv("QWEN_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN_SERVER_PORT", "8000"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "768"))
STARTUP_LOG_PATH = Path(os.getenv("STARTUP_LOG_PATH", "/app/startup.log"))
RAW_VIDEO_SAMPLE_FRAMES = int(os.getenv("QWEN_RAW_VIDEO_SAMPLE_FRAMES", "12"))
RAW_VIDEO_FRAME_WIDTH = int(os.getenv("QWEN_RAW_VIDEO_FRAME_WIDTH", "512"))

app = Flask(__name__)

TEMP_DIR = Path(tempfile.gettempdir()) / "hawkeye_qwen_vl_server"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

_state: dict[str, Any] = {
    "loaded": False,
    "load_error": None,
    "processor": None,
    "model": None,
    "process_vision_info": None,
}
_model_lock = threading.Lock()


def append_startup_log(message: str) -> None:
    STARTUP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STARTUP_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.utcnow().isoformat()}Z] {message}\n")


def read_startup_log_tail(max_chars: int = 4000) -> str:
    try:
        return STARTUP_LOG_PATH.read_text(encoding="utf-8")[-max_chars:]
    except Exception as exc:  # pragma: no cover
        return f"startup log unavailable: {type(exc).__name__}: {exc}"


def runtime_diagnostics() -> dict[str, Any]:
    nvidia_smi_output = ""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        nvidia_smi_output = (completed.stdout or completed.stderr).strip()
    except Exception as exc:  # pragma: no cover
        nvidia_smi_output = f"nvidia-smi unavailable: {type(exc).__name__}: {exc}"

    diagnostics = {
        "status": "ok",
        "mode": "qwen_vl_openai_server",
        "model": MODEL_ID,
        "loaded": _state["loaded"],
        "load_error": _state["load_error"],
        "torch_import_error": TORCH_IMPORT_ERROR,
        "cv2_import_error": CV2_IMPORT_ERROR,
        "startup_log_tail": read_startup_log_tail(),
        "nvidia_smi": nvidia_smi_output,
    }

    if torch is None:
        diagnostics.update(
            {
                "cuda_available": False,
                "device_count": 0,
                "torch_version": None,
                "torch_cuda_version": None,
                "cuda_device_name": None,
                "cuda_error": TORCH_IMPORT_ERROR,
                "device_map": None,
            }
        )
        return diagnostics

    cuda_available = torch.cuda.is_available()
    cuda_device_name = None
    cuda_error = None
    if cuda_available:
        try:
            cuda_device_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            cuda_error = f"{type(exc).__name__}: {exc}"

    diagnostics.update(
        {
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_device_name": cuda_device_name,
            "cuda_error": cuda_error,
            "device_map": str(getattr(_state["model"], "hf_device_map", None)),
        }
    )
    return diagnostics


def load_model_if_needed() -> None:
    if _state["loaded"]:
        return
    if _state["load_error"]:
        raise RuntimeError(_state["load_error"])

    with _model_lock:
        if _state["loaded"]:
            return
        if _state["load_error"]:
            raise RuntimeError(_state["load_error"])

        try:
            append_startup_log("lazy load begin")
            if torch is None:
                raise RuntimeError(f"torch import failed: {TORCH_IMPORT_ERROR}")

            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            append_startup_log(
                f"loading model={MODEL_ID} dtype={dtype} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}"
            )

            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto",
            )
            model.eval()

            _state["processor"] = processor
            _state["model"] = model
            _state["process_vision_info"] = process_vision_info
            _state["loaded"] = True
            append_startup_log("lazy load success")
        except Exception:
            _state["load_error"] = traceback.format_exc()
            append_startup_log(_state["load_error"])
            raise RuntimeError(_state["load_error"])


def decode_data_url_to_file(url: str, *, prefix: str) -> str:
    header, encoded = url.split(",", 1)
    suffix = ".bin"
    if "image/jpeg" in header or "image/jpg" in header:
        suffix = ".jpg"
    elif "image/png" in header:
        suffix = ".png"
    elif "video/mp4" in header:
        suffix = ".mp4"
    elif "video/quicktime" in header:
        suffix = ".mov"
    elif "video/webm" in header:
        suffix = ".webm"
    output_path = TEMP_DIR / f"{prefix}_{abs(hash(encoded))}{suffix}"
    if not output_path.exists():
        output_path.write_bytes(base64.b64decode(encoded))
    return str(output_path)


def resolve_image_path(url: str) -> str:
    if url.startswith("data:image/"):
        return decode_data_url_to_file(url, prefix="image")
    if url.startswith("file://"):
        return url[len("file://") :]
    if url.startswith(("http://", "https://")):
        return url
    path = Path(url)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


def resolve_video_path(url: str) -> str:
    if url.startswith("data:video/"):
        return decode_data_url_to_file(url, prefix="video")
    if url.startswith("file://"):
        return url[len("file://") :]
    if url.startswith(("http://", "https://")):
        return url
    path = Path(url)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


def sample_video_frames(video_path: str, n_frames: int, frame_width: int) -> list[str]:
    if cv2 is None:
        raise RuntimeError(f"opencv unavailable for raw video sampling: {CV2_IMPORT_ERROR}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for sampling: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames found in video: {video_path}")

    indices = sorted({int(i * (total_frames - 1) / max(n_frames - 1, 1)) for i in range(n_frames)})
    sampled_paths: list[str] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        if frame_width > 0 and frame.shape[1] > frame_width:
            scale = frame_width / frame.shape[1]
            resized_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (frame_width, resized_height), interpolation=cv2.INTER_AREA)

        out_path = TEMP_DIR / f"sampled_{Path(video_path).stem}_{idx}.jpg"
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        out_path.write_bytes(buffer.tobytes())
        sampled_paths.append(str(out_path))

    cap.release()

    if not sampled_paths:
        raise RuntimeError(f"Failed to sample any frames from video: {video_path}")

    append_startup_log(
        f"sampled raw video to {len(sampled_paths)} frames from {video_path} (requested={n_frames}, width={frame_width})"
    )
    return sampled_paths


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        items: list[dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    items.append({"type": "text", "text": item.get("text", "")})
                elif item_type in {"image_url", "input_image", "image"}:
                    image_url = item.get("image_url", {})
                    if item_type == "image":
                        image_url = item.get("image", "")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url", "")
                    if image_url:
                        items.append({"type": "image", "image": resolve_image_path(str(image_url))})
                elif item_type in {"video", "video_url", "input_video"}:
                    video_url = item.get("video_url", {})
                    if item_type == "video":
                        video_url = item.get("video", "")
                    if isinstance(video_url, dict):
                        video_url = video_url.get("url", "")
                    if video_url:
                        resolved_video = resolve_video_path(str(video_url))
                        sampled_frames = sample_video_frames(
                            resolved_video,
                            n_frames=RAW_VIDEO_SAMPLE_FRAMES,
                            frame_width=RAW_VIDEO_FRAME_WIDTH,
                        )
                        for sampled_path in sampled_frames:
                            items.append({"type": "image", "image": sampled_path})
        else:
            items = [{"type": "text", "text": str(content)}]
        normalized.append({"role": role, "content": items})
    return normalized


def generate_vlm(messages: list[dict[str, Any]], temperature: float, top_p: float, max_tokens: int) -> str:
    load_model_if_needed()

    processor = _state["processor"]
    model = _state["model"]
    process_vision_info = _state["process_vision_info"]

    normalized = normalize_messages(messages)
    chat_template = processor.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(normalized)
    inputs = processor(
        text=[chat_template],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_tokens,
        )
    trimmed_ids = [
        output_ids[input_ids.shape[0] :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


@app.route("/", methods=["GET"])
def root() -> Any:
    return jsonify(
        {
            "service": "hawkeye-qwen-vl-openai-server",
            "status": "ok",
            "loaded": _state["loaded"],
        }
    )


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(runtime_diagnostics())


@app.route("/v1/models", methods=["GET"])
def models() -> Any:
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "owned_by": "hawkeye",
                }
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions() -> Any:
    data = request.get_json(force=True) or {}
    messages = data.get("messages", [])
    model_name = data.get("model", MODEL_ID)
    temperature = float(data.get("temperature", 0.0))
    top_p = float(data.get("top_p", 1.0))
    max_tokens = int(data.get("max_tokens", MAX_NEW_TOKENS))

    try:
        content = generate_vlm(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(exc),
                        "type": "model_load_or_inference_error",
                    },
                    "diagnostics": runtime_diagnostics(),
                }
            ),
            500,
        )

    return jsonify(
        {
            "id": "chatcmpl-hawkeye-qwen-vl",
            "object": "chat.completion",
            "created": 0,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
    )


if __name__ == "__main__":
    append_startup_log(f"Starting Hawkeye Qwen VL OpenAI-compatible server on http://{HOST}:{PORT}")
    append_startup_log(f"Model: {MODEL_ID}")
    app.run(host=HOST, port=PORT, debug=False)
