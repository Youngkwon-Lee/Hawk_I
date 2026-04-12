#!/usr/bin/env python3
"""OpenAI-compatible text-only Qwen smoke server for Runpod debugging."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


MODEL_ID = os.getenv("QWEN_TEXT_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HOST = os.getenv("QWEN_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN_SERVER_PORT", "8000"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "256"))
STARTUP_LOG_PATH = Path(os.getenv("STARTUP_LOG_PATH", "/app/startup.log"))

app = Flask(__name__)
_model_lock = threading.Lock()
_state: dict[str, Any] = {
    "loaded": False,
    "load_error": None,
    "tokenizer": None,
    "model": None,
}


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
        "mode": "qwen_text_smoke_server",
        "model": MODEL_ID,
        "loaded": _state["loaded"],
        "load_error": _state["load_error"],
        "torch_import_error": TORCH_IMPORT_ERROR,
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
            if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
                raise RuntimeError(f"torch/transformers import failed: {TORCH_IMPORT_ERROR}")

            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            append_startup_log(
                f"loading model={MODEL_ID} dtype={dtype} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}"
            )

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto",
            )
            model.eval()

            _state["tokenizer"] = tokenizer
            _state["model"] = model
            _state["loaded"] = True
            append_startup_log("lazy load success")
        except Exception:
            _state["load_error"] = traceback.format_exc()
            append_startup_log(_state["load_error"])
            raise RuntimeError(_state["load_error"])


def generate_text(messages: list[dict[str, Any]], temperature: float, top_p: float, max_tokens: int) -> str:
    load_model_if_needed()
    tokenizer = _state["tokenizer"]
    model = _state["model"]

    prompt = "\n\n".join(
        f"{message.get('role', 'user').upper()}:\n{message.get('content', '')}"
        for message in messages
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.route("/", methods=["GET"])
def root() -> Any:
    return jsonify({"service": "hawkeye-qwen-text-smoke", "status": "ok", "loaded": _state["loaded"]})


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(runtime_diagnostics())


@app.route("/v1/models", methods=["GET"])
def models() -> Any:
    return jsonify(
        {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "hawkeye"}],
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
        content = generate_text(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
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
            "id": "chatcmpl-hawkeye-qwen-text-smoke",
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
    append_startup_log(f"Starting Hawkeye Qwen text smoke server on http://{HOST}:{PORT}")
    append_startup_log(f"Model: {MODEL_ID}")
    app.run(host=HOST, port=PORT, debug=False)

