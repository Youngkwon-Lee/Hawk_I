#!/usr/bin/env python3
"""HTTP bootstrap reporter for Qwen LoRA setup on Runpod."""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file


HOST = os.getenv("BOOTSTRAP_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("BOOTSTRAP_SERVER_PORT", "8000"))
REPO_URL = os.getenv("BOOTSTRAP_REPO_URL", "https://github.com/Youngkwon-Lee/Hawk_I.git")
REPO_REF = os.getenv("BOOTSTRAP_REPO_REF", "main")
REPO_ROOT = Path(os.getenv("BOOTSTRAP_REPO_ROOT", "/workspace/hawkeye"))
HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", "/workspace/.hf-cache"))
CONFIG_PATH = os.getenv(
    "BOOTSTRAP_CONFIG_PATH",
    "experiments/configs/vlm/qwen_vl_lora_motorexam_toy_v0_1.yaml",
)
AUTO_START = os.getenv("BOOTSTRAP_AUTO_START", "1") == "1"
LOG_PATH = Path(os.getenv("BOOTSTRAP_LOG_PATH", "/workspace/logs/hawkeye/qwen_lora_bootstrap.log"))
EXTRA_ARGS = os.getenv("BOOTSTRAP_EXTRA_ARGS", "")

app = Flask(__name__)

_state: dict[str, Any] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "error": None,
    "command": None,
}
_thread: threading.Thread | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)


def append_log(message: str) -> None:
    ensure_dirs()
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message}\n")


def read_log(max_chars: int = 12000) -> str:
    try:
        return LOG_PATH.read_text(encoding="utf-8")[-max_chars:]
    except Exception as exc:  # pragma: no cover
        return f"log unavailable: {type(exc).__name__}: {exc}"


def list_result_files(max_entries: int = 200) -> list[dict[str, Any]]:
    results_root = REPO_ROOT / "experiments" / "results" / "qwen_lora"
    if not results_root.exists():
        return []

    files = []
    for path in sorted(results_root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
        if len(files) >= max_entries:
            break
    return files


def safe_repo_path(relative_path: str) -> Path:
    candidate = (REPO_ROOT / relative_path).resolve()
    repo_root_resolved = REPO_ROOT.resolve()
    if repo_root_resolved not in candidate.parents and candidate != repo_root_resolved:
        raise ValueError(f"path escapes repo root: {relative_path}")
    return candidate


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def build_bootstrap_command(
    *,
    config_path: str | None = None,
    extra_args: str | None = None,
    custom_command: str | None = None,
) -> str:
    config_path = config_path or CONFIG_PATH
    if custom_command:
        return custom_command.strip()

    extra = (extra_args if extra_args is not None else EXTRA_ARGS).strip()
    extra_command = ""
    if extra:
        extra_command = f"python scripts/vlm/train_qwen_vl_lora.py --config {_shell_quote(config_path)} {extra}"
    return f"""
set -e
mkdir -p "{HF_CACHE_DIR}" "{REPO_ROOT.parent}"
if [ ! -d "{REPO_ROOT}/.git" ]; then
  git clone "{REPO_URL}" "{REPO_ROOT}"
fi
cd "{REPO_ROOT}"
git fetch origin "{REPO_REF}"
git checkout "{REPO_REF}"
git pull --ff-only origin "{REPO_REF}"
export HF_HOME="{HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="{HF_CACHE_DIR}"
python scripts/vlm/train_qwen_vl_lora.py --config {_shell_quote(config_path)} --dry-run --include-rationale
python scripts/vlm/train_qwen_vl_lora.py --config {_shell_quote(config_path)} --dry-run --include-rationale --bootstrap-model
{extra_command}
""".strip()


def run_bootstrap(
    *,
    config_path: str | None = None,
    extra_args: str | None = None,
    custom_command: str | None = None,
) -> None:
    global _state
    command = build_bootstrap_command(
        config_path=config_path,
        extra_args=extra_args,
        custom_command=custom_command,
    )
    _state = {
        "status": "running",
        "started_at": now_iso(),
        "finished_at": None,
        "return_code": None,
        "error": None,
        "command": command,
        "config_path": config_path or CONFIG_PATH,
        "extra_args": extra_args if extra_args is not None else EXTRA_ARGS,
    }
    append_log("bootstrap started")
    append_log(f"repo={REPO_URL} ref={REPO_REF}")
    append_log(f"repo_root={REPO_ROOT}")
    append_log(f"hf_cache={HF_CACHE_DIR}")
    append_log(f"config={config_path or CONFIG_PATH}")
    if extra_args is not None:
        append_log(f"extra_args={extra_args}")
    if custom_command:
        append_log("custom_command override active")

    with LOG_PATH.open("a", encoding="utf-8") as handle:
        process = subprocess.run(
            ["/bin/bash", "-lc", command],
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    _state["finished_at"] = now_iso()
    _state["return_code"] = process.returncode
    _state["status"] = "succeeded" if process.returncode == 0 else "failed"
    append_log(f"bootstrap finished rc={process.returncode}")


def start_background_bootstrap(
    *,
    config_path: str | None = None,
    extra_args: str | None = None,
    custom_command: str | None = None,
) -> bool:
    global _thread
    if _thread is not None and _thread.is_alive():
        return False
    _thread = threading.Thread(
        target=run_bootstrap,
        kwargs={
            "config_path": config_path,
            "extra_args": extra_args,
            "custom_command": custom_command,
        },
        daemon=True,
    )
    _thread.start()
    return True


@app.route("/", methods=["GET"])
def root() -> Any:
    return jsonify(
        {
            "service": "hawkeye-qwen-lora-bootstrap-server",
            "status": _state["status"],
        }
    )


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "service": "hawkeye-qwen-lora-bootstrap-server",
            "status": _state["status"],
            "started_at": _state["started_at"],
            "finished_at": _state["finished_at"],
            "return_code": _state["return_code"],
            "error": _state["error"],
        }
    )


@app.route("/status", methods=["GET"])
def status() -> Any:
    return jsonify(_state)


@app.route("/logs", methods=["GET"])
def logs() -> Any:
    return jsonify({"log_tail": read_log()})


@app.route("/artifacts", methods=["GET"])
def artifacts() -> Any:
    return jsonify({"files": list_result_files()})


@app.route("/upload", methods=["POST"])
def upload() -> Any:
    relative_path = request.form.get("relative_path") or request.args.get("relative_path")
    if not relative_path:
        return jsonify({"error": "relative_path is required"}), 400

    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "file is required"}), 400

    try:
        target = safe_repo_path(relative_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    target.parent.mkdir(parents=True, exist_ok=True)
    uploaded.save(target)
    append_log(f"uploaded file to {target}")
    return jsonify({"saved": str(target), "size": target.stat().st_size})


@app.route("/download", methods=["GET"])
def download() -> Any:
    relative_path = request.args.get("path")
    if not relative_path:
        return jsonify({"error": "path is required"}), 400
    try:
        target = safe_repo_path(relative_path)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not target.exists() or not target.is_file():
        return jsonify({"error": f"file not found: {relative_path}"}), 404
    append_log(f"downloaded file {target}")
    return send_file(target, as_attachment=True, download_name=target.name)


@app.route("/run", methods=["POST"])
def run() -> Any:
    payload = request.get_json(silent=True) or {}
    config_path = payload.get("config_path")
    extra_args = payload.get("extra_args")
    custom_command = payload.get("command")
    started = start_background_bootstrap(
        config_path=config_path,
        extra_args=extra_args,
        custom_command=custom_command,
    )
    return jsonify(
        {
            "started": started,
            "status": _state["status"],
            "config_path": config_path or CONFIG_PATH,
            "custom_command": bool(custom_command),
        }
    )


if __name__ == "__main__":
    ensure_dirs()
    append_log("bootstrap server starting")
    if AUTO_START:
        start_background_bootstrap()
    app.run(host=HOST, port=PORT, debug=False)
