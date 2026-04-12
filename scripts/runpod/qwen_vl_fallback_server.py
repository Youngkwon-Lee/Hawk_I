#!/usr/bin/env python3
"""Fallback diagnostics server for Qwen VL startup failures."""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify

HOST = os.getenv("QWEN_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN_SERVER_PORT", "8000"))
STARTUP_LOG_PATH = Path(os.getenv("STARTUP_LOG_PATH", "/app/startup.log"))
FALLBACK_REASON = os.getenv("QWEN_FALLBACK_REASON", "qwen-vl server exited before binding")

app = Flask(__name__)


def read_log() -> str:
    try:
        return STARTUP_LOG_PATH.read_text(encoding="utf-8")[-12000:]
    except Exception as exc:  # pragma: no cover
        return f"startup log unavailable: {type(exc).__name__}: {exc}"


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "hawkeye-qwen-vl-fallback",
            "status": "degraded",
            "reason": FALLBACK_REASON,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "degraded",
            "service": "hawkeye-qwen-vl-fallback",
            "reason": FALLBACK_REASON,
            "startup_log_tail": read_log(),
        }
    )


@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify(
        {
            "object": "list",
            "data": [],
            "warning": FALLBACK_REASON,
        }
    )


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
