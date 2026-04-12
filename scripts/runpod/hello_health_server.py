#!/usr/bin/env python3
"""Minimal HTTP health server for Runpod + GHCR smoke testing."""

from __future__ import annotations

import os
from flask import Flask, jsonify


HOST = os.getenv("HELLO_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("HELLO_SERVER_PORT", "8000"))

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "hawkeye-hello-health",
            "status": "ok",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "service": "hawkeye-hello-health",
            "status": "ok",
            "host": HOST,
            "port": PORT,
        }
    )


if __name__ == "__main__":
    print(f"Starting hello health server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)

