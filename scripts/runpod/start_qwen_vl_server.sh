#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${STARTUP_LOG_PATH:-/app/startup.log}"
REQ_FILE="${QWEN_SERVER_REQUIREMENTS:-/app/requirements-qwen-vl-server.txt}"

mkdir -p "$(dirname "$LOG_FILE")"
echo "[startup] $(date -u +%Y-%m-%dT%H:%M:%SZ) Starting Qwen VL server" | tee "$LOG_FILE"
echo "[startup] model=${QWEN_SERVE_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}" | tee -a "$LOG_FILE"
echo "[startup] host=${QWEN_SERVER_HOST:-0.0.0.0} port=${QWEN_SERVER_PORT:-8000}" | tee -a "$LOG_FILE"
echo "[startup] python=$(python --version 2>&1)" | tee -a "$LOG_FILE"
echo "[startup] requirements=$REQ_FILE" | tee -a "$LOG_FILE"

python -m pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
python -m pip install --no-cache-dir -r "$REQ_FILE" 2>&1 | tee -a "$LOG_FILE"

python /app/qwen_vl_openai_server.py 2>&1 | tee -a "$LOG_FILE"
