#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${STARTUP_LOG_PATH:-/app/startup.log}"
HOST="${QWEN_SERVER_HOST:-0.0.0.0}"
PORT="${QWEN_SERVER_PORT:-8000}"

mkdir -p "$(dirname "$LOG_FILE")"
echo "[wrapper] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting qwen-vl wrapper" > "$LOG_FILE"
echo "[wrapper] host=${HOST} port=${PORT}" >> "$LOG_FILE"

python /app/qwen_vl_openai_server.py >> "$LOG_FILE" 2>&1 &
QWEN_PID=$!
echo "[wrapper] qwen pid=${QWEN_PID}" >> "$LOG_FILE"

check_port() {
  python - "$PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect(("127.0.0.1", port))
except Exception:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
}

for _ in $(seq 1 20); do
  if check_port; then
    echo "[wrapper] qwen server bound to port ${PORT}" >> "$LOG_FILE"
    wait "$QWEN_PID"
    exit $?
  fi
  if ! kill -0 "$QWEN_PID" 2>/dev/null; then
    wait "$QWEN_PID" || true
    export QWEN_FALLBACK_REASON="qwen-vl server exited before binding"
    echo "[wrapper] qwen server exited early; starting fallback" >> "$LOG_FILE"
    exec python /app/qwen_vl_fallback_server.py
  fi
  sleep 1
done

echo "[wrapper] qwen server did not bind within timeout; starting fallback" >> "$LOG_FILE"
kill "$QWEN_PID" 2>/dev/null || true
wait "$QWEN_PID" || true
export QWEN_FALLBACK_REASON="qwen-vl server did not bind within timeout"
exec python /app/qwen_vl_fallback_server.py
