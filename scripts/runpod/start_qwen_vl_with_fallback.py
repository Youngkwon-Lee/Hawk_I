#!/usr/bin/env python3
"""Python wrapper that starts Qwen VL and falls back to diagnostics if boot fails."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


LOG_FILE = Path(os.getenv("STARTUP_LOG_PATH", "/app/startup.log"))
HOST = os.getenv("QWEN_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN_SERVER_PORT", "8000"))


def log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def port_is_open(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def start_fallback(reason: str) -> "NoReturn":
    os.environ["QWEN_FALLBACK_REASON"] = reason
    log(f"[wrapper] {reason}; starting fallback")
    os.execv(sys.executable, [sys.executable, "/app/qwen_vl_fallback_server.py"])


def main() -> int:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")
    log("[wrapper] starting qwen-vl python wrapper")
    log(f"[wrapper] host={HOST} port={PORT}")

    process = subprocess.Popen(
        [sys.executable, "/app/qwen_vl_openai_server.py"],
        stdout=LOG_FILE.open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    log(f"[wrapper] qwen pid={process.pid}")

    for _ in range(20):
        if port_is_open(PORT):
            log(f"[wrapper] qwen server bound to port {PORT}")
            return process.wait()
        if process.poll() is not None:
            start_fallback("qwen-vl server exited before binding")
        time.sleep(1)

    try:
        process.kill()
    except Exception:
        pass
    try:
        process.wait(timeout=5)
    except Exception:
        pass
    start_fallback("qwen-vl server did not bind within timeout")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
