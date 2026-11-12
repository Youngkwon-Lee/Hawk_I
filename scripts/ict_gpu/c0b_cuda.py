#!/usr/bin/env python3
"""Guarded, foreground-only llama.cpp CUDA C0B launcher for the ICT WSL host.

No installation, transfers, model download, service changes or network listeners
occur during the default preflight. --serve requires four SHA-256-pinned files.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROLES = ("server", "model", "mmproj", "adapter")


class PreflightError(RuntimeError):
    pass


def locate_smi() -> str:
    for candidate in (shutil.which("nvidia-smi"), "/usr/lib/wsl/lib/nvidia-smi"):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise PreflightError("NVIDIA-SMI_NOT_AVAILABLE")


def smi(*args: str) -> str:
    try:
        result = subprocess.run([locate_smi(), *args], text=True,
                                capture_output=True, timeout=12, check=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise PreflightError("NVIDIA_QUERY_FAILED") from exc
    return result.stdout.strip()


def parse_gpu_csv(stdout: str, gpu_index: int = 0) -> dict:
    rows = list(csv.reader(stdout.splitlines(), skipinitialspace=True))
    for row in rows:
        if len(row) != 4:
            raise PreflightError("GPU_QUERY_UNEXPECTED_FORMAT")
        try:
            index = int(row[0].strip())
            free_mib = int(row[2].strip())
            util_percent = int(row[3].strip())
        except ValueError as exc:
            raise PreflightError("GPU_QUERY_UNEXPECTED_FORMAT") from exc
        if index == gpu_index:
            if not row[1].strip().startswith("GPU-"):
                raise PreflightError("GPU_UUID_MISSING")
            return {"index": index, "uuid": row[1].strip(),
                    "free_mib": free_mib, "util_percent": util_percent}
    raise PreflightError("GPU_0_NOT_FOUND")


def parse_compute_gpu_uuids(stdout: str) -> set[str]:
    if not stdout or stdout.strip().lower() == "no running processes found":
        return set()
    uuids = set()
    for row in csv.reader(stdout.splitlines(), skipinitialspace=True):
        if len(row) != 2 or not row[0].strip().startswith("GPU-"):
            raise PreflightError("GPU_PROCESS_QUERY_UNEXPECTED_FORMAT")
        if not row[1].strip().isdigit():
            raise PreflightError("GPU_PROCESS_QUERY_UNEXPECTED_FORMAT")
        uuids.add(row[0].strip())
    return uuids


def check_gpu(min_free_mib: int, max_util_percent: int) -> dict:
    gpu = parse_gpu_csv(smi("--query-gpu=index,uuid,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits"))
    if gpu["free_mib"] < min_free_mib:
        raise PreflightError("GPU_0_INSUFFICIENT_FREE_MEMORY")
    if gpu["util_percent"] > max_util_percent:
        raise PreflightError("GPU_0_ALREADY_BUSY")
    processes = parse_compute_gpu_uuids(smi("--query-compute-apps=gpu_uuid,pid",
                                           "--format=csv,noheader,nounits"))
    if gpu["uuid"] in processes:
        raise PreflightError("GPU_0_HAS_COMPUTE_PROCESS")
    return gpu


def check_host(expected_host: str, expected_user: str) -> None:
    if not expected_host or socket.gethostname().casefold() != expected_host.casefold():
        raise PreflightError("WRONG_HOST")
    if not expected_user or getpass.getuser() != expected_user:
        raise PreflightError("WRONG_USER")
    if sys.platform != "linux":
        raise PreflightError("LINUX_WSL_REQUIRED")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise PreflightError("MANIFEST_NOT_FOUND")
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    except (KeyError, ValueError, OSError, TypeError) as exc:
        raise PreflightError("INVALID_MANIFEST") from exc
    if not isinstance(entries, dict) or set(entries) != set(ROLES):
        raise PreflightError("INVALID_MANIFEST_ROLES")
    paths: dict[str, str] = {}
    for role in ROLES:
        entry = entries[role]
        if not isinstance(entry, dict):
            raise PreflightError("INVALID_MANIFEST_ENTRY")
        raw_path = entry.get("path")
        expected_sha = entry.get("sha256")
        if (not isinstance(raw_path, str)
                or not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha)):
            raise PreflightError("INVALID_MANIFEST_ENTRY")
        path = Path(raw_path)
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise PreflightError("MODEL_ASSET_NOT_PRESENT")
        if role == "server" and not os.access(path, os.X_OK):
            raise PreflightError("SERVER_NOT_EXECUTABLE")
        if sha256_file(path) != expected_sha:
            raise PreflightError("ASSET_SHA256_MISMATCH")
        paths[role] = str(path)
    return paths


def launch_args(paths: dict[str, str], port: int, ctx_size: int) -> list[str]:
    return [
        paths["server"], "--model", paths["model"],
        "--mmproj", paths["mmproj"], "--lora", paths["adapter"],
        "--alias", "hawkeye-c0b-seed42",
        "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(ctx_size), "--parallel", "1",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--n-gpu-layers", "99", "--jinja", "--temp", "0",
        "--top-k", "1", "--top-p", "1",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--expected-user", required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="operator-verified local SHA-256 manifest; not generated here")
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument("--max-util-pct", type=int, default=5)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--serve", action="store_true",
                        help="explicitly start a foreground process, after all guards")
    args = parser.parse_args()
    try:
        if not (1 <= args.port <= 65535 and 1024 <= args.ctx_size <= 16384):
            raise PreflightError("INVALID_PORT_OR_CONTEXT")
        if not (0 <= args.max_util_pct <= 100 and args.min_free_mib >= 12000):
            raise PreflightError("UNSAFE_GPU_GUARD")
        check_host(args.expected_host, args.expected_user)
        gpu = check_gpu(args.min_free_mib, args.max_util_pct)
        paths = verify_manifest(args.manifest)
        report = {"status": "preflight_pass", "gpu_index": 0,
                  "free_mib": gpu["free_mib"], "util_percent": gpu["util_percent"],
                  "files_sha256_verified": list(paths), "mode": "serve" if args.serve else "check"}
        print(json.dumps(report, sort_keys=True), flush=True)
        if not args.serve:
            return 0
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "0"
        os.execve(paths["server"], launch_args(paths, args.port, args.ctx_size), env)
    except (PreflightError, OSError) as exc:
        error = str(exc) if isinstance(exc, PreflightError) else "SERVER_EXEC_FAILED"
        print(json.dumps({"status": "blocked", "reason": error}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
