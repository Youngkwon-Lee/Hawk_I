"""Poll a Runpod bootstrap pod and stop it automatically when training finishes."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True, help="Runpod pod id to stop when training finishes")
    parser.add_argument("--base-url", required=True, help="Bootstrap reporter base URL, e.g. https://<pod>-8000.proxy.runpod.net")
    parser.add_argument("--interval-sec", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--log-file", type=Path, default=Path("experiments/results/runpod/overnight_monitor.log"))
    parser.add_argument(
        "--artifact-path",
        action="append",
        default=[],
        help="Relative repo path to download from /download before stopping the pod. May be passed multiple times.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("experiments/results/runpod/downloads"),
        help="Local directory where downloaded artifacts will be stored.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message}\n")


def download_artifacts(base_url: str, artifact_paths: list[str], download_dir: Path, log_file: Path) -> None:
    if not artifact_paths:
        return
    for relative_path in artifact_paths:
        url = f"{base_url}/download?path={quote(relative_path, safe='')}"
        local_path = download_dir / Path(relative_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(url, timeout=300)
            append_log(log_file, f"download request path={relative_path} status={response.status_code}")
            if response.status_code != 200:
                append_log(log_file, f"download failed path={relative_path} body={response.text[:500]}")
                continue
            local_path.write_bytes(response.content)
            append_log(log_file, f"downloaded path={relative_path} bytes={len(response.content)} to={local_path}")
        except Exception as exc:  # pragma: no cover
            append_log(log_file, f"download error path={relative_path} error={repr(exc)}")


def main() -> int:
    args = parse_args()
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")

    headers = {"Authorization": f"Bearer {api_key}"}
    append_log(args.log_file, f"monitor started pod={args.pod_id} base={args.base_url}")

    while True:
        try:
            health = requests.get(args.base_url + "/health", timeout=20)
            status = health.json() if health.status_code == 200 else {"status_code": health.status_code, "text": health.text[:500]}
            append_log(args.log_file, f"health={json.dumps(status, ensure_ascii=False)}")
            if isinstance(status, dict) and status.get("status") in {"succeeded", "failed"}:
                download_artifacts(args.base_url, args.artifact_path, args.download_dir, args.log_file)
                stop = requests.post(f"https://rest.runpod.io/v1/pods/{args.pod_id}/stop", headers=headers, timeout=120)
                append_log(args.log_file, f"stop status={stop.status_code} body={stop.text[:500]}")
                return 0
        except Exception as exc:  # pragma: no cover
            append_log(args.log_file, f"monitor error={repr(exc)}")
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
