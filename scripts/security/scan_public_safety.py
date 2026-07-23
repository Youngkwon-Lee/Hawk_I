from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SECRET_PATTERNS = [
    ("openai_api_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{20,}")),
    ("github_token", re.compile(r"(?:github_pat_|gh[poasu]_)[A-Za-z0-9_]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
]

BLOCKED_MEDIA = re.compile(
    r"^(?:frontend/public/videos|demo_videos)/.*\.(?:mp4|mov|avi|webm|mkv|jpg|jpeg|png)$",
    re.IGNORECASE,
)

BINARY_SUFFIXES = {
    ".avi",
    ".gif",
    ".h5",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp4",
    ".onnx",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".svg",
    ".webm",
}


def git_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    seen = set()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8")
        if rel in seen:
            continue
        seen.add(rel)
        paths.append(Path(rel))
    return paths


def is_blocked_env_path(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".env.example"):
        return False
    return name == ".env" or name.endswith(".env") or ".env." in name


def scan_text(path: Path, failures: list[str]) -> None:
    full_path = ROOT / path
    if path.suffix.lower() in BINARY_SUFFIXES:
        return
    if full_path.stat().st_size > 2 * 1024 * 1024:
        return
    try:
        data = full_path.read_bytes()
    except OSError as error:
        failures.append(f"{path}: could not read file: {error}")
        return
    if b"\0" in data:
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    for line_number, line in enumerate(text.splitlines(), start=1):
        for name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                failures.append(f"{path}:{line_number}: secret pattern detected: {name}")


def main() -> int:
    failures = []
    for path in git_files():
        normalized = path.as_posix()
        if is_blocked_env_path(path):
            failures.append(f"{path}: tracked environment file is not allowed")
        if BLOCKED_MEDIA.match(normalized):
            failures.append(f"{path}: tracked subject/demo media is not allowed")
        scan_text(path, failures)

    if failures:
        print("Public safety scan failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Public safety scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
