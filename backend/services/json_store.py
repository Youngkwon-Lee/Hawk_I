"""Durable JSON persistence helpers for analysis state and results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
) -> Path:
    """Write JSON without exposing a partially written destination file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=ensure_ascii,
                indent=indent,
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None

        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support directory fsync. The atomic
            # replace still prevents readers from observing partial JSON.
            pass

        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
