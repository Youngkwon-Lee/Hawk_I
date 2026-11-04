"""Resolve dataset locations from the shared data-drive registry.

The data drive (home desktop D:, WSL /mnt/d) keeps a machine-readable
``data_registry.json`` at its root. Code asks for datasets by logical name
instead of hardcoding drive paths, so moving a folder only means editing the
registry. Override the root with ``KINELO_DATA_ROOT``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT_ENV = "KINELO_DATA_ROOT"
REGISTRY_FILENAME = "data_registry.json"
DEFAULT_ROOTS = (Path("/mnt/d"), Path("D:/"))


class DatasetNotLocalError(LookupError):
    """The dataset is registered but not present on the data drive."""


def data_root() -> Path:
    configured = os.environ.get(ROOT_ENV)
    if configured:
        return Path(configured)
    for candidate in DEFAULT_ROOTS:
        if (candidate / REGISTRY_FILENAME).is_file():
            return candidate
    raise FileNotFoundError(
        f"{ROOT_ENV} is not set and no {REGISTRY_FILENAME} was found at "
        + " or ".join(str(root) for root in DEFAULT_ROOTS)
    )


def load_registry(root: Path | None = None) -> dict[str, Any]:
    registry_path = (root or data_root()) / REGISTRY_FILENAME
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def data_path(name: str, *parts: str, root: Path | None = None) -> Path:
    """Return the local path of a registered dataset, optionally joined with parts.

    Raises KeyError for unknown names and DatasetNotLocalError for datasets that
    are registered elsewhere (for example still only in the Google Drive backup),
    so callers fail loudly instead of reading a path that does not exist.
    """
    root = Path(root) if root is not None else data_root()
    datasets = load_registry(root)["datasets"]
    if name not in datasets:
        raise KeyError(
            f"unknown dataset {name!r}; registered: {', '.join(sorted(datasets))}"
        )
    entry = datasets[name]
    if entry.get("kind") != "local":
        raise DatasetNotLocalError(
            f"dataset {name!r} is not on the data drive "
            f"(kind={entry.get('kind')}, status={entry.get('status')}); "
            "restore it first or pass an explicit path"
        )
    return root.joinpath(entry["path"], *parts)
