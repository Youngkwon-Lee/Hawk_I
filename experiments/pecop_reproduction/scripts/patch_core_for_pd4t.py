#!/usr/bin/env python3
"""Patch the pinned CoRe checkout with the isolated PD4T Gait adapter.

The patcher is exact-string based. If upstream content differs from the pinned
revision, it fails instead of guessing.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_exact(path: Path, old: str, new: str, expected_count=None) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if expected_count is not None and count != expected_count:
        raise RuntimeError(f"{path}: expected {expected_count} occurrences, found {count}: {old!r}")
    if count == 0:
        raise RuntimeError(f"{path}: patch anchor not found: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("core_dir", type=Path)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "core_adapter" / "PD4TPair.py",
    )
    args = parser.parse_args()

    core = args.core_dir.resolve()
    if not (core / ".git").exists():
        raise RuntimeError(f"not a git checkout: {core}")

    target = core / "datasets" / "PD4TPair.py"
    shutil.copy2(args.adapter, target)

    init_path = core / "datasets" / "__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    import_line = "from .PD4TPair import PD4T_Dataset as PD4T\n"
    if import_line not in init_text:
        init_path.write_text(init_text.rstrip() + "\n" + import_line, encoding="utf-8")

    replace_exact(
        core / "utils" / "parser.py",
        "choices=[\'MTL\', \'Seven\']",
        "choices=[\'MTL\', \'Seven\', \'PD4T\']",
        expected_count=1,
    )

    replace_exact(
        core / "tools" / "runner.py",
        "elif args.benchmark == \'Seven\':",
        "elif args.benchmark in (\'Seven\', \'PD4T\'):",
        expected_count=3,
    )
    replace_exact(
        core / "tools" / "helper.py",
        "elif args.benchmark == \'Seven\':",
        "elif args.benchmark in (\'Seven\', \'PD4T\'):",
        expected_count=2,
    )

    config_src = Path(__file__).resolve().parent.parent / "core_adapter" / "PD4T_CoRe.yaml"
    shutil.copy2(config_src, core / "configs" / "PD4T_CoRe.yaml")

    print(f"patched CoRe for PD4T Gait: {core}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
