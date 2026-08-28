#!/usr/bin/env python3
"""Validate and privacy-stage a gait handoff for Qwen3 C0B training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.c0b_training_data import C0BDataError, stage_handoff  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--drive-root", type=Path, default=Path("/mnt"))
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write staged JSONL and manifest without copying media (validation/debug only).",
    )
    args = parser.parse_args(argv)
    try:
        manifest = stage_handoff(
            args.export_dir,
            args.output_dir,
            expected_dataset_sha256=args.expected_dataset_sha256,
            drive_root=args.drive_root,
            copy_media=not args.metadata_only,
        )
    except C0BDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "stage_sha256": manifest["stage_sha256"],
                "source_dataset_sha256": manifest["source_dataset_sha256"],
                "prompt_contract_sha256": manifest["prompt_contract_sha256"],
                "splits": {
                    split: details["records"] for split, details in manifest["splits"].items()
                },
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
