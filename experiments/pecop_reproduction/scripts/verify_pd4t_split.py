#!/usr/bin/env python3
"""Verify the original PD4T train/test split for PECoP reproduction.

This script intentionally uses the original PD4T annotation CSVs and must not
be reused for Hawkeye production training. It validates the paper-comparison
boundary without exporting participant identifiers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

DEFAULT_ROOT = Path(os.environ.get("PD4T_ROOT", "data/raw/PD4T/PD4T/PD4T"))


def parse_subject(annotation_id: str) -> str:
    """Return the de-identified subject suffix from a PD4T annotation id."""
    value = str(annotation_id).strip()
    match = re.search(r"_([0-9]{3})$", value)
    if not match:
        raise ValueError(f"cannot parse PD4T subject suffix from {annotation_id!r}")
    return match.group(1)


def read_annotation(path: Path) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 3:
                raise ValueError(f"{path}:{line_no}: expected >=3 columns, got {len(row)}")
            annotation_id = row[0].strip()
            try:
                frame_count = int(float(row[1]))
                score = float(row[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid frame_count/score") from exc
            parse_subject(annotation_id)
            rows.append((annotation_id, frame_count, score))
    return rows


def subject_set(rows: Iterable[tuple[str, int, float]]) -> set[str]:
    return {parse_subject(row[0]) for row in rows}


def digest_subjects(subjects: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(subjects))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_self_test() -> None:
    cases = {
        "15-005087_l_042": "042",
        "15-005097_r_042": "042",
        "15-001760_009": "009",
    }
    for value, expected in cases.items():
        actual = parse_subject(value)
        assert actual == expected, (value, actual, expected)
    for invalid in ("15-005087_l", "005087", ""):
        try:
            parse_subject(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected parse failure for {invalid!r}")
    print("self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task", default="Gait")
    parser.add_argument("--expected-train-subjects", type=int, default=22)
    parser.add_argument("--expected-test-subjects", type=int, default=8)
    parser.add_argument("--expected-total-videos", type=int, default=426)
    parser.add_argument("--expected-test-videos", type=int, default=116)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--show-subjects", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    annotation_dir = args.data_root / "Annotations" / args.task
    train_path = annotation_dir / "train.csv"
    test_path = annotation_dir / "test.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(path)

    train_rows = read_annotation(train_path)
    test_rows = read_annotation(test_path)
    train_subjects = subject_set(train_rows)
    test_subjects = subject_set(test_rows)
    overlap = train_subjects & test_subjects

    checks = {
        "train_subject_count": len(train_subjects) == args.expected_train_subjects,
        "test_subject_count": len(test_subjects) == args.expected_test_subjects,
        "subject_overlap_zero": len(overlap) == 0,
        "total_video_count": len(train_rows) + len(test_rows) == args.expected_total_videos,
        "test_video_count": len(test_rows) == args.expected_test_videos,
    }

    manifest = {
        "task": args.task,
        "annotation_dir": str(annotation_dir),
        "train": {
            "videos": len(train_rows),
            "subjects": len(train_subjects),
            "subject_digest_sha256": digest_subjects(train_subjects),
            "csv_sha256": sha256_file(train_path),
        },
        "test": {
            "videos": len(test_rows),
            "subjects": len(test_subjects),
            "subject_digest_sha256": digest_subjects(test_subjects),
            "csv_sha256": sha256_file(test_path),
        },
        "subject_overlap_count": len(overlap),
        "checks": checks,
        "pass": all(checks.values()),
    }

    if args.show_subjects:
        manifest["train"]["subject_ids"] = sorted(train_subjects)
        manifest["test"]["subject_ids"] = sorted(test_subjects)
        manifest["overlap_subject_ids"] = sorted(overlap)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if manifest["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
