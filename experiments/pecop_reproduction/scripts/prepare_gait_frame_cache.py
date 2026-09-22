#!/usr/bin/env python3
"""Prepare a deterministic 103-frame PD4T Gait cache for CoRe reproduction.

The cache is local-only and must stay outside Git. It contains sampled image
frames derived from protected PD4T videos; treat it with the same access
controls as the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

_SUBJECT_RE = re.compile(r"_([0-9]{3})$")


def parse_subject(annotation_id: str) -> str:
    match = _SUBJECT_RE.search(str(annotation_id).strip())
    if not match:
        raise ValueError(f"invalid PD4T annotation id: {annotation_id!r}")
    return match.group(1)


def video_stem(annotation_id: str) -> str:
    value = str(annotation_id).strip()
    parse_subject(value)
    return value.rsplit("_", 1)[0]


def read_ids(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 3:
                raise ValueError(f"{path}:{line_no}: expected >=3 columns")
            annotation_id = row[0].strip()
            parse_subject(annotation_id)
            ids.append(annotation_id)
    return ids


def uniform_indices(frame_count: int, length: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return np.rint(np.linspace(0, frame_count - 1, num=length)).astype(np.int64)


def extract_one(video_path: Path, out_dir: Path, length: int) -> int:
    expected = [out_dir / f"img_{i:05d}.jpg" for i in range(length)]
    if all(path.exists() for path in expected):
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"invalid frame count: {video_path}")

    wanted = uniform_indices(frame_count, length)
    frames = []
    for frame_idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed reading frame {frame_idx}: {video_path}")
        frames.append(frame)
    cap.release()

    for i, frame in enumerate(frames):
        dest = out_dir / f"img_{i:05d}.jpg"
        if not cv2.imwrite(str(dest), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"failed writing {dest}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("PD4T_ROOT", "")))
    parser.add_argument("--frame-root", type=Path, default=Path(os.environ.get("PD4T_FRAME_ROOT", "experiments/pecop_reproduction/results/runtime/gait_frames")))
    parser.add_argument("--length", type=int, default=103)
    args = parser.parse_args()

    if not str(args.data_root):
        raise RuntimeError("PD4T_ROOT or --data-root is required")

    annotation_dir = args.data_root / "Annotations" / "Gait"
    ids = read_ids(annotation_dir / "train.csv") + read_ids(annotation_dir / "test.csv")
    if len(ids) != 426:
        raise RuntimeError(f"expected 426 gait videos, found {len(ids)}")

    created = 0
    for n, annotation_id in enumerate(ids, start=1):
        subject = parse_subject(annotation_id)
        stem = video_stem(annotation_id)
        video_path = args.data_root / "Videos" / "Gait" / subject / f"{stem}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        out_dir = args.frame_root / subject / stem
        created += extract_one(video_path, out_dir, args.length)
        if n % 25 == 0 or n == len(ids):
            print(f"prepared {n}/{len(ids)} videos")

    manifest = {
        "task": "Gait",
        "videos": len(ids),
        "frames_per_video": args.length,
        "sampling": "uniform_full_video_v0",
        "jpeg_quality": 95,
        "newly_created_videos": created,
    }
    args.frame_root.mkdir(parents=True, exist_ok=True)
    (args.frame_root / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
