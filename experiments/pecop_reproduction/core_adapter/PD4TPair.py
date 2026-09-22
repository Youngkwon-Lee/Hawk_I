"""PD4T Gait adapter for the pinned CoRe implementation.

Copied into the private upstream CoRe checkout by patch_core_for_pd4t.py.
Reads local PD4T annotations/videos at runtime and writes no row-level data.
"""

from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_SUBJECT_RE = re.compile(r"_([0-9]{3})$")


def parse_subject(annotation_id: str) -> str:
    match = _SUBJECT_RE.search(str(annotation_id).strip())
    if not match:
        raise ValueError("invalid PD4T annotation id: %r" % annotation_id)
    return match.group(1)


def video_stem(annotation_id: str) -> str:
    value = str(annotation_id).strip()
    parse_subject(value)
    return value.rsplit("_", 1)[0]


def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 3:
                raise ValueError("%s:%d: expected >=3 columns" % (path, line_no))
            annotation_id = row[0].strip()
            parse_subject(annotation_id)
            rows.append({
                "annotation_id": annotation_id,
                "frame_count": int(float(row[1])),
                "score": float(row[2]),
            })
    return rows


def uniform_indices(frame_count: int, length: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return np.rint(np.linspace(0, frame_count - 1, num=length)).astype(np.int64)


class PD4T_Dataset(torch.utils.data.Dataset):
    """CoRe-compatible PD4T Gait pair dataset.

    Temporal policy v0 uniformly samples frame_length frames across each source
    video. This is an explicit reproduction hypothesis because the PECoP paper
    does not publish the downstream PD4T temporal sampling implementation.
    """

    def __init__(self, args, subset, transform):
        if getattr(args, "pd4t_task", "Gait") != "Gait":
            raise NotImplementedError("v0 adapter is intentionally Gait-only")
        self.args = args
        self.subset = subset
        self.transforms = transform
        self.length = int(args.frame_length)
        self.voter_number = int(args.voter_number)
        self.seed = int(args.seed)
        configured_root = getattr(args, "pd4t_root", None)
        root_value = os.environ.get("PD4T_ROOT", configured_root or "")
        if not root_value:
            raise RuntimeError("PD4T_ROOT or config pd4t_root is required")
        self.root = Path(root_value)
        self.task = "Gait"
        annotation_dir = self.root / "Annotations" / self.task
        self.train_rows = read_rows(annotation_dir / "train.csv")
        self.test_rows = read_rows(annotation_dir / "test.csv")
        self.dataset = self.test_rows if subset == "test" else self.train_rows
        train_subjects = {parse_subject(r["annotation_id"]) for r in self.train_rows}
        test_subjects = {parse_subject(r["annotation_id"]) for r in self.test_rows}
        overlap = train_subjects & test_subjects
        if overlap:
            raise RuntimeError("PD4T train/test subject overlap detected")
        expected_train = int(getattr(args, "expected_train_subjects", 22))
        expected_test = int(getattr(args, "expected_test_subjects", 8))
        if len(train_subjects) != expected_train or len(test_subjects) != expected_test:
            raise RuntimeError("unexpected PD4T subject counts: train=%d test=%d" % (len(train_subjects), len(test_subjects)))

    def _video_path(self, row) -> Path:
        subject = parse_subject(row["annotation_id"])
        stem = video_stem(row["annotation_id"])
        return self.root / "Videos" / self.task / subject / (stem + ".mp4")

    def _load_uniform_clip(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("failed to open video: %s" % path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise RuntimeError("invalid video frame count: %s" % path)
        frames = []
        for frame_idx in uniform_indices(frame_count, self.length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError("failed reading frame %d from %s" % (frame_idx, path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return self.transforms(frames)

    def _pack(self, row):
        score = float(row["score"])
        return {
            "video": self._load_uniform_clip(self._video_path(row)),
            "final_score": score,
            "difficulty": 1.0,
            "completeness": score,
        }

    def delta(self):
        scores = [float(row["score"]) for row in self.train_rows]
        return [abs(scores[i] - scores[j]) for i in range(len(scores)) for j in range(i + 1, len(scores))]

    def __getitem__(self, index):
        row = self.dataset[index]
        data = self._pack(row)
        rng = random.Random(self.seed + index + (100000 if self.subset == "test" else 0))
        if self.subset == "test":
            candidates = list(range(len(self.train_rows)))
            rng.shuffle(candidates)
            chosen = candidates[: min(self.voter_number, len(candidates))]
            return data, [self._pack(self.train_rows[i]) for i in chosen]
        candidates = list(range(len(self.train_rows)))
        if len(candidates) > 1:
            candidates.remove(index)
        target_idx = rng.choice(candidates)
        return data, self._pack(self.train_rows[target_idx])

    def __len__(self):
        return len(self.dataset)
