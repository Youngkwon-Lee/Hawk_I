"""PD4T Gait adapter for the pinned CoRe implementation.

Copied into the private upstream CoRe checkout by patch_core_for_pd4t.py.
Reads protected PD4T annotations and a local-only sampled-frame cache.
"""

from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path

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


class PD4T_Dataset(torch.utils.data.Dataset):
    """CoRe-compatible PD4T Gait pair dataset.

    Temporal policy v0 uses a deterministic 103-frame cache uniformly sampled
    across each full source video. This is an explicit reproduction hypothesis
    because the PECoP paper does not publish downstream PD4T CoRe sampling.
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

        configured_frames = getattr(args, "pd4t_frame_root", None)
        frame_value = os.environ.get("PD4T_FRAME_ROOT", configured_frames or "")
        if not frame_value:
            raise RuntimeError("PD4T_FRAME_ROOT or config pd4t_frame_root is required")
        self.frame_root = Path(frame_value)

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

    def _frame_dir(self, row) -> Path:
        subject = parse_subject(row["annotation_id"])
        stem = video_stem(row["annotation_id"])
        return self.frame_root / subject / stem

    def _load_cached_clip(self, row):
        frame_dir = self._frame_dir(row)
        paths = sorted(frame_dir.glob("img_*.jpg"))
        if len(paths) != self.length:
            raise RuntimeError("expected %d cached frames in %s, found %d" % (self.length, frame_dir, len(paths)))
        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB").copy())
        return self.transforms(frames)

    def _pack(self, row):
        score = float(row["score"])
        return {
            "video": self._load_cached_clip(row),
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
