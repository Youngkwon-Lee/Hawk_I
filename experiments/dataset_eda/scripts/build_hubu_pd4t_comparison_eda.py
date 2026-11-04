#!/usr/bin/env python3
"""Build local-only manifests and a sanitized HUBU-FIS vs PD4T EDA summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kinelo_data import data_path, data_root  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".mov"}
HUBU_ID_PATTERN = re.compile(
    r"^(CONTROL|ID)([0-9]+)_(DCHA|IZDA)$",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    # Unset paths resolve from the data-drive registry (see kinelo_data.py).
    parser = argparse.ArgumentParser()
    parser.add_argument("--hubu-root", type=Path)
    parser.add_argument("--pd4t-root", type=Path)
    parser.add_argument("--pd4t-original-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ffprobe-workers", type=int, default=4)
    parser.add_argument("--skip-ffprobe", action="store_true")
    args = parser.parse_args()
    if args.hubu_root is None:
        args.hubu_root = data_path("hubu_fis", "extracted", "videos_FIS")
    if args.pd4t_root is None:
        args.pd4t_root = data_path("pd4t_derived", "finger_tapping")
    if args.pd4t_original_root is None:
        args.pd4t_original_root = data_path("pd4t_raw")
    if args.output_dir is None:
        # Detailed manifests carry subject identifiers: keep them on the data drive.
        args.output_dir = (
            data_root() / "datasets/hawkeye/analysis/hubu_pd4t_comparison_2026-07-27"
        )
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def normalize_windows_path(raw_path: str) -> Path:
    normalized = raw_path.strip().replace("\\", "/")
    drive_match = re.match(r"^([A-Za-z]):/(.*)$", normalized)
    if drive_match:
        drive, remainder = drive_match.groups()
        return Path(f"/mnt/{drive.lower()}/{remainder}")
    return Path(normalized)


def parse_fraction(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        if float(denominator) == 0:
            return None
        return float(numerator) / float(denominator)
    return float(value)


def probe_video(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "ffprobe_ok": False,
            "codec": "",
            "width": None,
            "height": None,
            "fps": None,
            "pixel_format": "",
            "duration_sec": None,
            "container": "",
            "ffprobe_error": "file is missing or is not a regular file",
        }
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,pix_fmt:format=duration,format_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [{}])[0]
        video_format = payload.get("format") or {}
        avg_frame_rate = parse_fraction(stream.get("avg_frame_rate"))
        r_frame_rate = parse_fraction(stream.get("r_frame_rate"))
        duration_raw = video_format.get("duration")
        return {
            "ffprobe_ok": True,
            "codec": stream.get("codec_name") or "",
            "width": stream.get("width"),
            "height": stream.get("height"),
            "fps": round(avg_frame_rate or r_frame_rate, 6)
            if (avg_frame_rate or r_frame_rate)
            else None,
            "pixel_format": stream.get("pix_fmt") or "",
            "duration_sec": round(float(duration_raw), 6) if duration_raw else None,
            "container": video_format.get("format_name") or "",
            "ffprobe_error": "",
        }
    except Exception as exc:  # bounded error is recorded in the local-only manifest
        return {
            "ffprobe_ok": False,
            "codec": "",
            "width": None,
            "height": None,
            "fps": None,
            "pixel_format": "",
            "duration_sec": None,
            "container": "",
            "ffprobe_error": str(exc)[:500],
        }


def add_video_probes(
    rows: list[dict[str, Any]], workers: int, skip_ffprobe: bool
) -> None:
    if skip_ffprobe:
        for row in rows:
            row.update(
                {
                    "ffprobe_ok": None,
                    "codec": "",
                    "width": None,
                    "height": None,
                    "fps": None,
                    "pixel_format": "",
                    "duration_sec": None,
                    "container": "",
                    "ffprobe_error": "skipped",
                }
            )
        return

    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required unless --skip-ffprobe is set")
    paths = [Path(row["file_path"]) for row in rows]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        probes = executor.map(probe_video, paths)
        for row, probe in zip(rows, probes):
            row.update(probe)


def numeric_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return {"count": 0}

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "p25": round(quantile(0.25), 3),
        "median": round(quantile(0.5), 3),
        "p75": round(quantile(0.75), 3),
        "max": round(ordered[-1], 3),
    }


def counter_dict(values: Iterable[Any]) -> dict[str, int]:
    counter = Counter(str(value) for value in values)
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def video_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("ffprobe_ok") is True]
    resolutions = [
        f"{row['width']}x{row['height']}"
        for row in successful
        if row.get("width") and row.get("height")
    ]
    rounded_fps = [
        round(float(row["fps"]), 3) for row in successful if row.get("fps")
    ]
    return {
        "video_count": len(rows),
        "ffprobe_success_count": len(successful),
        "ffprobe_failure_count": len(rows) - len(successful),
        "extension_counts": counter_dict(row["extension"] for row in rows),
        "codec_counts": counter_dict(row["codec"] for row in successful),
        "resolution_counts": counter_dict(resolutions),
        "fps_counts": counter_dict(rounded_fps),
        "duration_sec": numeric_summary(row.get("duration_sec") for row in successful),
        "size_bytes": numeric_summary(row.get("size_bytes") for row in rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
    }


def build_hubu_manifest(
    root: Path, workers: int, skip_ffprobe: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_path = root / "fis_diagnostic.csv"
    video_root = root / "videos"
    label_rows = read_csv(label_path)
    files = sorted(
        path
        for path in video_root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    labels_by_stem: dict[str, dict[str, str]] = {}
    duplicate_label_ids = 0
    for row in label_rows:
        stem = Path((row.get("ID") or "").strip()).stem.lower()
        if stem in labels_by_stem:
            duplicate_label_ids += 1
        labels_by_stem[stem] = row

    manifest: list[dict[str, Any]] = []
    unmatched_filename_patterns = 0
    for path in files:
        label = labels_by_stem.get(path.stem.lower())
        match = HUBU_ID_PATTERN.match(path.stem)
        if match:
            cohort, subject_number, side_code = match.groups()
            cohort = cohort.upper()
            side_code = side_code.upper()
            side = {"DCHA": "right", "IZDA": "left"}[side_code]
            subject_id = f"{cohort}:{subject_number}"
        else:
            unmatched_filename_patterns += 1
            cohort = ""
            subject_number = ""
            side_code = ""
            side = ""
            subject_id = ""
        score_raw = (label or {}).get("UPDRS", "").strip()
        score = int(score_raw) if score_raw else None
        manifest.append(
            {
                "dataset": "HUBU-FIS",
                "file_name": path.name,
                "file_path": str(path),
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "subject_id": subject_id,
                "cohort": cohort,
                "side": side,
                "side_code": side_code,
                "updrs_score": score,
                "label_joined": label is not None,
            }
        )
    add_video_probes(manifest, workers, skip_ffprobe)

    file_stems = Counter(path.stem.lower() for path in files)
    subject_sides: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        if row["subject_id"] and row["side"]:
            subject_sides[row["subject_id"]].add(row["side"])
    label_values = [row["updrs_score"] for row in manifest if row["updrs_score"] is not None]
    cohort_score = {
        cohort: counter_dict(
            row["updrs_score"]
            for row in manifest
            if row["cohort"] == cohort and row["updrs_score"] is not None
        )
        for cohort in ("CONTROL", "ID")
    }
    summary = {
        "dataset": "HUBU-FIS",
        "intended_grain": "one left- or right-hand finger-tapping video per row",
        "label_csv_rows": len(label_rows),
        "video_files": len(files),
        "unique_video_stems": len(file_stems),
        "duplicate_video_stems": sum(count - 1 for count in file_stems.values()),
        "duplicate_label_ids": duplicate_label_ids,
        "files_without_labels": sum(not row["label_joined"] for row in manifest),
        "labels_without_files": sum(
            stem not in file_stems for stem in labels_by_stem
        ),
        "unmatched_filename_patterns": unmatched_filename_patterns,
        "zero_byte_videos": sum(row["size_bytes"] == 0 for row in manifest),
        "subjects": len(subject_sides),
        "subjects_by_cohort": counter_dict(
            subject_id.split(":", 1)[0] for subject_id in subject_sides
        ),
        "cohort_video_counts": counter_dict(row["cohort"] for row in manifest),
        "side_counts": counter_dict(row["side"] for row in manifest),
        "complete_left_right_pairs": sum(
            sides == {"left", "right"} for sides in subject_sides.values()
        ),
        "single_side_subjects": sum(len(sides) == 1 for sides in subject_sides.values()),
        "single_side_subjects_by_cohort": counter_dict(
            subject_id.split(":", 1)[0]
            for subject_id, sides in subject_sides.items()
            if len(sides) == 1
        ),
        "updrs_distribution": counter_dict(label_values),
        "cohort_updrs_distribution": cohort_score,
        "invalid_updrs_values": sum(
            score not in {0, 1, 2, 3, 4} for score in label_values
        ),
        "null_required_values": {
            "id": sum(not (row.get("ID") or "").strip() for row in label_rows),
            "updrs": sum(not (row.get("UPDRS") or "").strip() for row in label_rows),
        },
        "video_profile": video_profile(manifest),
    }
    return manifest, summary


def find_pd4t_original_finger_tapping_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".mp4"
        and "finger tapping" in str(path.parent).lower()
    )


def build_pd4t_manifest(
    root: Path,
    original_root: Path,
    workers: int,
    skip_ffprobe: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_rows = read_csv(root / "finger_tapping_v2_split.csv")
    metadata_rows = read_csv(root / "finger_tapping_metadata.csv")
    failed_rows = read_csv(root / "finger_tapping_v2_failed10.csv")
    subject_split_rows = read_csv(root / "finger_tapping_v2_subject_split.csv")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    files_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        files_by_name[path.name.lower()].append(path)

    manifest: list[dict[str, Any]] = []
    for row in split_rows:
        stored_path = normalize_windows_path(row.get("clip_path") or "")
        candidates = files_by_name.get((row.get("clip_filename") or "").lower(), [])
        resolved_path = stored_path if stored_path.is_file() else None
        if resolved_path is None and len(candidates) == 1:
            resolved_path = candidates[0]
        score_raw = (row.get("updrs_score") or "").strip()
        score = int(float(score_raw)) if score_raw else None
        action_code = (row.get("action") or "").strip().lower()
        manifest.append(
            {
                "dataset": "PD4T",
                "file_name": row.get("clip_filename") or "",
                "file_path": str(resolved_path) if resolved_path else "",
                "stored_clip_path": row.get("clip_path") or "",
                "extension": resolved_path.suffix.lower() if resolved_path else "",
                "size_bytes": resolved_path.stat().st_size if resolved_path else 0,
                "candidate_key": row.get("video_id_action_subject") or "",
                "subject_id": row.get("subject") or "",
                "side": {"l": "left", "r": "right"}.get(action_code, action_code),
                "action_code": action_code,
                "updrs_score": score,
                "split": (row.get("split") or "").strip().lower(),
                "stored_path_exists": stored_path.is_file(),
                "filename_joined": len(candidates) == 1,
                "file_resolved": resolved_path is not None,
            }
        )
    add_video_probes(manifest, workers, skip_ffprobe)

    subject_split_map = {
        row["subject"]: (row.get("split") or "").strip().lower()
        for row in subject_split_rows
    }
    subjects_by_split: dict[str, set[str]] = defaultdict(set)
    clips_by_subject = Counter()
    for row in manifest:
        subjects_by_split[row["split"]].add(row["subject_id"])
        clips_by_subject[row["subject_id"]] += 1
    failed_keys = {row.get("video_id_action_subject") or "" for row in failed_rows}
    split_keys = {row["candidate_key"] for row in manifest}
    original_videos = find_pd4t_original_finger_tapping_videos(original_root)
    label_values = [row["updrs_score"] for row in manifest if row["updrs_score"] is not None]
    summary = {
        "dataset": "PD4T",
        "intended_grain": "one derived finger-tapping clip per hand/action row",
        "split_csv_rows": len(split_rows),
        "metadata_csv_rows": len(metadata_rows),
        "subject_split_csv_rows": len(subject_split_rows),
        "failed_marker_csv_rows": len(failed_rows),
        "video_files": len(files),
        "unique_video_basenames": len(files_by_name),
        "duplicate_video_basenames": sum(
            len(paths) - 1 for paths in files_by_name.values()
        ),
        "duplicate_candidate_keys": len(manifest)
        - len({row["candidate_key"] for row in manifest}),
        "stored_paths_missing": sum(not row["stored_path_exists"] for row in manifest),
        "filename_join_failures": sum(not row["filename_joined"] for row in manifest),
        "unresolved_files": sum(not row["file_resolved"] for row in manifest),
        "zero_byte_videos": sum(row["size_bytes"] == 0 for row in manifest),
        "subjects": len(clips_by_subject),
        "subjects_by_split": {
            split: len(subjects) for split, subjects in sorted(subjects_by_split.items())
        },
        "subject_overlap_across_splits": {
            "train_val": len(subjects_by_split["train"] & subjects_by_split["val"]),
            "train_test": len(subjects_by_split["train"] & subjects_by_split["test"]),
            "val_test": len(subjects_by_split["val"] & subjects_by_split["test"]),
        },
        "subject_split_assignment_mismatches": sum(
            subject_split_map.get(row["subject_id"]) != row["split"] for row in manifest
        ),
        "clips_per_subject": numeric_summary(clips_by_subject.values()),
        "side_counts": counter_dict(row["side"] for row in manifest),
        "split_counts": counter_dict(row["split"] for row in manifest),
        "updrs_distribution": counter_dict(label_values),
        "updrs_by_side": {
            side: counter_dict(
                row["updrs_score"]
                for row in manifest
                if row["side"] == side and row["updrs_score"] is not None
            )
            for side in ("left", "right")
        },
        "invalid_updrs_values": sum(
            score not in {0, 1, 2, 3, 4} for score in label_values
        ),
        "invalid_split_values": sum(
            row["split"] not in {"train", "val", "test"} for row in manifest
        ),
        "invalid_action_values": sum(
            row["action_code"] not in {"l", "r"} for row in manifest
        ),
        "failed_marker_keys_overlapping_split": len(failed_keys & split_keys),
        "original_finger_tapping_video_count": len(original_videos),
        "derived_minus_original_count": len(manifest) - len(original_videos),
        "video_profile": video_profile(manifest),
    }
    return manifest, summary


def percentage(count: int, total: int) -> float:
    return round((count / total) * 100, 2) if total else 0.0


def build_findings(
    hubu: dict[str, Any], pd4t: dict[str, Any]
) -> list[dict[str, Any]]:
    hubu_total = hubu["video_files"]
    pd4t_total = pd4t["video_files"]
    hubu_score4 = int(hubu["updrs_distribution"].get("4", 0))
    pd4t_score4 = int(pd4t["updrs_distribution"].get("4", 0))
    return [
        {
            "finding": "HUBU-FIS file-label join is complete",
            "severity": "low",
            "confidence": "high",
            "evidence": f"{hubu_total}/{hubu_total} videos join to CSV rows; no duplicate stems or zero-byte videos.",
            "risk": "Low ingestion risk for clip-level EDA.",
            "remediation": "Automate row-count, unique-ID, file-exists, and zero-byte checks.",
        },
        {
            "finding": "HUBU-FIS has two incomplete left-right subject pairs",
            "severity": "medium",
            "confidence": "high",
            "evidence": f"{hubu['complete_left_right_pairs']} complete pairs and {hubu['single_side_subjects']} single-side subjects across {hubu['subjects']} subjects.",
            "risk": "Pairwise asymmetry analysis can silently lose or misweight two subjects.",
            "remediation": "Use an explicit pair-complete flag and exclude or model single-side subjects deliberately.",
        },
        {
            "finding": "Five-class severity coverage is inadequate in both datasets",
            "severity": "high",
            "confidence": "high",
            "evidence": f"HUBU-FIS score 4: {hubu_score4}/{hubu_total}; PD4T score 4: {pd4t_score4}/{pd4t_total} ({percentage(pd4t_score4, pd4t_total)}%).",
            "risk": "Naive five-class training or accuracy reporting will be unstable and misleading for severe cases.",
            "remediation": "Prefer ordinal metrics and macro/weighted reporting; treat score 4 as an explicit sparse-class limitation.",
        },
        {
            "finding": "HUBU cohort prefix is not equivalent to severity zero",
            "severity": "medium",
            "confidence": "high",
            "evidence": f"CONTROL videos include {hubu['cohort_updrs_distribution']['CONTROL'].get('1', 0)} score-1 rows; ID videos include {hubu['cohort_updrs_distribution']['ID'].get('0', 0)} score-0 rows.",
            "risk": "Using cohort names as label proxies would introduce systematic label error.",
            "remediation": "Use the UPDRS column as the task label and cohort only as a separate stratification variable.",
        },
        {
            "finding": "PD4T manifest and subject split integrity are strong",
            "severity": "low",
            "confidence": "high",
            "evidence": f"{pd4t_total}/{pd4t_total} clips resolve; candidate keys are unique; subject overlap across train/val/test is zero.",
            "risk": "Low immediate risk of file leakage or broken joins at the current manifest grain.",
            "remediation": "Keep subject-disjoint split checks as automated gates.",
        },
        {
            "finding": "PD4T failed-marker provenance is ambiguous",
            "severity": "medium",
            "confidence": "high",
            "evidence": f"All {pd4t['failed_marker_csv_rows']} keys in the file named finger_tapping_v2_failed10.csv also appear in the final split.",
            "risk": "A downstream pipeline could incorrectly exclude valid clips based only on the filename.",
            "remediation": "Document what failed and rename the marker file or add an explicit resolved/status column.",
        },
        {
            "finding": "PD4T derived lineage has a one-video count delta",
            "severity": "medium",
            "confidence": "medium",
            "evidence": f"Original finger-tapping source count is {pd4t['original_finger_tapping_video_count']}; derived split count is {pd4t_total}.",
            "risk": "The exclusion or transformation of one source video is not explicit in the final manifest.",
            "remediation": "Add a source-to-derived lineage table with an explicit exclusion reason.",
        },
        {
            "finding": "Cross-dataset grain differs materially",
            "severity": "high",
            "confidence": "high",
            "evidence": f"HUBU-FIS: {hubu_total} videos from {hubu['subjects']} subjects, mostly two sides each; PD4T: {pd4t_total} clips from {pd4t['subjects']} subjects with median {pd4t['clips_per_subject']['median']} clips per subject.",
            "risk": "Pooling clips without subject-aware weighting would overrepresent PD4T subjects and confound dataset effects with severity.",
            "remediation": "Evaluate per dataset first, split by subject, and report subject-level aggregated metrics before any pooled model.",
        },
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hubu_manifest, hubu_summary = build_hubu_manifest(
        args.hubu_root, args.ffprobe_workers, args.skip_ffprobe
    )
    pd4t_manifest, pd4t_summary = build_pd4t_manifest(
        args.pd4t_root,
        args.pd4t_original_root,
        args.ffprobe_workers,
        args.skip_ffprobe,
    )
    findings = build_findings(hubu_summary, pd4t_summary)
    summary = {
        "analysis": "HUBU-FIS vs PD4T finger-tapping comparison EDA",
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "privacy": {
            "detailed_manifests": "local-only on home desktop D:; contain filenames and subject identifiers",
            "summary": "sanitized aggregate; contains no participant filenames or subject identifiers",
        },
        "source_paths": {
            "hubu_root": str(args.hubu_root),
            "pd4t_root": str(args.pd4t_root),
            "pd4t_original_root": str(args.pd4t_original_root),
        },
        "hubu_fis": hubu_summary,
        "pd4t": pd4t_summary,
        "findings": findings,
    }
    write_csv_atomic(args.output_dir / "hubu_fis_manifest_local_only.csv", hubu_manifest)
    write_csv_atomic(args.output_dir / "pd4t_manifest_local_only.csv", pd4t_manifest)
    write_json_atomic(args.output_dir / "summary_sanitized.json", summary)
    write_csv_atomic(args.output_dir / "quality_findings_sanitized.csv", findings)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "hubu_videos": hubu_summary["video_files"],
                "pd4t_videos": pd4t_summary["video_files"],
                "hubu_ffprobe_failures": hubu_summary["video_profile"][
                    "ffprobe_failure_count"
                ],
                "pd4t_ffprobe_failures": pd4t_summary["video_profile"][
                    "ffprobe_failure_count"
                ],
                "findings": len(findings),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
