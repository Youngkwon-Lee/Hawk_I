#!/usr/bin/env python3
"""Benchmark a vision-language model on freezing-of-gait timing.

Freezing is too rare here to train on (28 positive clips of 426) but that is
enough to benchmark, and freezing spans are real durations - median 6.3s - so
temporal overlap actually means something.

    # see what would be sent, without sending anything
    python backend/scripts/benchmark_fog_detection.py --clips clips.jsonl --model gpt-4o

    # actually call the API (see the data-use note below)
    python backend/scripts/benchmark_fog_detection.py --clips clips.jsonl --model gpt-4o \\
        --api-base https://api.openai.com/v1 --send --data-use-approved

Then grade the output with the existing evaluator:

    python backend/scripts/evaluate_primitive_predictions.py \\
        --labels labels_gait.jsonl --predictions fog_preds.jsonl

DATA USE
--------
Sending clip frames to a hosted model discloses study video to a third party.
--send is refused unless --data-use-approved is also given, so that disclosure
is a deliberate act. Confirm the dataset's terms permit it first. Pointing
--api-base at a locally hosted OpenAI-compatible server keeps the video on your
own hardware and needs no such approval.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.fog_benchmark import (  # noqa: E402
    build_prediction_record,
    build_prompt,
    parse_freezing_response,
    require_data_use_approval,
)


def sample_frames(video_path: str, max_frames: int) -> tuple[list[str], float]:
    """Uniformly sample frames as base64 JPEGs, and return the clip duration."""
    import cv2  # imported here so --dry-run works without a full CV stack

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps if fps else 0.0
    if total <= 0:
        capture.release()
        raise ValueError(f"video reports no frames: {video_path}")

    step = max(1, total // max_frames)
    frames: list[str] = []
    for index in range(0, total, step):
        if len(frames) >= max_frames:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            frames.append(base64.b64encode(buffer).decode("utf-8"))
    capture.release()
    return frames, duration


def ask_model(api_base: str, api_key: str, model: str, prompt: str, frames: list[str], timeout: float) -> str:
    import requests

    content: list[dict] = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})

    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, required=True,
                        help="JSONL with clip_id and a local video path (media_path / source_clip_path)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=Path("fog_preds.jsonl"))
    parser.add_argument("--api-base", default=os.getenv("FOG_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--send", action="store_true", help="actually call the model (default is a dry run)")
    parser.add_argument("--data-use-approved", action="store_true",
                        help="confirm the dataset's terms allow sending video to this endpoint")
    args = parser.parse_args(argv)

    if not args.clips.exists():
        print(f"error: {args.clips} not found", file=sys.stderr)
        return 2

    clips = []
    for line in args.clips.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("clip_id"):
            clips.append(record)
    if args.limit:
        clips = clips[: args.limit]

    local_endpoint = any(host in args.api_base for host in ("localhost", "127.0.0.1", "0.0.0.0"))
    print(f"clips {len(clips)} | model {args.model} | endpoint {args.api_base}")

    if not args.send:
        prompt = build_prompt(duration_sec=20.0, n_frames=args.max_frames)
        print("\n-- DRY RUN: nothing is sent. Prompt that would be used --")
        print(prompt)
        print(f"\n{len(clips)} clip(s) x up to {args.max_frames} frames would be uploaded to {args.api_base}.")
        if not local_endpoint:
            print("This endpoint is remote: study video would leave your machine.")
        print("Re-run with --send (and --data-use-approved for a remote endpoint) to proceed.")
        return 0

    if not local_endpoint:
        try:
            require_data_use_approval(args.data_use_approved, args.api_base)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    api_key = os.getenv(args.api_key_env, "")
    if not api_key and not local_endpoint:
        print(f"error: {args.api_key_env} is not set", file=sys.stderr)
        return 2

    written = failed = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for clip in clips:
            clip_id = str(clip["clip_id"])
            path = clip.get("media_path") or clip.get("source_clip_path") or clip.get("file_path")
            try:
                frames, duration = sample_frames(str(path), args.max_frames)
                reply = ask_model(
                    args.api_base, api_key, args.model,
                    build_prompt(duration, len(frames)), frames, args.timeout,
                )
                intervals = parse_freezing_response(reply, duration_sec=duration)
                record = build_prediction_record(
                    clip_id, intervals, model=args.model,
                    dataset=clip.get("dataset", "PD4T"), split=clip.get("split"),
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
                print(f"  {clip_id}: {len(intervals)} interval(s)")
            except Exception as exc:  # one bad clip must not end the run
                failed += 1
                print(f"  FAIL {clip_id}: {exc}", file=sys.stderr)

    print(f"\nwrote {written} prediction(s) to {args.out}; {failed} failed")
    print("grade with: backend/scripts/evaluate_primitive_predictions.py --labels <labels> --predictions "
          f"{args.out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
