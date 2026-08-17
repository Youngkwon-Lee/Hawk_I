"""Research-only GPT-5.6 Terra observations for Hawk I.

GPT-5.6 Terra accepts images, not video.  This service samples a small,
ordered set of frames from an already-authorized research video and submits
only those frames to the Responses API.  It is deliberately independent from
the baseline and fine-tuned scoring paths.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OBSERVATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "task_confirmed",
        "speed",
        "amplitude",
        "rhythm",
        "pauses_or_hesitations",
        "laterality_or_asymmetry",
        "overall_observation",
        "limitations",
    ],
    "properties": {
        "task_confirmed": {"type": "boolean"},
        "speed": {"type": "string"},
        "amplitude": {"type": "string"},
        "rhythm": {"type": "string"},
        "pauses_or_hesitations": {"type": "string"},
        "laterality_or_asymmetry": {"type": "string"},
        "overall_observation": {"type": "string"},
        "limitations": {"type": "string"},
    },
}


@dataclass(frozen=True)
class GPT56TerraConfig:
    api_key: str
    model: str
    max_frames: int
    reasoning_effort: str


def get_config() -> GPT56TerraConfig | None:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        max_frames = int(os.getenv("GPT56_TERRA_MAX_FRAMES", "8"))
    except ValueError:
        max_frames = 8

    return GPT56TerraConfig(
        api_key=api_key,
        model=(os.getenv("GPT56_TERRA_MODEL") or "gpt-5.6-terra").strip(),
        max_frames=max(1, min(max_frames, 12)),
        reasoning_effort=(os.getenv("GPT56_TERRA_REASONING_EFFORT") or "none").strip(),
    )


def research_prompt(task_type: str, frame_count: int) -> str:
    task_label = {
        "finger_tapping": "finger-tapping",
        "hand_movement": "hand opening/closing",
        "leg_agility": "leg agility",
        "gait": "gait/walking",
    }.get(task_type, task_type or "movement")
    return f"""This is a de-identified public research video represented by {frame_count}
chronologically ordered still frames of a {task_label} task. This is movement-
observation research only, not a clinical diagnosis or clinical decision.

Describe only what is visible across the sampled frames: speed, amplitude,
rhythm, pauses/hesitations, and any observable asymmetry. Do not infer motion
between frames. Do not output an MDS-UPDRS score, a diagnosis, or treatment
advice. State sampling/visibility limits explicitly. Return the requested JSON
object only."""


def _parse_observation(text: str | None) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class GPT56TerraResearchVLM:
    """Frame-sampled, research-only GPT-5.6 Terra observation service."""

    def __init__(self, config: GPT56TerraConfig | None = None, client: Any = None):
        self.config = config or get_config()
        self.client = client
        if self.client is None and self.config is not None:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.config.api_key)
            except ImportError:
                self.client = None

    def is_available(self) -> bool:
        return self.config is not None and self.client is not None

    def status(self) -> dict[str, Any]:
        return {
            "available": self.is_available(),
            "model": self.config.model if self.config else None,
            "input_mode": "ordered_sampled_frames",
            "research_only": True,
            "external_processing_confirmation_required": True,
        }

    @staticmethod
    def _extract_frames(video_path: str, max_frames: int) -> list[Any]:
        import cv2

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return []
            frame_indices = (
                list(range(total_frames))
                if total_frames <= max_frames
                else [int(index * (total_frames - 1) / (max_frames - 1)) for index in range(max_frames)]
                if max_frames > 1
                else [0]
            )
            frames: list[Any] = []
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if success:
                    frames.append(cv2.resize(frame, (512, 512)))
            return frames
        finally:
            capture.release()

    @staticmethod
    def _encode_frame(frame: Any) -> str:
        import cv2

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Could not encode video frame")
        return base64.b64encode(buffer).decode("ascii")

    def analyze_video(self, video_path: str, task_type: str) -> dict[str, Any]:
        if not self.is_available():
            return {"success": False, "error": "GPT-5.6 Terra is not configured."}
        if not Path(video_path).is_file():
            return {"success": False, "error": "Analysis video is unavailable."}

        try:
            frames = self._extract_frames(video_path, self.config.max_frames)
            if not frames:
                return {"success": False, "error": "No usable video frames were extracted."}

            content: list[dict[str, Any]] = [
                {"type": "input_text", "text": research_prompt(task_type, len(frames))}
            ]
            for frame in frames:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{self._encode_frame(frame)}",
                        "detail": "low",
                    }
                )

            response = self.client.responses.create(
                model=self.config.model,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": self.config.reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "movement_observation",
                        "strict": True,
                        "schema": OBSERVATION_SCHEMA,
                    }
                },
            )
            observation = _parse_observation(getattr(response, "output_text", None))
            if observation is None:
                return {"success": False, "error": "GPT-5.6 Terra returned an unstructured observation."}
            return {
                "success": True,
                "provider": "openai",
                "model": self.config.model,
                "input_mode": "ordered_sampled_frames",
                "frames_analyzed": len(frames),
                "research_only": True,
                "observation": observation,
            }
        except Exception as exc:
            return {"success": False, "error": f"GPT-5.6 Terra analysis failed: {exc}"}
