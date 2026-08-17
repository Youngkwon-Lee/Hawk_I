"""Research-only GPT-5.6 Terra observations for Hawk I.

GPT-5.6 Terra accepts images, not video.  This service samples a small,
ordered set of frames from an already-authorized research video and submits
only those frames to the Responses API.  It is deliberately independent from
the baseline and fine-tuned scoring paths.
"""

from __future__ import annotations

import base64
import json
import math
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

SCORE_EVALUATION_SCHEMA = {
    **OBSERVATION_SCHEMA,
    "required": [
        *OBSERVATION_SCHEMA["required"],
        "research_ordinal_score",
        "score_confidence",
        "score_rationale",
    ],
    "properties": {
        **OBSERVATION_SCHEMA["properties"],
        "research_ordinal_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "score_confidence": {"type": "string"},
        "score_rationale": {"type": "string"},
    },
}


@dataclass(frozen=True)
class GPT56TerraConfig:
    api_key: str
    model: str
    sample_fps: float
    max_frames: int
    reasoning_effort: str


def get_config() -> GPT56TerraConfig | None:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        sample_fps = float(os.getenv("GPT56_TERRA_SAMPLE_FPS", "5"))
    except ValueError:
        sample_fps = 5.0
    try:
        max_frames = int(os.getenv("GPT56_TERRA_MAX_FRAMES", "180"))
    except ValueError:
        max_frames = 180

    return GPT56TerraConfig(
        api_key=api_key,
        model=(os.getenv("GPT56_TERRA_MODEL") or "gpt-5.6-terra").strip(),
        sample_fps=max(0.5, min(sample_fps, 10.0)),
        max_frames=max(1, min(max_frames, 180)),
        reasoning_effort=(os.getenv("GPT56_TERRA_REASONING_EFFORT") or "none").strip(),
    )


def research_prompt(task_type: str, frame_count: int, sample_fps: float) -> str:
    task_label = {
        "finger_tapping": "finger-tapping",
        "hand_movement": "hand opening/closing",
        "leg_agility": "leg agility",
        "gait": "gait/walking",
    }.get(task_type, task_type or "movement")
    return f"""This is a de-identified public research video represented by {frame_count}
chronologically ordered still frames of a {task_label} task, sampled at about
{sample_fps:g} fps. Each frame is labelled with its timestamp. This is
movement-observation research only, not a clinical diagnosis or clinical decision.

Describe only what is visible across the timestamped samples: speed, amplitude,
rhythm, pauses/hesitations, and any observable asymmetry. Do not claim finer
temporal precision than this sampling rate permits. Do not output an MDS-UPDRS
score, a diagnosis, or treatment advice. State sampling/visibility limits
explicitly. Return the requested JSON object only."""


def score_evaluation_prompt(task_type: str, frame_count: int, sample_fps: float) -> str:
    return research_prompt(task_type, frame_count, sample_fps).replace(
        "Do not output an MDS-UPDRS\nscore, a diagnosis, or treatment advice.",
        "For this pre-specified research evaluation only, also provide one\n"
        "MDS-UPDRS Part III item 3.10 gait ordinal estimate (0–4). This is not\n"
        "a clinical score or decision. Explain the visible basis and uncertainty.",
    )


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
            "input_mode": "timestamped_sampled_frames",
            "sample_fps": self.config.sample_fps if self.config else None,
            "research_only": True,
            "external_processing_confirmation_required": True,
        }

    @staticmethod
    def _extract_frames(
        video_path: str, sample_fps: float, max_frames: int
    ) -> tuple[list[tuple[Any, float]], float]:
        import cv2

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if total_frames <= 0:
                return [], 0.0
            if source_fps <= 0:
                source_fps = 30.0
            duration_sec = total_frames / source_fps
            # Derive indices from a target timebase instead of merely spreading
            # a fixed number of stills over the entire clip.
            requested = min(max_frames, max(1, math.ceil(duration_sec * sample_fps)))
            frame_indices = [
                min(total_frames - 1, int(round(index * source_fps / sample_fps)))
                for index in range(requested)
            ]
            frame_indices = list(dict.fromkeys(frame_indices))
            frames: list[tuple[Any, float]] = []
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if success:
                    frames.append((cv2.resize(frame, (512, 512)), frame_index / source_fps))
            return frames, duration_sec
        finally:
            capture.release()

    @staticmethod
    def _encode_frame(frame: Any) -> str:
        import cv2

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Could not encode video frame")
        return base64.b64encode(buffer).decode("ascii")

    def analyze_video(
        self,
        video_path: str,
        task_type: str,
        *,
        include_research_score: bool = False,
    ) -> dict[str, Any]:
        if not self.is_available():
            return {"success": False, "error": "GPT-5.6 Terra is not configured."}
        if not Path(video_path).is_file():
            return {"success": False, "error": "Analysis video is unavailable."}

        try:
            frames, duration_sec = self._extract_frames(
                video_path, self.config.sample_fps, self.config.max_frames
            )
            if not frames:
                return {"success": False, "error": "No usable video frames were extracted."}

            content: list[dict[str, Any]] = [
                {
                    "type": "input_text",
                    "text": (
                        score_evaluation_prompt(
                            task_type, len(frames), self.config.sample_fps
                        )
                        if include_research_score
                        else research_prompt(
                            task_type, len(frames), self.config.sample_fps
                        )
                    ),
                }
            ]
            for frame, timestamp_sec in frames:
                content.append(
                    {
                        "type": "input_text",
                        "text": f"Frame timestamp: {timestamp_sec:.2f} seconds.",
                    }
                )
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
                        "name": (
                            "gait_research_score_evaluation"
                            if include_research_score
                            else "movement_observation"
                        ),
                        "strict": True,
                        "schema": (
                            SCORE_EVALUATION_SCHEMA
                            if include_research_score
                            else OBSERVATION_SCHEMA
                        ),
                    }
                },
            )
            observation = _parse_observation(getattr(response, "output_text", None))
            if observation is None:
                return {"success": False, "error": "GPT-5.6 Terra returned an unstructured observation."}
            research_score = None
            if include_research_score:
                raw_score = observation.pop("research_ordinal_score", None)
                score_confidence = observation.pop("score_confidence", None)
                score_rationale = observation.pop("score_rationale", None)
                if not isinstance(raw_score, int) or isinstance(raw_score, bool) or not 0 <= raw_score <= 4:
                    return {"success": False, "error": "GPT-5.6 Terra returned an invalid research score."}
                if not isinstance(score_confidence, str) or not isinstance(score_rationale, str):
                    return {"success": False, "error": "GPT-5.6 Terra returned an incomplete research score."}
                research_score = {
                    "value": raw_score,
                    "confidence": score_confidence,
                    "rationale": score_rationale,
                }
            return {
                "success": True,
                "provider": "openai",
                "model": self.config.model,
                "input_mode": "timestamped_sampled_frames",
                "sample_fps": self.config.sample_fps,
                "frames_analyzed": len(frames),
                "duration_seconds": round(duration_sec, 3),
                "research_only": True,
                "observation": observation,
                **({"research_score": research_score} if research_score is not None else {}),
            }
        except Exception as exc:
            return {"success": False, "error": f"GPT-5.6 Terra analysis failed: {exc}"}
