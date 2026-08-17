"""Research-only Gemini video observations for Hawk I.

This integration is intentionally kept separate from the baseline and
fine-tuned scoring paths.  It sends an already-authorized research video to
Gemini only when the caller explicitly confirms external processing.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class GeminiVLMConfig:
    api_key: str
    model: str
    upload_timeout_seconds: int


def get_config() -> GeminiVLMConfig | None:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None

    try:
        timeout = int(os.getenv("GEMINI_VLM_UPLOAD_TIMEOUT_SECONDS", "120"))
    except ValueError:
        timeout = 120

    return GeminiVLMConfig(
        api_key=api_key,
        model=(os.getenv("GEMINI_VLM_MODEL") or "gemini-2.5-flash").strip(),
        upload_timeout_seconds=max(10, timeout),
    )


def parse_observation(text: str | None) -> dict[str, Any] | None:
    """Return a structured model observation without guessing malformed JSON."""
    if not isinstance(text, str) or not text.strip():
        return None

    candidate = text.strip()
    fenced = JSON_BLOCK.search(candidate)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def research_prompt(task_type: str) -> str:
    task_label = {
        "finger_tapping": "finger-tapping",
        "hand_movement": "hand opening/closing",
        "leg_agility": "leg agility",
        "gait": "gait/walking",
    }.get(task_type, task_type or "movement")
    return f"""This is a de-identified public research video of a {task_label} task.
This is movement-observation research only, not a clinical diagnosis or a
clinical decision. Analyze observable movement only. Return ONLY valid JSON
with the fields task_confirmed, tap_speed, amplitude, rhythm,
pauses_or_hesitations, laterality_or_asymmetry, overall_observation, and
limitations. Do not output an MDS-UPDRS score or a diagnosis."""


class GeminiResearchVLM:
    """Calls Gemini's native video API and deletes the remote file afterwards."""

    def __init__(self, config: GeminiVLMConfig | None = None):
        self.config = config or get_config()

    def is_available(self) -> bool:
        if self.config is None:
            return False
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return True

    def status(self) -> dict[str, Any]:
        return {
            "available": self.is_available(),
            "model": self.config.model if self.config else None,
            "research_only": True,
            "external_processing_confirmation_required": True,
        }

    def analyze_video(self, video_path: str, task_type: str) -> dict[str, Any]:
        if self.config is None:
            return {"success": False, "error": "Gemini VLM is not configured."}
        if not Path(video_path).is_file():
            return {"success": False, "error": "Analysis video is unavailable."}

        try:
            from google import genai
        except ImportError:
            return {"success": False, "error": "google-genai is not installed."}

        client = genai.Client(api_key=self.config.api_key)
        uploaded = None
        try:
            uploaded = client.files.upload(file=video_path)
            deadline = time.monotonic() + self.config.upload_timeout_seconds
            while time.monotonic() < deadline:
                uploaded = client.files.get(name=uploaded.name)
                state = str(getattr(uploaded, "state", ""))
                if "ACTIVE" in state:
                    break
                if "FAILED" in state:
                    return {"success": False, "error": "Gemini rejected the uploaded video."}
                time.sleep(1)
            else:
                return {"success": False, "error": "Gemini video processing timed out."}

            response = client.models.generate_content(
                model=self.config.model,
                contents=[uploaded, research_prompt(task_type)],
                config={
                    "response_mime_type": "application/json",
                    "max_output_tokens": 512,
                    "temperature": 0,
                    # This endpoint returns a constrained observation object;
                    # reserve generation tokens for that object rather than
                    # spending the small response budget on hidden reasoning.
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            raw_output = (getattr(response, "text", None) or "").strip()
            observation = parse_observation(raw_output)
            if observation is None:
                return {
                    "success": False,
                    "error": "Gemini returned an unstructured observation.",
                }
            return {
                "success": True,
                "provider": "gemini",
                "model": self.config.model,
                "research_only": True,
                "observation": observation,
            }
        except Exception as exc:  # Provider failures are returned as an API error.
            return {"success": False, "error": f"Gemini analysis failed: {exc}"}
        finally:
            if uploaded is not None:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass
