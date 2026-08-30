"""Call a self-hosted fine-tuned VLM and read back ontology primitives.

The existing VLMScorer talks only to OpenAI and returns a free-text score. The
fine-tuned model is different in both directions: it lives behind whatever
OpenAI-compatible endpoint it is served from, and it returns the same primitive
structure the labelers use, so the clinical view can render it directly.

Nothing here assumes the endpoint is up. If it is not configured or not
reachable, the caller keeps its existing score rather than failing the analysis.
"""

from __future__ import annotations

import json
from numbers import Real
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from services.vlm_training_contract import build_c0b_messages

GAIT_PRIMITIVES = [
    "gait_speed_reduction",
    "shortened_stride",
    "step_length_asymmetry",
    "arm_swing_asymmetry",
    "festination",
    "freezing_of_gait",
    "trunk_flexion",
    "postural_instability",
]

OBSERVABILITY = frozenset({"observed", "unobservable", "uncertain"})

PROMPT = """You are assisting a movement-disorders research study on Parkinson gait video.

The frames are sampled uniformly from a single walking clip lasting {duration:.1f} seconds, so frame i is at approximately i * {duration:.1f} / {n_frames} seconds.

Rate each of these findings. Use severity 0 (none), 1 (mild), 2 (moderate),
3 (severe). If a finding cannot be judged from the video, set observability to
"unobservable" and severity to null - do NOT report it as 0.

{primitive_list}

Freezing of gait means a sudden transient inability to start or continue
stepping with the feet appearing stuck. Deliberate stopping and ordinary
slowness are NOT freezing.

Also report the time spans where turning occurs and where freezing occurs.

Respond with JSON only:
{{"primitives": {{"<name>": {{"observability": "observed|unobservable|uncertain",
 "severity": <0-3 or null>, "confidence": "low|medium|high"}}}},
 "turning_intervals": [{{"start": <sec>, "end": <sec>}}],
 "freezing_intervals": [{{"start": <sec>, "end": <sec>}}],
 "updrs_3_10": <0-4 or null>,
 "summary": "<one sentence describing what was observed>"}}"""


class FinetunedVLMUnavailable(RuntimeError):
    """Raised when the endpoint is not configured or cannot be reached."""


@dataclass(frozen=True)
class FinetunedVLMConfig:
    base_url: str
    model: str
    api_key: str = ""
    # C3BE training uses 100 frames at 5 fps; callers can lower this explicitly
    # for smoke tests via HAWKEYE_VLM_MAX_FRAMES.
    max_frames: int = 100
    timeout_seconds: float = 180.0
    condition: str = "C0B"

    @property
    def is_local(self) -> bool:
        return any(h in self.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))


def get_config() -> FinetunedVLMConfig | None:
    """Read endpoint settings from the environment; None means "not wired up yet"."""
    base_url = (os.getenv("HAWKEYE_VLM_BASE_URL") or "").strip()
    model = (os.getenv("HAWKEYE_VLM_MODEL") or "").strip()
    if not base_url or not model:
        return None

    try:
        max_frames = int(os.getenv("HAWKEYE_VLM_MAX_FRAMES", "100"))
    except ValueError:
        max_frames = 100
    try:
        timeout = float(os.getenv("HAWKEYE_VLM_TIMEOUT_SECONDS", "180"))
    except ValueError:
        timeout = 180.0

    return FinetunedVLMConfig(
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=(os.getenv("HAWKEYE_VLM_API_KEY") or "").strip(),
        max_frames=max(1, max_frames),
        timeout_seconds=timeout,
        condition=(os.getenv("HAWKEYE_VLM_CONDITION") or "C0B").strip().upper(),
    )


def status() -> dict[str, Any]:
    """Return safe endpoint state for diagnostics; never expose the API key."""
    config = get_config()
    return {
        "configured": config is not None,
        "model": config.model if config else None,
        "base_url_configured": bool(config),
        "max_frames": config.max_frames if config else None,
        "condition": config.condition if config else None,
    }


def build_prompt(duration_sec: float, n_frames: int) -> str:
    """The prompt carries the timebase; the model only sees ordered frames."""
    if duration_sec <= 0 or n_frames <= 0:
        raise ValueError("duration and frame count must be positive")
    listed = "\n".join(f"- {name}" for name in GAIT_PRIMITIVES)
    return PROMPT.format(duration=duration_sec, n_frames=n_frames, primitive_list=listed)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _intervals(raw: Any, duration_sec: float | None) -> list[dict[str, float]]:
    if not isinstance(raw, list):
        return []
    spans = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        start, end = _number(entry.get("start")), _number(entry.get("end"))
        if start is None or end is None or end <= start or start < 0:
            continue
        if duration_sec is not None:
            if start >= duration_sec:
                continue
            end = min(end, duration_sec)
        spans.append({"start": round(start, 2), "end": round(end, 2)})
    return sorted(spans, key=lambda s: s["start"])


def parse_response(text: str, duration_sec: float | None = None) -> dict[str, Any]:
    """Read the model reply, dropping anything that breaks the ontology rules.

    A malformed rating is discarded rather than repaired: a severity invented
    for an unobservable finding would teach the clinical view that an occluded
    turn looked healthy.
    """
    if not isinstance(text, str) or not text.strip():
        raise FinetunedVLMUnavailable("empty response from model")

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace:
            candidate = brace.group(0)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        # The training prompts in Drive terminate with `answer: <0-4>`.
        # Accept that contract as a score-only response instead of discarding
        # a valid model prediction merely because it is not JSON.
        answer = re.search(r"(?:^|\n)\s*answer\s*:\s*([0-4])\s*$", candidate, re.IGNORECASE)
        if not answer:
            raise FinetunedVLMUnavailable("model reply was not JSON or an answer line")
        payload = {"updrs_3_10": int(answer.group(1))}
    if not isinstance(payload, dict):
        raise FinetunedVLMUnavailable("model reply was not a JSON object")

    primitives: dict[str, Any] = {}
    raw_primitives = payload.get("primitives")
    if isinstance(raw_primitives, dict):
        for name, rating in raw_primitives.items():
            if name not in GAIT_PRIMITIVES or not isinstance(rating, dict):
                continue
            observability = rating.get("observability")
            if observability not in OBSERVABILITY:
                continue
            severity = _number(rating.get("severity"))
            if observability != "observed":
                severity = None  # not visible is never normal
            elif severity is not None:
                if not 0 <= severity <= 3:
                    continue
                severity = int(round(severity))
            confidence = rating.get("confidence")
            primitives[name] = {
                "observability": observability,
                "severity": severity,
                "confidence": confidence if confidence in ("low", "medium", "high") else None,
            }

    score = _number(payload.get("updrs_3_10"))
    if score is None:
        score = _number(payload.get("score"))
    if score is not None and not 0 <= score <= 4:
        score = None

    summary = payload.get("summary")
    return {
        "primitives": primitives,
        "turning_intervals": _intervals(payload.get("turning_intervals"), duration_sec),
        "freezing_intervals": _intervals(payload.get("freezing_intervals"), duration_sec),
        "updrs_3_10": None if score is None else int(round(score)),
        "summary": summary.strip() if isinstance(summary, str) and summary.strip() else None,
    }


def call_endpoint(
    frames_b64: list[str],
    duration_sec: float,
    config: FinetunedVLMConfig,
) -> str:
    """POST frames to an OpenAI-compatible chat endpoint and return the raw reply."""
    if config.condition != "C0B":
        raise FinetunedVLMUnavailable(
            f"condition {config.condition} is not wired for autonomous inference; use C0B"
        )

    messages = build_c0b_messages()
    content = messages[1]["content"]
    for frame in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    try:
        response = requests.post(
            f"{config.base_url}/chat/completions",
            headers=headers,
            json={
                "model": config.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 16,
            },
            timeout=config.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise FinetunedVLMUnavailable(f"cannot reach {config.base_url}: {exc}") from exc

    if response.status_code >= 400:
        raise FinetunedVLMUnavailable(
            f"{response.status_code} from {config.model}: {response.text[:200]}"
        )

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise FinetunedVLMUnavailable("unexpected response shape from endpoint") from exc


def to_analysis_fields(parsed: dict[str, Any], config: FinetunedVLMConfig) -> dict[str, Any]:
    """Shape the result the way the analysis pipeline and clinical view expect."""
    fields: dict[str, Any] = {
        "primitives": parsed["primitives"],
        "gait_events": {
            "turning_intervals": parsed["turning_intervals"],
            "freezing_intervals": parsed["freezing_intervals"],
        },
        "scoring_method": "finetuned_vlm",
        "ml_model_type": config.model,
    }

    if parsed["summary"]:
        fields["ai_interpretation"] = {
            "summary": parsed["summary"],
            "explanation": parsed["summary"],
            "recommendations": [],
        }

    score = parsed["updrs_3_10"]
    if score is not None:
        severity = ("Normal", "Slight", "Mild", "Moderate", "Severe")[score]
        # The score stays a separate anchor; severities are never summed into it.
        fields["updrs_score"] = {
            "score": float(score),
            "total_score": float(score),
            "severity": severity,
            "method": "finetuned_vlm",
            "details": {"model": config.model, "source": "model_predicted"},
        }
    return fields


def apply_to_analysis_response(response: dict[str, Any], finetuned: dict[str, Any]) -> None:
    """Promote a real model score while retaining the prior pipeline reference.

    The rule/CORAL pipeline runs first so analysis still works when the endpoint
    is unavailable. Once the fine-tuned endpoint returns a score, however, that
    score must become the primary result; otherwise the response advertises a
    fine-tuned method while still showing a rule score.
    """
    model_fields = dict(finetuned)
    model_interpretation = model_fields.get("ai_interpretation")
    supplied_score = model_fields.pop("updrs_score", None)
    previous_score = response.get("updrs_score")
    previous_interpretation = response.get("ai_interpretation")
    response.update(model_fields)

    if not isinstance(supplied_score, dict):
        return

    if isinstance(previous_score, dict):
        reference_score = previous_score.get("total_score")
        if reference_score is None:
            reference_score = previous_score.get("score")
        pipeline_reference = {
            "score": reference_score,
            "method": previous_score.get("method") or response.get("scoring_method"),
        }
        if isinstance(previous_interpretation, dict):
            reference_summary = previous_interpretation.get("summary")
            if isinstance(reference_summary, str) and reference_summary.strip():
                pipeline_reference["summary"] = reference_summary.strip()
        supplied_score.setdefault("details", {})["pipeline_reference"] = pipeline_reference
    response["updrs_score"] = supplied_score

    # The C0B training contract commonly returns only ``answer: <0-4>``. In
    # that case the rule pipeline's earlier narrative must not remain as the
    # primary finding after the model score has replaced the rule score. Keep
    # the rule narrative in ``pipeline_reference`` above and generate a
    # deliberately conservative, score-consistent model finding for display.
    if not isinstance(model_interpretation, dict):
        score = supplied_score.get("total_score", supplied_score.get("score"))
        model_name = response.get("ml_model_type") or "fine-tuned VLM"
        if isinstance(score, Real):
            score_index = int(round(float(score)))
            severity = supplied_score.get("severity")
            if not severity and 0 <= score_index <= 4:
                severity = ("Normal", "Slight", "Mild", "Moderate", "Severe")[score_index]
            severity = severity or "Unknown"
            display_score = int(score) if float(score).is_integer() else float(score)
            response["ai_interpretation"] = {
                "summary": (
                    f"미세조정 모델은 이번 보행의 UPDRS 3.10 점수를 "
                    f"{display_score}점({severity})으로 추정했습니다."
                ),
                "explanation": (
                    f"최종 표시는 {model_name}의 모델 예측값을 사용합니다. "
                    "보행 속도·보폭 등 정량 지표와 원본 영상을 함께 검토하세요."
                ),
                "recommendations": [],
            }


def frames_to_base64(video_path: str, max_frames: int | None = None) -> tuple[list[str], float]:
    """Sample frames as base64 JPEGs and return them with the clip duration.

    Returns an empty list when the endpoint is not configured, so callers do not
    pay decoding cost for a model that is not there.
    """
    config = get_config()
    if config is None:
        return [], 0.0

    import base64

    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return [], 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps if fps else 0.0
    if total <= 0:
        capture.release()
        return [], 0.0

    limit = max_frames or config.max_frames
    step = max(1, total // limit)
    frames: list[str] = []
    for index in range(0, total, step):
        if len(frames) >= limit:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (512, 512))
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames.append(base64.b64encode(buffer).decode("utf-8"))
    capture.release()
    return frames, duration


def analyze(frames_b64: list[str], duration_sec: float) -> dict[str, Any] | None:
    """Run the fine-tuned model, or return None when it is not available.

    Returning None rather than raising keeps a missing endpoint from failing an
    otherwise good analysis - the caller falls back to its existing scorer.
    """
    config = get_config()
    if config is None or not frames_b64:
        return None
    try:
        reply = call_endpoint(frames_b64, duration_sec, config)
        return to_analysis_fields(parse_response(reply, duration_sec), config)
    except FinetunedVLMUnavailable as exc:
        print(f"[finetuned_vlm] unavailable, falling back: {exc}")
        return None
