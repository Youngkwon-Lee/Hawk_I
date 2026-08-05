"""Batch ingestion of offline VLM predictions into the shared patient timeline.

For the KHF demo the fine-tuned model is run offline on a training GPU rather
than served live, so its predictions arrive as a JSONL file. This module turns
each prediction into the same analysis-result shape the live pipeline produces,
which lets it reuse ``build_activity_session_row`` / ``build_observation_row``
and land in ``observations`` exactly like an online Hawk I analysis.

Predictions from a research dataset are NOT clinical observations. Every row
written here carries a ``research_provenance`` block naming the dataset, split,
model, and experiment condition so the timeline can never be mistaken for a
real clinical record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import json

REQUIRED_FIELDS = ("clip_id", "task")

OBSERVABILITY_VALUES = frozenset({"observed", "unobservable", "uncertain"})

SEVERITY_BY_SCORE = {
    0: "Normal",
    1: "Slight",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
}


class PredictionValidationError(ValueError):
    """Raised when a prediction record cannot be converted."""


@dataclass
class IngestSummary:
    total: int = 0
    converted: int = 0
    skipped: list[tuple[int, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "converted": self.converted,
            "skipped": [{"line": line, "reason": reason} for line, reason in self.skipped],
        }


def _severity(score: float) -> str:
    return SEVERITY_BY_SCORE.get(int(round(score)), "Unknown")


def score_from(prediction: dict[str, Any]) -> Any:
    """Accept either the v1 flat score or the v2 ontology score anchor."""
    if prediction.get("predicted_score") not in (None, ""):
        return prediction["predicted_score"]
    anchor = prediction.get("score_anchor")
    if isinstance(anchor, dict):
        for field in ("updrs_3_10", "updrs_3_4", "score"):
            if anchor.get(field) is not None:
                return anchor[field]
    return None


def validate_primitives(primitives: Any) -> None:
    """Enforce the ontology rules that stop an absent observation reading as normal."""
    if primitives in (None, {}):
        return
    if not isinstance(primitives, dict):
        raise PredictionValidationError("primitives must be an object keyed by primitive name")

    for name, rating in primitives.items():
        if not isinstance(rating, dict):
            raise PredictionValidationError(f"primitive '{name}': must be an object")

        observability = rating.get("observability")
        if observability not in OBSERVABILITY_VALUES:
            raise PredictionValidationError(
                f"primitive '{name}': observability must be one of {sorted(OBSERVABILITY_VALUES)}"
            )

        severity = rating.get("severity")
        if observability != "observed":
            if severity is not None:
                raise PredictionValidationError(
                    f"primitive '{name}': severity must be null unless observed "
                    "(not visible is not the same as normal)"
                )
            continue

        if severity is None:
            continue
        if isinstance(severity, bool) or not isinstance(severity, int) or not 0 <= severity <= 3:
            raise PredictionValidationError(f"primitive '{name}': severity must be an integer 0-3 or null")
        if severity > 0 and not rating.get("evidence"):
            raise PredictionValidationError(
                f"primitive '{name}': a positive finding requires at least one evidence span"
            )


def validate_prediction(prediction: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_FIELDS if prediction.get(name) in (None, "")]
    if missing:
        raise PredictionValidationError(f"missing required field(s): {', '.join(missing)}")

    validate_primitives(prediction.get("primitives"))

    quality_gate = prediction.get("quality_gate")
    on_hold = isinstance(quality_gate, dict) and quality_gate.get("status") == "hold"

    raw_score = score_from(prediction)
    if raw_score is None:
        # A held clip, or a primitive-only run, legitimately carries no score.
        if on_hold or prediction.get("primitives"):
            if not str(prediction.get("dataset") or "").strip():
                raise PredictionValidationError(
                    "dataset is required so research predictions stay distinguishable from clinical records"
                )
            return
        raise PredictionValidationError(
            "missing required field(s): predicted_score (or score_anchor, or primitives)"
        )

    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise PredictionValidationError("predicted_score must be numeric") from exc

    if not 0 <= score <= 4:
        raise PredictionValidationError("predicted_score must be within the UPDRS 0-4 range")

    if not str(prediction.get("dataset") or "").strip():
        raise PredictionValidationError(
            "dataset is required so research predictions stay distinguishable from clinical records"
        )


def analysis_id_for(prediction: dict[str, Any]) -> str:
    """Stable id so re-ingesting the same prediction updates instead of duplicating."""
    model = str(prediction.get("model") or "model").strip()
    condition = str(prediction.get("condition") or "NA").strip()
    clip_id = str(prediction["clip_id"]).strip()
    return f"offline-{model}-{condition}-{clip_id}".replace(" ", "_")


def research_provenance(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "is_research_prediction": True,
        "is_clinical_record": False,
        "serving_mode": "offline_batch",
        "dataset": prediction.get("dataset"),
        "split": prediction.get("split"),
        "clip_id": prediction.get("clip_id"),
        "research_subject_ref": prediction.get("subject_ref"),
        "model": prediction.get("model"),
        "condition": prediction.get("condition"),
        "reference_score": prediction.get("true_score"),
    }


def _primitives_to_narrative(primitives: dict[str, Any]) -> str:
    """Render positive findings as one readable line, ordered by severity."""
    findings = [
        (rating.get("severity"), name)
        for name, rating in primitives.items()
        if isinstance(rating, dict)
        and rating.get("observability") == "observed"
        and isinstance(rating.get("severity"), int)
        and rating.get("severity", 0) > 0
    ]
    if not findings:
        return ""
    findings.sort(reverse=True)
    return ", ".join(f"{name} {severity}단계" for severity, name in findings) + " 관찰됨"


def prediction_to_result(prediction: dict[str, Any]) -> dict[str, Any]:
    """Convert one offline prediction into the live pipeline's result shape."""
    validate_prediction(prediction)

    raw_score = score_from(prediction)
    score = None if raw_score is None else float(raw_score)
    rationale = str(prediction.get("rationale") or "").strip()
    primitives = prediction.get("primitives") if isinstance(prediction.get("primitives"), dict) else {}
    quality_gate = prediction.get("quality_gate") if isinstance(prediction.get("quality_gate"), dict) else {}
    on_hold = quality_gate.get("status") == "hold"

    result: dict[str, Any] = {
        "success": True,
        "id": analysis_id_for(prediction),
        "patient_id": prediction.get("subject_ref") or prediction.get("clip_id"),
        "video_type": str(prediction["task"]),
        "auto_detected": False,
        "confidence": prediction.get("confidence"),
        "scoring_method": "vlm_offline",
        "ml_model_type": prediction.get("model"),
        "updrs_score": {
            "score": score,
            "total_score": score,
            "severity": None if score is None else _severity(score),
            "method": "vlm_offline",
            "confidence": prediction.get("confidence"),
            "details": {
                "condition": prediction.get("condition"),
                "model": prediction.get("model"),
                "reference_score": prediction.get("reference_score", prediction.get("true_score")),
                "score_anchor_source": (prediction.get("score_anchor") or {}).get("source")
                if isinstance(prediction.get("score_anchor"), dict) else None,
            },
        },
        "metrics": prediction.get("metrics") or {},
        "events": [],
    }

    if primitives:
        # The structured observation is the evidence a clinician reads; the score
        # stays a separate anchor and is never derived by summing severities.
        result["primitives"] = primitives
    if quality_gate:
        result["quality_gate"] = quality_gate
    if on_hold:
        result["performability_assessment"] = {
            "status": "hold",
            "summary": quality_gate.get("note") or "quality gate hold",
        }
        result["score_advisory"] = {
            "level": "reference_only",
            "summary": ", ".join(quality_gate.get("reasons") or []) or "quality gate hold",
        }

    narrative = rationale or _primitives_to_narrative(primitives)
    if narrative:
        result["ai_interpretation"] = {
            "summary": narrative,
            "explanation": rationale or narrative,
            "recommendations": [],
        }

    if prediction.get("observed_at"):
        result["observed_at"] = prediction["observed_at"]

    return result


def attach_research_provenance(
    row: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Mark an observation row as an offline research prediction."""
    context = row.get("measurement_context")
    if not isinstance(context, dict):
        context = {}
    context["research_provenance"] = research_provenance(prediction)
    if isinstance(prediction.get("primitives"), dict) and prediction["primitives"]:
        context["primitives"] = prediction["primitives"]
    if isinstance(prediction.get("quality_gate"), dict) and prediction["quality_gate"]:
        context["quality_gate"] = prediction["quality_gate"]
    row["measurement_context"] = context

    categories = row.get("category")
    categories = list(categories) if isinstance(categories, list) else []
    if "research-prediction" not in categories:
        categories.append("research-prediction")
    row["category"] = categories

    if prediction.get("observed_at"):
        row["effective_datetime"] = prediction["observed_at"]

    return row


def load_predictions(lines: Iterable[str]) -> tuple[list[dict[str, Any]], IngestSummary]:
    """Parse JSONL text into validated predictions, collecting per-line failures."""
    summary = IngestSummary()
    predictions: list[dict[str, Any]] = []

    for index, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        summary.total += 1
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            summary.skipped.append((index, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(record, dict):
            summary.skipped.append((index, "line is not a JSON object"))
            continue
        try:
            validate_prediction(record)
        except PredictionValidationError as exc:
            summary.skipped.append((index, str(exc)))
            continue
        predictions.append(record)
        summary.converted += 1

    return predictions, summary
