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

REQUIRED_FIELDS = ("clip_id", "task", "predicted_score")

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


def validate_prediction(prediction: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_FIELDS if prediction.get(name) in (None, "")]
    if missing:
        raise PredictionValidationError(f"missing required field(s): {', '.join(missing)}")

    try:
        score = float(prediction["predicted_score"])
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


def prediction_to_result(prediction: dict[str, Any]) -> dict[str, Any]:
    """Convert one offline prediction into the live pipeline's result shape."""
    validate_prediction(prediction)

    score = float(prediction["predicted_score"])
    rationale = str(prediction.get("rationale") or "").strip()

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
            "severity": _severity(score),
            "method": "vlm_offline",
            "confidence": prediction.get("confidence"),
            "details": {
                "condition": prediction.get("condition"),
                "model": prediction.get("model"),
                "reference_score": prediction.get("true_score"),
            },
        },
        "metrics": prediction.get("metrics") or {},
        "events": [],
    }

    if rationale:
        result["ai_interpretation"] = {
            "summary": rationale,
            "explanation": rationale,
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
