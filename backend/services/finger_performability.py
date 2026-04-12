"""
Finger tapping performability gate.

Purpose:
- decide whether a patient appears able to perform the task at all
- separate near-impossible performance from analyzable-but-slow performance
- give an explicit pre-score gate before the 0-4 UPDRS scorer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional


PerformabilityStatus = Literal[
    "performable",
    "uncertain",
    "non_performable_or_near_impossible",
    "unscorable_due_to_tracking",
]


@dataclass
class FingerPerformabilityAssessment:
    status: PerformabilityStatus
    confidence: float
    summary: str
    analyzable: bool
    evidence: Dict[str, float]
    triggers: list[str]


class FingerPerformabilityGate:
    """
    Rule-based pre-score gate for finger tapping.

    This is intentionally conservative:
    - only call "non_performable_or_near_impossible" when multiple severe cues align
    - preserve an "uncertain" bucket for cases that need clinician review
    """

    def assess(self, metrics: Any, detection_rate: Optional[float] = None) -> FingerPerformabilityAssessment:
        data = self._to_dict(metrics)
        evidence = {
            "total_taps": float(data.get("total_taps", 0.0) or 0.0),
            "tapping_speed": float(data.get("tapping_speed", 0.0) or 0.0),
            "peak_velocity_mean": float(data.get("peak_velocity_mean", 0.0) or 0.0),
            "effective_tap_count": float(data.get("effective_tap_count", 0.0) or 0.0),
            "effective_tap_ratio": float(data.get("effective_tap_ratio", 0.0) or 0.0),
            "below_threshold_cycle_ratio": float(data.get("below_threshold_cycle_ratio", 0.0) or 0.0),
            "longest_pause": float(data.get("longest_pause", 0.0) or 0.0),
            "post_onset_pause_ratio": float(data.get("post_onset_pause_ratio", 0.0) or 0.0),
            "halt_count": float(data.get("halt_count", 0.0) or 0.0),
            "incomplete_tap_ratio": float(data.get("incomplete_tap_ratio", 0.0) or 0.0),
            "rhythm_variability": float(data.get("rhythm_variability", 0.0) or 0.0),
            "variability_second_half": float(data.get("variability_second_half", 0.0) or 0.0),
            "velocity_decrement": float(data.get("velocity_decrement", 0.0) or 0.0),
            "velocity_slope": float(data.get("velocity_slope", 0.0) or 0.0),
        }
        if detection_rate is not None:
            evidence["detection_rate"] = float(detection_rate)

        analyzable, tracking_triggers = self._assess_tracking(evidence)
        if not analyzable:
            return FingerPerformabilityAssessment(
                status="unscorable_due_to_tracking",
                confidence=0.85,
                summary="Hand tracking or usable tapping signal is insufficient for reliable scoring.",
                analyzable=False,
                evidence=evidence,
                triggers=tracking_triggers,
            )

        severe_score = 0
        triggers: list[str] = []

        if evidence["tapping_speed"] <= 1.2:
            severe_score += 2
            triggers.append("very_slow_tapping_speed")
        elif evidence["tapping_speed"] <= 1.6:
            severe_score += 1
            triggers.append("slow_tapping_speed")

        if evidence["peak_velocity_mean"] <= 12.0:
            severe_score += 2
            triggers.append("low_peak_velocity")
        elif evidence["peak_velocity_mean"] <= 16.0:
            severe_score += 1
            triggers.append("reduced_peak_velocity")

        if evidence["rhythm_variability"] >= 28.0:
            severe_score += 1
            triggers.append("high_rhythm_variability")

        if evidence["post_onset_pause_ratio"] >= 1.8:
            severe_score += 1
            triggers.append("long_post_onset_pause")
        elif evidence["post_onset_pause_ratio"] >= 1.55:
            severe_score += 1
            triggers.append("moderate_post_onset_pause")

        if evidence["halt_count"] >= 2:
            severe_score += 1
            triggers.append("multiple_halts")
        elif evidence["halt_count"] >= 1:
            severe_score += 1
            triggers.append("halt_present")

        if evidence["variability_second_half"] >= 28.0:
            severe_score += 1
            triggers.append("high_second_half_variability")

        if evidence["velocity_decrement"] <= -20.0:
            severe_score += 1
            triggers.append("strong_velocity_drop")

        perform_score = 0
        if evidence["tapping_speed"] >= 2.3:
            perform_score += 1
        if evidence["peak_velocity_mean"] >= 18.0:
            perform_score += 1
        if evidence["rhythm_variability"] <= 24.0:
            perform_score += 1
        if evidence["halt_count"] <= 1:
            perform_score += 1
        if evidence["below_threshold_cycle_ratio"] <= 0.55:
            perform_score += 1

        interruption_cue = (
            evidence["halt_count"] >= 1
            or evidence["post_onset_pause_ratio"] >= 1.55
            or evidence["variability_second_half"] >= 28.0
        )

        if severe_score >= 4 and evidence["tapping_speed"] <= 1.8 and interruption_cue:
            confidence = min(0.95, 0.55 + 0.07 * severe_score)
            return FingerPerformabilityAssessment(
                status="non_performable_or_near_impossible",
                confidence=round(confidence, 3),
                summary="Tapping attempts are present, but the pattern suggests near-impossible or non-performable task execution.",
                analyzable=True,
                evidence=evidence,
                triggers=triggers,
            )

        if perform_score >= 4 and severe_score <= 1:
            confidence = min(0.95, 0.55 + 0.06 * perform_score)
            return FingerPerformabilityAssessment(
                status="performable",
                confidence=round(confidence, 3),
                summary="A sufficient tapping pattern is present to proceed with scoring.",
                analyzable=True,
                evidence=evidence,
                triggers=[],
            )

        confidence = 0.55 + 0.03 * max(severe_score, perform_score)
        return FingerPerformabilityAssessment(
            status="uncertain",
            confidence=round(min(confidence, 0.8), 3),
            summary="Some tapping signal is present, but performability is borderline and should be reviewed.",
            analyzable=True,
            evidence=evidence,
            triggers=triggers,
        )

    def _to_dict(self, metrics: Any) -> Dict[str, Any]:
        if hasattr(metrics, "__dataclass_fields__"):
            return asdict(metrics)
        if isinstance(metrics, dict):
            return metrics
        raise ValueError(f"Unsupported metrics type: {type(metrics)}")

    def _assess_tracking(self, evidence: Dict[str, float]) -> tuple[bool, list[str]]:
        triggers: list[str] = []
        detection_rate = evidence.get("detection_rate")
        if detection_rate is not None and detection_rate < 0.3:
            triggers.append("low_detection_rate")
        if evidence["total_taps"] <= 1:
            triggers.append("almost_no_detected_taps")
        if evidence["peak_velocity_mean"] <= 0.0:
            triggers.append("no_velocity_signal")
        if evidence["effective_tap_count"] <= 0 and evidence["tapping_speed"] <= 0.2:
            triggers.append("no_effective_motor_signal")
        return (len(triggers) == 0, triggers)


_gate_singleton: Optional[FingerPerformabilityGate] = None


def get_finger_performability_gate() -> FingerPerformabilityGate:
    global _gate_singleton
    if _gate_singleton is None:
        _gate_singleton = FingerPerformabilityGate()
    return _gate_singleton
