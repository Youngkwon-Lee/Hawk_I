"""
UPDRS Scoring Module - Rule-based and ML-based scoring
"""

from dataclasses import dataclass
from typing import Literal
from services.metrics_calculator import FingerTappingMetrics, GaitMetrics

ScoringMethod = Literal["rule", "ml", "ensemble"]

@dataclass
class UPDRSScore:
    total_score: float
    base_score: int
    penalties: float
    severity: str
    details: dict
    method: str = "rule"
    confidence: float = 1.0

def _get_severity(score: float) -> str:
    if score < 0.5: return "Normal"
    elif score < 1.5: return "Slight"
    elif score < 2.5: return "Mild"
    elif score < 3.5: return "Moderate"
    else: return "Severe"

class UPDRSScorer:
    def __init__(self, method: ScoringMethod = "rule", ml_model_type: str = "rf"):
        self.method = method
        self.ml_model_type = ml_model_type
        self._ml_scorer = None

    def _get_ml_scorer(self):
        if self._ml_scorer is None:
            try:
                from services.ml_scorer import get_ml_scorer
                self._ml_scorer = get_ml_scorer()
                self._ml_scorer.load_models()
            except Exception as e:
                print(f"ML scorer load failed: {e}")
        return self._ml_scorer

    def score_finger_tapping(self, metrics: FingerTappingMetrics) -> UPDRSScore:
        if self.method == "ml": return self._score_ft_ml(metrics)
        elif self.method == "ensemble": return self._score_ft_ensemble(metrics)
        return self._score_ft_rule(metrics)

    def score_gait(self, metrics: GaitMetrics) -> UPDRSScore:
        if self.method == "ml": return self._score_gait_ml(metrics)
        elif self.method == "ensemble": return self._score_gait_ensemble(metrics)
        return self._score_gait_rule(metrics)

    def _score_ft_ml(self, m: FingerTappingMetrics) -> UPDRSScore:
        ml = self._get_ml_scorer()
        if ml is None: return self._score_ft_rule(m)
        p = ml.predict_finger_tapping(m, self.ml_model_type)
        if p is None: return self._score_ft_rule(m)
        return UPDRSScore(float(p.score), p.score, 0.0, _get_severity(p.score),
            {"method": "ml", "raw": p.raw_prediction}, "ml", p.confidence)

    def _score_gait_ml(self, m: GaitMetrics) -> UPDRSScore:
        ml = self._get_ml_scorer()
        if ml is None: return self._score_gait_rule(m)
        p = ml.predict_gait(m, self.ml_model_type)
        if p is None: return self._score_gait_rule(m)
        return UPDRSScore(float(p.score), p.score, 0.0, _get_severity(p.score),
            {"method": "ml", "raw": p.raw_prediction}, "ml", p.confidence)

    def _score_ft_ensemble(self, m: FingerTappingMetrics) -> UPDRSScore:
        r, ml = self._score_ft_rule(m), self._score_ft_ml(m)
        avg = (r.total_score + ml.total_score) / 2
        return UPDRSScore(round(avg, 1), int(round(avg)), 0.0, _get_severity(avg),
            {"rule": r.total_score, "ml": ml.total_score}, "ensemble", (1 + ml.confidence) / 2)

    def _score_gait_ensemble(self, m: GaitMetrics) -> UPDRSScore:
        r, ml = self._score_gait_rule(m), self._score_gait_ml(m)
        avg = (r.total_score + ml.total_score) / 2
        return UPDRSScore(round(avg, 1), int(round(avg)), 0.0, _get_severity(avg),
            {"rule": r.total_score, "ml": ml.total_score}, "ensemble", (1 + ml.confidence) / 2)

    def _score_ft_rule(self, m: FingerTappingMetrics) -> UPDRSScore:
        sp = m.tapping_speed
        ss = 0 if sp >= 2.5 else (1 if sp >= 2.0 else (2 if sp >= 1.5 else (3 if sp >= 1.0 else 4)))
        dec = m.amplitude_decrement
        ds = 0 if dec < 10 else (1 if dec < 20 else (2 if dec < 40 else 3))
        rv = m.rhythm_variability
        rs = 0 if rv < 15 else (1 if rv < 25 else (2 if rv < 40 else 3))
        w = ss * 0.5 + ds * 0.3 + rs * 0.2
        bs = min(4, int(w + 0.5))
        pen = 0.0
        if m.hesitation_count > 2: pen += min(0.3, (m.hesitation_count - 2) / 5 * 0.3)
        if m.halt_count > 1: pen += min(0.4, (m.halt_count - 1) / 3 * 0.4)
        tot = min(4.0, bs + pen)
        return UPDRSScore(round(tot, 1), bs, round(pen, 2), _get_severity(tot),
            {"speed": ss, "decrement": ds, "rhythm": rs}, "rule", 1.0)

    def _score_gait_rule(self, m: GaitMetrics) -> UPDRSScore:
        arm = m.arm_swing_amplitude_mean
        asc = 0 if arm >= 0.07 else (1 if arm >= 0.05 else (2 if arm >= 0.03 else (3 if arm >= 0.02 else 4)))
        sp = m.walking_speed
        ssc = 0 if sp >= 0.8 else (1 if sp >= 0.6 else (2 if sp >= 0.4 else (3 if sp >= 0.2 else 4)))
        cad = m.cadence
        csc = 0 if 100 <= cad <= 120 else (1 if 80 <= cad < 100 or 120 < cad <= 140 else (2 if 60 <= cad < 80 or 140 < cad <= 160 else 3))
        sh = m.step_height_mean
        shc = 0 if sh >= 0.05 else (1 if sh >= 0.04 else (2 if sh >= 0.03 else 3))
        w = asc * 0.4 + ssc * 0.25 + csc * 0.2 + shc * 0.15
        bs = min(4, int(w + 0.5))
        pen = 0.0
        if m.stride_variability > 10: pen += min(0.3, (m.stride_variability - 10) / 20 * 0.3)
        if m.arm_swing_asymmetry > 15: pen += min(0.4, (m.arm_swing_asymmetry - 15) / 25 * 0.4)
        if m.step_count < 10: pen += 0.2
        tot = min(4.0, bs + pen)
        return UPDRSScore(round(tot, 1), bs, round(pen, 2), _get_severity(tot),
            {"arm": asc, "speed": ssc, "cadence": csc, "step_h": shc}, "rule", 1.0)
