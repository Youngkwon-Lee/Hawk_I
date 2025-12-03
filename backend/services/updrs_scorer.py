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
        # Speed (Hz): Normal >= 4.0, Slight >= 3.0, Mild >= 2.0, Moderate >= 1.2
        sp = m.tapping_speed
        ss = 0 if sp >= 4.0 else (1 if sp >= 3.0 else (2 if sp >= 2.0 else (3 if sp >= 1.2 else 4)))

        # Amplitude decrement (%): Normal < 10, Slight < 25, Mild < 50, Moderate < 70
        dec = m.amplitude_decrement
        ds = 0 if dec < 10 else (1 if dec < 25 else (2 if dec < 50 else (3 if dec < 70 else 4)))

        # Rhythm variability (ratio 0-1): Normal < 0.08, Slight < 0.18, Mild < 0.32, Moderate < 0.50
        rv = m.rhythm_variability
        rs = 0 if rv < 0.08 else (1 if rv < 0.18 else (2 if rv < 0.32 else (3 if rv < 0.50 else 4)))

        # Velocity decrement (%): Normal < 10, Slight < 25, Mild < 45, Moderate < 60
        vd = m.velocity_decrement
        vs = 0 if vd < 10 else (1 if vd < 25 else (2 if vd < 45 else (3 if vd < 60 else 4)))

        # Weighted score - balanced across all factors
        w = ss * 0.30 + ds * 0.25 + rs * 0.20 + vs * 0.25
        bs = min(4, int(w + 0.5))

        # Penalties for halts, hesitations, and freezes (reduced impact)
        pen = 0.0
        if m.hesitation_count > 2: pen += min(0.2, (m.hesitation_count - 2) / 5 * 0.2)
        if m.halt_count > 1: pen += min(0.3, (m.halt_count - 1) / 3 * 0.3)
        if m.freeze_episodes > 1: pen += min(0.2, (m.freeze_episodes - 1) * 0.1)

        tot = min(4.0, bs + pen)
        return UPDRSScore(round(tot, 1), bs, round(pen, 2), _get_severity(tot),
            {"speed": ss, "decrement": ds, "rhythm": rs, "velocity": vs}, "rule", 1.0)

    def _score_gait_rule(self, m: GaitMetrics) -> UPDRSScore:
        # Arm swing amplitude (degrees): Normal >= 20, Slight >= 15, Mild >= 10, Moderate >= 5
        arm = m.arm_swing_amplitude_mean
        asc = 0 if arm >= 20 else (1 if arm >= 15 else (2 if arm >= 10 else (3 if arm >= 5 else 4)))

        # Walking speed (m/s): Normal >= 1.0, Slight >= 0.8, Mild >= 0.6, Moderate >= 0.4
        sp = m.walking_speed
        ssc = 0 if sp >= 1.0 else (1 if sp >= 0.8 else (2 if sp >= 0.6 else (3 if sp >= 0.4 else 4)))

        # Cadence (steps/min): Normal 100-120, slight deviation, moderate deviation
        cad = m.cadence
        csc = 0 if 100 <= cad <= 120 else (1 if 85 <= cad < 100 or 120 < cad <= 135 else (2 if 70 <= cad < 85 or 135 < cad <= 150 else 3))

        # Step height (meters): Normal >= 0.12, Slight >= 0.10, Mild >= 0.07, Moderate >= 0.04
        sh = m.step_height_mean
        shc = 0 if sh >= 0.12 else (1 if sh >= 0.10 else (2 if sh >= 0.07 else (3 if sh >= 0.04 else 4)))

        # Stride length (meters): Normal >= 1.2, Slight >= 0.9, Mild >= 0.6
        sl = m.stride_length
        slc = 0 if sl >= 1.2 else (1 if sl >= 0.9 else (2 if sl >= 0.6 else 3))

        # Weighted score: arm swing most important for PD
        w = asc * 0.30 + ssc * 0.20 + csc * 0.15 + shc * 0.15 + slc * 0.20
        bs = min(4, int(w + 0.5))

        # Penalties for variability and asymmetry
        pen = 0.0
        if m.stride_variability > 0.08: pen += min(0.3, (m.stride_variability - 0.08) / 0.20 * 0.3)
        if m.arm_swing_asymmetry > 0.10: pen += min(0.4, (m.arm_swing_asymmetry - 0.10) / 0.30 * 0.4)
        if m.festination_index > 0.05: pen += min(0.3, m.festination_index * 2)
        if m.step_count < 10: pen += 0.2

        tot = min(4.0, bs + pen)
        return UPDRSScore(round(tot, 1), bs, round(pen, 2), _get_severity(tot),
            {"arm": asc, "speed": ssc, "cadence": csc, "step_h": shc, "stride": slc}, "rule", 1.0)
