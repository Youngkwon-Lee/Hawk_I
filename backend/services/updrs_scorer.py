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
        # PD4T-calibrated thresholds (2024-12-09)
        # Speed (Hz): Score0=2.47, Score1=2.08, Score2=1.60
        sp = m.tapping_speed
        ss = 0 if sp >= 2.25 else (1 if sp >= 1.85 else (2 if sp >= 1.45 else (3 if sp >= 1.0 else 4)))

        # Amplitude decrement (%): Score0=3%, Score1=5%, Score2=11%
        dec = m.amplitude_decrement
        ds = 0 if dec < 4 else (1 if dec < 8 else (2 if dec < 15 else (3 if dec < 25 else 4)))

        # Rhythm variability - handle CV% (17-19) or ratio (0-1)
        rv = m.rhythm_variability
        if rv > 1.0:  # CV% format
            rs = 0 if rv < 15 else (1 if rv < 20 else (2 if rv < 30 else (3 if rv < 45 else 4)))
        else:  # Ratio format
            rs = 0 if rv < 0.15 else (1 if rv < 0.20 else (2 if rv < 0.30 else (3 if rv < 0.45 else 4)))

        # Velocity decrement (%): Score0=0%, Score1=-7%, Score2=6%
        vd = m.velocity_decrement
        vs = 0 if vd < 5 else (1 if vd < 15 else (2 if vd < 30 else (3 if vd < 50 else 4)))

        # Weighted: speed most discriminative
        w = ss * 0.45 + ds * 0.30 + rs * 0.10 + vs * 0.15
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
        # Arm swing amplitude: Check if normalized (0-1) or degrees
        # PD4T data is normalized: 0.1-0.2 range; Real degrees: 5-25 range
        arm = m.arm_swing_amplitude_mean
        if arm < 1.0:  # Normalized data (0-1 scale) - PD4T calibrated
            # Score0=0.128, Score1=0.097, Score2=0.088, Score3=0.056
            asc = 0 if arm >= 0.11 else (1 if arm >= 0.09 else (2 if arm >= 0.07 else (3 if arm >= 0.05 else 4)))
        else:  # Real degrees
            asc = 0 if arm >= 20 else (1 if arm >= 15 else (2 if arm >= 10 else (3 if arm >= 5 else 4)))

        # Walking speed - PD4T calibrated: Score0=0.77, Score1=0.62, Score2=0.53
        sp = m.walking_speed
        ssc = 0 if sp >= 0.70 else (1 if sp >= 0.57 else (2 if sp >= 0.45 else (3 if sp >= 0.35 else 4)))

        # Cadence: PD4T data has high cadence (118-171), normal is 100-120
        # Higher cadence in PD can indicate festination
        cad = m.cadence
        if cad > 150:  # Very high cadence = mild issue
            csc = 2
        elif cad > 135:  # High cadence = slight issue
            csc = 1
        elif 100 <= cad <= 120:  # Normal range
            csc = 0
        elif 85 <= cad < 100 or 120 < cad <= 135:
            csc = 1
        else:
            csc = 2

        # Step height - PD4T calibrated: Score0=0.068, Score1=0.055, Score2=0.045
        sh = m.step_height_mean
        if sh < 0.1:  # Normalized
            shc = 0 if sh >= 0.060 else (1 if sh >= 0.050 else (2 if sh >= 0.040 else (3 if sh >= 0.030 else 4)))
        else:  # Real meters
            shc = 0 if sh >= 0.12 else (1 if sh >= 0.10 else (2 if sh >= 0.07 else (3 if sh >= 0.04 else 4)))

        # Stride length - PD4T calibrated: Score0=0.328, Score1=0.276, Score2=0.235
        sl = m.stride_length
        if sl < 0.5:  # Normalized data
            slc = 0 if sl >= 0.30 else (1 if sl >= 0.255 else (2 if sl >= 0.22 else 3))
        else:  # Real meters
            slc = 0 if sl >= 1.2 else (1 if sl >= 0.9 else (2 if sl >= 0.6 else 3))

        # Weighted score: arm swing most important for PD
        w = asc * 0.30 + ssc * 0.20 + csc * 0.15 + shc * 0.15 + slc * 0.20
        bs = min(4, int(w + 0.5))

        # Penalties for variability and asymmetry (PD4T has high variability values 25-45)
        pen = 0.0
        var_threshold = 25 if m.stride_variability > 1 else 0.08
        if m.stride_variability > var_threshold:
            pen += min(0.3, (m.stride_variability - var_threshold) / (var_threshold * 2.5) * 0.3)
        if m.arm_swing_asymmetry > 0.20: pen += min(0.4, (m.arm_swing_asymmetry - 0.20) / 0.30 * 0.4)
        if m.festination_index > 0.05: pen += min(0.3, m.festination_index * 2)
        if m.step_count < 10: pen += 0.2

        tot = min(4.0, bs + pen)
        return UPDRSScore(round(tot, 1), bs, round(pen, 2), _get_severity(tot),
            {"arm": asc, "speed": ssc, "cadence": csc, "step_h": shc, "stride": slc}, "rule", 1.0)
