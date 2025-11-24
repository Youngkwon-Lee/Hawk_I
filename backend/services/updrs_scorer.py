"""
Rule-based UPDRS Scoring Module
Converts kinematic metrics to MDS-UPDRS Part III scores (0-4)

Based on MDS-UPDRS Part III scoring criteria:
- 0: Normal
- 1: Slight (mild impairment)
- 2: Mild (moderate impairment)
- 3: Moderate (severe impairment)
- 4: Severe (cannot perform or requires assistance)
"""

from dataclasses import dataclass
from services.metrics_calculator import FingerTappingMetrics, GaitMetrics


@dataclass
class UPDRSScore:
    """UPDRS score with detailed breakdown"""
    total_score: float  # 0-4
    base_score: int  # Base score from primary metric
    penalties: float  # Additional penalties from secondary metrics
    severity: str  # "Normal", "Slight", "Mild", "Moderate", "Severe"
    details: dict  # Breakdown of scoring factors


class UPDRSScorer:
    """
    Rule-based UPDRS scorer using clinical thresholds

    References:
    - MDS-UPDRS Part III scoring criteria
    - Clinical literature on kinematic thresholds for PD
    """

    def __init__(self):
        pass

    def score_finger_tapping(self, metrics: FingerTappingMetrics) -> UPDRSScore:
        """
        Score finger tapping task (MDS-UPDRS 3.4)

        Primary metric: Tapping speed (Hz)
        Secondary metrics: Amplitude, hesitation, fatigue

        Clinical thresholds based on research:
        - Normal: ≥3.0 Hz, amplitude >40px, minimal fatigue
        - Slight: 2.0-3.0 Hz, mild amplitude decrement
        - Mild: 1.0-2.0 Hz, moderate amplitude decrement
        - Moderate: 0.5-1.0 Hz, severe slowing, hesitations
        - Severe: <0.5 Hz, barely performs

        Args:
            metrics: FingerTappingMetrics from metrics_calculator

        Returns:
            UPDRSScore with total score (0-4) and breakdown
        """
        # Primary scoring based on tapping speed
        speed = metrics.tapping_speed

        if speed >= 3.0:
            base_score = 0
            severity = "Normal"
        elif speed >= 2.0:
            base_score = 1
            severity = "Slight"
        elif speed >= 1.0:
            base_score = 2
            severity = "Mild"
        elif speed >= 0.5:
            base_score = 3
            severity = "Moderate"
        else:
            base_score = 4
            severity = "Severe"

        # Calculate penalties from secondary metrics
        penalties = 0.0
        penalty_details = {}

        # Amplitude penalty (amplitude decrement is a key PD symptom)
        # Normal: >0.8 (80% of index finger length), Abnormal: <0.8
        # Units: normalized by index finger length (dimensionless)
        if metrics.amplitude_mean < 0.8:
            amplitude_penalty = min(0.5, (0.8 - metrics.amplitude_mean) / 0.8 * 0.5)
            penalties += amplitude_penalty
            penalty_details['amplitude'] = amplitude_penalty

        # Fatigue penalty (progressive amplitude decrement)
        # Normal: <20%, Mild: 20-40%, Severe: >40%
        if metrics.fatigue_rate > 20:
            fatigue_penalty = min(0.5, (metrics.fatigue_rate - 20) / 40 * 0.5)
            penalties += fatigue_penalty
            penalty_details['fatigue'] = fatigue_penalty

        # Hesitation penalty
        # Normal: ≤2, Abnormal: >2
        if metrics.hesitation_count > 2:
            hesitation_penalty = min(0.3, (metrics.hesitation_count - 2) / 5 * 0.3)
            penalties += hesitation_penalty
            penalty_details['hesitation'] = hesitation_penalty

        # Rhythm variability penalty
        # Normal: <15%, Abnormal: >15%
        if metrics.rhythm_variability > 15:
            rhythm_penalty = min(0.2, (metrics.rhythm_variability - 15) / 20 * 0.2)
            penalties += rhythm_penalty
            penalty_details['rhythm'] = rhythm_penalty

        # Total score (capped at 4.0)
        total_score = min(4.0, base_score + penalties)

        # Update severity based on total score
        if total_score < 0.5:
            severity = "Normal"
        elif total_score < 1.5:
            severity = "Slight"
        elif total_score < 2.5:
            severity = "Mild"
        elif total_score < 3.5:
            severity = "Moderate"
        else:
            severity = "Severe"

        details = {
            'primary_metric': f'{speed:.2f} Hz',
            'base_score': base_score,
            'penalties': penalty_details,
            'amplitude': f'{metrics.amplitude_mean:.1f}px',
            'fatigue_rate': f'{metrics.fatigue_rate:.1f}%',
            'hesitation_count': metrics.hesitation_count,
            'rhythm_variability': f'{metrics.rhythm_variability:.1f}%'
        }

        return UPDRSScore(
            total_score=round(total_score, 1),
            base_score=base_score,
            penalties=round(penalties, 2),
            severity=severity,
            details=details
        )

    def score_gait(self, metrics: GaitMetrics) -> UPDRSScore:
        """
        Score gait task (MDS-UPDRS 3.10)

        Primary metrics: Walking speed, cadence
        Secondary metrics: Stride variability, arm swing asymmetry

        Clinical thresholds:
        - Normal: speed ≥0.8 m/s, cadence 100-120 steps/min
        - Slight: speed 0.6-0.8 m/s, mild variability
        - Mild: speed 0.4-0.6 m/s, moderate impairment
        - Moderate: speed 0.2-0.4 m/s, substantial impairment
        - Severe: speed <0.2 m/s, requires assistance

        Args:
            metrics: GaitMetrics from metrics_calculator

        Returns:
            UPDRSScore with total score (0-4) and breakdown
        """
        # Primary scoring based on walking speed
        speed = metrics.walking_speed
        cadence = metrics.cadence

        # Speed-based scoring
        if speed >= 0.8:
            speed_score = 0
        elif speed >= 0.6:
            speed_score = 1
        elif speed >= 0.4:
            speed_score = 2
        elif speed >= 0.2:
            speed_score = 3
        else:
            speed_score = 4

        # Cadence-based scoring (abnormal cadence indicates impairment)
        # Normal: 100-120 steps/min
        if 100 <= cadence <= 120:
            cadence_score = 0
        elif 80 <= cadence < 100 or 120 < cadence <= 140:
            cadence_score = 1
        elif 60 <= cadence < 80 or 140 < cadence <= 160:
            cadence_score = 2
        else:
            cadence_score = 3

        # Base score is weighted average (speed is more important)
        base_score = int((speed_score * 0.7 + cadence_score * 0.3) + 0.5)

        if base_score == 0:
            severity = "Normal"
        elif base_score == 1:
            severity = "Slight"
        elif base_score == 2:
            severity = "Mild"
        elif base_score == 3:
            severity = "Moderate"
        else:
            severity = "Severe"

        # Calculate penalties from secondary metrics
        penalties = 0.0
        penalty_details = {}

        # Stride variability penalty
        # Normal: <10%, Abnormal: >10%
        if metrics.stride_variability > 10:
            variability_penalty = min(0.5, (metrics.stride_variability - 10) / 20 * 0.5)
            penalties += variability_penalty
            penalty_details['stride_variability'] = variability_penalty

        # Arm swing asymmetry penalty
        # Normal: <10%, Abnormal: >10%
        if metrics.arm_swing_asymmetry > 10:
            asymmetry_penalty = min(0.5, (metrics.arm_swing_asymmetry - 10) / 20 * 0.5)
            penalties += asymmetry_penalty
            penalty_details['arm_swing_asymmetry'] = asymmetry_penalty

        # Step count penalty (too few steps suggests difficulty)
        # Expected: at least 10 steps for reliable assessment
        if metrics.step_count < 10:
            step_penalty = 0.3
            penalties += step_penalty
            penalty_details['insufficient_steps'] = step_penalty

        # Total score (capped at 4.0)
        total_score = min(4.0, base_score + penalties)

        # Update severity based on total score
        if total_score < 0.5:
            severity = "Normal"
        elif total_score < 1.5:
            severity = "Slight"
        elif total_score < 2.5:
            severity = "Mild"
        elif total_score < 3.5:
            severity = "Moderate"
        else:
            severity = "Severe"

        details = {
            'primary_metrics': f'{speed:.2f} m/s, {cadence:.0f} steps/min',
            'speed_score': speed_score,
            'cadence_score': cadence_score,
            'base_score': base_score,
            'penalties': penalty_details,
            'stride_variability': f'{metrics.stride_variability:.1f}%',
            'arm_swing_asymmetry': f'{metrics.arm_swing_asymmetry:.1f}%',
            'step_count': metrics.step_count
        }

        return UPDRSScore(
            total_score=round(total_score, 1),
            base_score=base_score,
            penalties=round(penalties, 2),
            severity=severity,
            details=details
        )


# Example usage
if __name__ == "__main__":
    from services.metrics_calculator import FingerTappingMetrics, GaitMetrics

    scorer = UPDRSScorer()

    # Test finger tapping scoring
    print("\n=== Finger Tapping UPDRS Scoring Test ===")

    # Normal case
    ft_normal = FingerTappingMetrics(
        tapping_speed=4.5,
        amplitude_mean=60.0,
        amplitude_std=5.0,
        rhythm_variability=8.0,
        fatigue_rate=10.0,
        hesitation_count=1,
        total_taps=45,
        duration=10.0
    )
    score = scorer.score_finger_tapping(ft_normal)
    print(f"\nNormal case: {score.total_score} ({score.severity})")
    print(f"Details: {score.details}")

    # Mild PD case
    ft_mild = FingerTappingMetrics(
        tapping_speed=1.8,
        amplitude_mean=35.0,
        amplitude_std=8.0,
        rhythm_variability=22.0,
        fatigue_rate=35.0,
        hesitation_count=4,
        total_taps=18,
        duration=10.0
    )
    score = scorer.score_finger_tapping(ft_mild)
    print(f"\nMild PD case: {score.total_score} ({score.severity})")
    print(f"Details: {score.details}")

    # Test gait scoring
    print("\n\n=== Gait UPDRS Scoring Test ===")

    # Normal case
    gait_normal = GaitMetrics(
        walking_speed=1.0,
        stride_length=0.75,
        cadence=110,
        stride_variability=8.0,
        arm_swing_asymmetry=7.0,
        step_count=22,
        duration=12.0
    )
    score = scorer.score_gait(gait_normal)
    print(f"\nNormal case: {score.total_score} ({score.severity})")
    print(f"Details: {score.details}")

    # Moderate PD case
    gait_moderate = GaitMetrics(
        walking_speed=0.45,
        stride_length=0.40,
        cadence=85,
        stride_variability=18.0,
        arm_swing_asymmetry=25.0,
        step_count=12,
        duration=15.0
    )
    score = scorer.score_gait(gait_moderate)
    print(f"\nModerate PD case: {score.total_score} ({score.severity})")
    print(f"Details: {score.details}")
