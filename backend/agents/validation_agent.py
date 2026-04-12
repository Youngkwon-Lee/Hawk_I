"""
ValidationAgent - Quality Assurance and Reliability Assessment

This agent validates the analysis pipeline results by:
1. Checking skeleton extraction quality
2. Validating metric calculations
3. Assessing score reliability
4. Detecting anomalies and outliers
5. Providing confidence-weighted final scores

Validation Checks:
- Frame coverage: % of frames with valid landmarks
- Landmark stability: Movement consistency across frames
- Metric range: Values within expected physiological ranges
- Score consistency: Agreement between rule-based and ML scores
- Temporal coherence: Logical progression of metrics over time
"""

from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class ValidationResult:
    """Result of validation checks"""
    is_valid: bool
    overall_confidence: float
    quality_score: float  # 0-1
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class ValidationAgent(BaseAgent):
    """Agent for validating analysis results and assessing reliability"""

    # Thresholds for validation checks
    THRESHOLDS = {
        # Skeleton quality
        'min_frame_coverage': 0.7,       # At least 70% frames with landmarks
        'min_landmark_confidence': 0.5,  # Minimum landmark detection confidence
        'max_jitter_ratio': 0.3,         # Maximum acceptable position jitter

        # Metric ranges (physiologically plausible)
        'finger_tapping': {
            'speed_min': 0.5, 'speed_max': 8.0,  # taps per second
            'amplitude_min': 0.01, 'amplitude_max': 0.5,  # normalized
        },
        'gait': {
            'cadence_min': 40, 'cadence_max': 140,  # steps per minute
            'stride_length_min': 0.2, 'stride_length_max': 2.0,  # meters
            'cycle_time_min': 0.5, 'cycle_time_max': 3.0,  # seconds
        },

        # Score validation
        'max_rule_ml_diff': 2.0,         # Max acceptable difference between methods
        'min_confidence': 0.5,           # Minimum acceptable confidence

        # Temporal coherence
        'max_metric_jump': 3.0,          # Max std devs for sudden metric changes
    }

    def __init__(self):
        pass

    def process(self, ctx: AnalysisContext) -> AnalysisContext:
        """
        Validate analysis results and compute reliability scores

        Args:
            ctx: Analysis context with completed analysis

        Returns:
            Updated context with validation results
        """
        try:
            # Only run after clinical analysis
            if ctx.status not in ['clinical_done', 'report_done']:
                ctx.log("validation", "Skipping: Analysis not complete")
                return ctx

            ctx.log("validation", "Starting validation checks...")

            checks = {}
            issues = []
            warnings = []

            # 1. Validate skeleton quality
            skeleton_result = self._validate_skeleton(ctx)
            checks['skeleton'] = skeleton_result
            issues.extend(skeleton_result.get('issues', []))
            warnings.extend(skeleton_result.get('warnings', []))

            # 2. Validate metrics
            metrics_result = self._validate_metrics(ctx)
            checks['metrics'] = metrics_result
            issues.extend(metrics_result.get('issues', []))
            warnings.extend(metrics_result.get('warnings', []))

            # 3. Validate scores
            scores_result = self._validate_scores(ctx)
            checks['scores'] = scores_result
            issues.extend(scores_result.get('issues', []))
            warnings.extend(scores_result.get('warnings', []))

            # 4. Temporal coherence (if gait data available)
            if ctx.gait_cycle_data:
                temporal_result = self._validate_temporal_coherence(ctx)
                checks['temporal'] = temporal_result
                issues.extend(temporal_result.get('issues', []))
                warnings.extend(temporal_result.get('warnings', []))

            # Calculate overall quality score
            quality_scores = [
                checks.get('skeleton', {}).get('quality', 0.5),
                checks.get('metrics', {}).get('quality', 0.5),
                checks.get('scores', {}).get('quality', 0.5),
            ]
            if 'temporal' in checks:
                quality_scores.append(checks['temporal'].get('quality', 0.5))

            overall_quality = float(np.mean(quality_scores))

            # Determine overall confidence
            confidence_factors = [
                checks.get('skeleton', {}).get('confidence', 0.5),
                checks.get('scores', {}).get('confidence', 0.5),
            ]
            overall_confidence = float(np.mean(confidence_factors))

            # Determine if valid
            is_valid = (
                len([i for i in issues if 'critical' in i.lower()]) == 0 and
                overall_quality >= 0.4 and
                overall_confidence >= 0.4
            )

            # Create validation result
            validation = ValidationResult(
                is_valid=is_valid,
                overall_confidence=round(overall_confidence, 3),
                quality_score=round(overall_quality, 3),
                issues=issues,
                warnings=warnings,
                checks=checks
            )

            # Store in context
            ctx.validation_result = validation.to_dict()

            # Log results
            status_emoji = "PASS" if is_valid else "FAIL"
            ctx.log("validation",
                f"Validation {status_emoji}: Quality={overall_quality:.2f}, Confidence={overall_confidence:.2f}",
                meta={
                    'is_valid': is_valid,
                    'quality_score': overall_quality,
                    'confidence': overall_confidence,
                    'issue_count': len(issues),
                    'warning_count': len(warnings)
                })

            if issues:
                ctx.log("validation", f"Issues: {', '.join(issues[:3])}")
            if warnings:
                ctx.log("validation", f"Warnings: {', '.join(warnings[:3])}")

            ctx.log("validation", "Validation completed")

        except Exception as e:
            ctx.log("validation", f"Validation error: {str(e)}")
            import traceback
            traceback.print_exc()

        return ctx

    def _validate_skeleton(self, ctx: AnalysisContext) -> Dict[str, Any]:
        """Validate skeleton extraction quality"""
        result = {
            'passed': True,
            'quality': 0.5,
            'confidence': 0.5,
            'issues': [],
            'warnings': [],
            'details': {}
        }

        landmarks = ctx.skeleton_data.get('landmarks', []) if ctx.skeleton_data else []

        if not landmarks:
            result['passed'] = False
            result['quality'] = 0.0
            result['issues'].append("Critical: No skeleton data available")
            return result

        total_frames = ctx.vision_meta.get('frame_count', len(landmarks)) if ctx.vision_meta else len(landmarks)

        # Frame coverage
        frame_coverage = len(landmarks) / max(total_frames, 1)
        result['details']['frame_coverage'] = round(frame_coverage, 3)

        if frame_coverage < self.THRESHOLDS['min_frame_coverage']:
            result['warnings'].append(f"Low frame coverage: {frame_coverage:.1%}")

        # Landmark confidence (average visibility/confidence if available)
        confidences = []
        for frame in landmarks:
            keypoints = frame.get('landmarks', frame.get('keypoints', []))
            for kp in keypoints:
                if 'visibility' in kp:
                    confidences.append(kp['visibility'])
                elif 'confidence' in kp:
                    confidences.append(kp['confidence'])

        if confidences:
            avg_confidence = float(np.mean(confidences))
            result['details']['avg_landmark_confidence'] = round(avg_confidence, 3)

            if avg_confidence < self.THRESHOLDS['min_landmark_confidence']:
                result['warnings'].append(f"Low landmark confidence: {avg_confidence:.2f}")
        else:
            avg_confidence = 0.7  # Default if not available

        # Position jitter (stability check)
        jitter_ratio = self._calculate_jitter(landmarks)
        result['details']['jitter_ratio'] = round(jitter_ratio, 3)

        if jitter_ratio > self.THRESHOLDS['max_jitter_ratio']:
            result['warnings'].append(f"High position jitter: {jitter_ratio:.2f}")

        # Calculate quality score
        quality = (
            0.4 * frame_coverage +
            0.3 * avg_confidence +
            0.3 * (1.0 - min(jitter_ratio, 1.0))
        )
        result['quality'] = round(quality, 3)
        result['confidence'] = round(avg_confidence, 3)

        result['passed'] = quality >= 0.5

        return result

    def _calculate_jitter(self, landmarks: List[Dict]) -> float:
        """Calculate position jitter as indicator of tracking stability"""
        if len(landmarks) < 3:
            return 0.0

        try:
            # Extract first landmark position across frames
            positions = []
            for frame in landmarks:
                keypoints = frame.get('landmarks', frame.get('keypoints', []))
                if keypoints:
                    kp = keypoints[0]
                    positions.append([kp.get('x', 0), kp.get('y', 0)])

            positions = np.array(positions)
            if len(positions) < 3:
                return 0.0

            # Calculate velocity (first derivative)
            velocities = np.diff(positions, axis=0)

            # Calculate acceleration (second derivative)
            accelerations = np.diff(velocities, axis=0)

            # Jitter = ratio of acceleration magnitude to velocity magnitude
            vel_mag = np.mean(np.linalg.norm(velocities, axis=1))
            acc_mag = np.mean(np.linalg.norm(accelerations, axis=1))

            if vel_mag < 1e-6:
                return 0.0

            return float(acc_mag / vel_mag)

        except Exception:
            return 0.0

    def _validate_metrics(self, ctx: AnalysisContext) -> Dict[str, Any]:
        """Validate kinematic metrics are within physiological ranges"""
        result = {
            'passed': True,
            'quality': 0.5,
            'issues': [],
            'warnings': [],
            'details': {}
        }

        metrics = ctx.kinematic_metrics
        if not metrics:
            result['quality'] = 0.0
            result['issues'].append("No kinematic metrics available")
            return result

        task_type = ctx.task_type or 'finger_tapping'
        thresholds = self.THRESHOLDS.get(
            'finger_tapping' if task_type in ['finger_tapping', 'hand_movement'] else 'gait',
            {}
        )

        out_of_range = []

        # Check speed/cadence
        if task_type in ['finger_tapping', 'hand_movement']:
            speed = metrics.get('tapping_speed', metrics.get('speed', 0))
            if speed < thresholds.get('speed_min', 0) or speed > thresholds.get('speed_max', 100):
                out_of_range.append(f"speed={speed:.2f}")
                result['details']['speed_out_of_range'] = True

            amplitude = metrics.get('amplitude_mean', metrics.get('amplitude', 0))
            if amplitude < thresholds.get('amplitude_min', 0) or amplitude > thresholds.get('amplitude_max', 1):
                out_of_range.append(f"amplitude={amplitude:.3f}")
                result['details']['amplitude_out_of_range'] = True
        else:
            cadence = metrics.get('cadence', 0)
            if cadence < thresholds.get('cadence_min', 0) or cadence > thresholds.get('cadence_max', 200):
                out_of_range.append(f"cadence={cadence:.1f}")

            stride = metrics.get('stride_length', 0)
            if stride < thresholds.get('stride_length_min', 0) or stride > thresholds.get('stride_length_max', 3):
                out_of_range.append(f"stride={stride:.2f}m")

        if out_of_range:
            result['warnings'].append(f"Metrics out of range: {', '.join(out_of_range)}")

        # Calculate quality based on how many metrics are in range
        total_checks = len(thresholds) // 2  # Each metric has min/max
        issues_count = len(out_of_range)
        result['quality'] = round(1.0 - (issues_count / max(total_checks, 1)), 3)

        result['passed'] = issues_count == 0

        return result

    def _validate_scores(self, ctx: AnalysisContext) -> Dict[str, Any]:
        """Validate UPDRS scores and method agreement"""
        result = {
            'passed': True,
            'quality': 0.5,
            'confidence': 0.5,
            'issues': [],
            'warnings': [],
            'details': {}
        }

        scores = ctx.clinical_scores
        if not scores:
            result['quality'] = 0.0
            result['issues'].append("No clinical scores available")
            return result

        # Extract scores
        total_score = scores.get('total_score', scores.get('score', 0))
        confidence = scores.get('confidence', 0.5)
        details = scores.get('details', {})

        result['details']['total_score'] = total_score
        result['details']['confidence'] = confidence

        # Check confidence
        if confidence < self.THRESHOLDS['min_confidence']:
            result['warnings'].append(f"Low confidence score: {confidence:.2f}")

        # Check rule vs ML agreement
        rule_score = details.get('rule', total_score)
        ml_score = details.get('ml', total_score)
        score_diff = abs(rule_score - ml_score)

        result['details']['rule_score'] = rule_score
        result['details']['ml_score'] = ml_score
        result['details']['score_difference'] = score_diff

        if score_diff > self.THRESHOLDS['max_rule_ml_diff']:
            result['warnings'].append(
                f"Large rule-ML disagreement: rule={rule_score:.1f}, ml={ml_score:.1f}"
            )

        # Calculate quality
        agreement_quality = 1.0 - (score_diff / 4.0)  # Normalize by max score range
        result['quality'] = round((agreement_quality + confidence) / 2, 3)
        result['confidence'] = round(confidence, 3)

        result['passed'] = score_diff <= self.THRESHOLDS['max_rule_ml_diff']

        return result

    def _validate_temporal_coherence(self, ctx: AnalysisContext) -> Dict[str, Any]:
        """Validate temporal consistency of gait cycles"""
        result = {
            'passed': True,
            'quality': 0.5,
            'issues': [],
            'warnings': [],
            'details': {}
        }

        gait_data = ctx.gait_cycle_data
        if not gait_data:
            return result

        timing = gait_data.get('timing', {})
        cycle_time_cv = timing.get('cycle_time_cv_percent', 0)

        result['details']['cycle_time_cv'] = cycle_time_cv

        # High variability is a PD indicator but also affects reliability
        if cycle_time_cv > 50:
            result['warnings'].append(f"Very high gait variability: {cycle_time_cv:.1f}%")
            result['quality'] = 0.6  # Still valid, but lower quality

        # Check asymmetry
        asymmetry = gait_data.get('asymmetry_percent', {})
        max_asymmetry = max(
            asymmetry.get('stance_time', 0),
            asymmetry.get('swing_time', 0),
            asymmetry.get('step_length', 0)
        )

        result['details']['max_asymmetry'] = max_asymmetry

        if max_asymmetry > 30:
            result['warnings'].append(f"High asymmetry: {max_asymmetry:.1f}%")

        # Calculate quality
        variability_penalty = min(cycle_time_cv / 100, 0.3)
        asymmetry_penalty = min(max_asymmetry / 100, 0.2)
        result['quality'] = round(1.0 - variability_penalty - asymmetry_penalty, 3)

        result['passed'] = result['quality'] >= 0.5

        return result
