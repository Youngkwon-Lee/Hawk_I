"""
GaitCycleAgent - Specialized agent for detailed gait cycle analysis

This agent analyzes gait patterns to detect:
- Heel strike and toe-off events
- Gait cycle phases (stance/swing)
- Temporal and spatial parameters
- Asymmetry and variability metrics

These metrics are crucial for Parkinson's disease assessment as PD patients
often show:
- Reduced stride length
- Increased double support time
- Higher gait variability
- Asymmetry in timing
"""

from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from services.gait_cycle_analyzer import GaitCycleAnalyzer, GaitCycleAnalysis
from typing import Dict, Any, Optional
import traceback


class GaitCycleAgent(BaseAgent):
    """Agent for detailed gait cycle analysis"""

    # PD-specific thresholds based on literature
    THRESHOLDS = {
        'cycle_time_cv_normal': 3.0,      # CV% - normal < 3%
        'cycle_time_cv_mild': 5.0,        # mild PD: 3-5%
        'cycle_time_cv_moderate': 8.0,    # moderate PD: 5-8%

        'stance_percent_normal_min': 58.0,  # normal: 58-62%
        'stance_percent_normal_max': 62.0,

        'double_support_normal': 20.0,     # normal ~20%
        'double_support_pd': 30.0,         # PD patients often >30%

        'asymmetry_normal': 5.0,           # normal < 5%
        'asymmetry_mild': 10.0,            # mild: 5-10%
        'asymmetry_moderate': 15.0,        # moderate: 10-15%

        'step_length_cv_normal': 5.0,      # normal < 5%
        'step_length_cv_mild': 8.0,        # mild: 5-8%
    }

    def __init__(self):
        self.analyzer = None  # Lazy initialization

    def process(self, ctx: AnalysisContext) -> AnalysisContext:
        """
        Perform gait cycle analysis if task_type is gait-related

        Args:
            ctx: Analysis context with skeleton data

        Returns:
            Updated context with gait cycle analysis
        """
        try:
            # Only process gait-related tasks
            if ctx.task_type not in ['gait', 'leg_agility']:
                ctx.log("gait_cycle", f"Skipping: task_type is {ctx.task_type}")
                return ctx

            # Check prerequisites
            if ctx.status not in ['vision_done', 'clinical_done']:
                ctx.log("gait_cycle", "Skipping: Vision analysis not completed")
                return ctx

            landmarks = ctx.skeleton_data.get("landmarks", [])
            if len(landmarks) < 30:
                ctx.log("gait_cycle", f"Insufficient frames for gait analysis: {len(landmarks)}")
                return ctx

            ctx.log("gait_cycle", "Starting detailed gait cycle analysis...")

            # Initialize analyzer with video FPS
            fps = ctx.vision_meta.get("fps", 30.0)
            self.analyzer = GaitCycleAnalyzer(fps=fps)

            # Run analysis
            try:
                analysis = self.analyzer.analyze(landmarks)
                analysis_dict = self.analyzer.to_dict(analysis)
            except ValueError as e:
                ctx.log("gait_cycle", f"Analysis failed: {str(e)}")
                # Not a critical error - continue without gait cycle data
                return ctx

            # Store results in context
            ctx.gait_cycle_data = analysis_dict

            # Calculate PD-specific severity indicators
            pd_indicators = self._calculate_pd_indicators(analysis)
            ctx.gait_cycle_data['pd_indicators'] = pd_indicators

            # Log key findings
            summary = analysis_dict['summary']
            timing = analysis_dict['timing']

            ctx.log("gait_cycle",
                f"Detected {summary['total_cycles']} cycles "
                f"(L:{summary['num_cycles_left']}, R:{summary['num_cycles_right']})")

            ctx.log("gait_cycle",
                f"Cycle time: {timing['cycle_time_mean_sec']:.3f}s, "
                f"CV: {timing['cycle_time_cv_percent']:.1f}%",
                meta={
                    'cycle_time_mean': timing['cycle_time_mean_sec'],
                    'cycle_time_cv': timing['cycle_time_cv_percent'],
                    'total_cycles': summary['total_cycles']
                })

            # Log PD indicators
            severity = pd_indicators.get('overall_severity', 'unknown')
            ctx.log("gait_cycle",
                f"PD Gait Severity: {severity}",
                meta=pd_indicators)

            ctx.log("gait_cycle", "Gait cycle analysis completed successfully")

        except Exception as e:
            ctx.log("gait_cycle", f"Error in gait cycle analysis: {str(e)}")
            traceback.print_exc()
            # Non-critical error - don't fail the entire pipeline

        return ctx

    def _calculate_pd_indicators(self, analysis: GaitCycleAnalysis) -> Dict[str, Any]:
        """
        Calculate Parkinson's disease specific gait indicators

        Based on literature:
        - Morris et al. (1996) - PD gait characteristics
        - Hausdorff et al. (2003) - Gait variability in PD
        - Plotnik et al. (2005) - Asymmetry in PD gait
        """
        indicators = {}

        # 1. Gait Variability (CV of cycle time)
        cv = analysis.cycle_time_cv
        if cv < self.THRESHOLDS['cycle_time_cv_normal']:
            variability_severity = 'normal'
            variability_score = 0
        elif cv < self.THRESHOLDS['cycle_time_cv_mild']:
            variability_severity = 'mild'
            variability_score = 1
        elif cv < self.THRESHOLDS['cycle_time_cv_moderate']:
            variability_severity = 'moderate'
            variability_score = 2
        else:
            variability_severity = 'severe'
            variability_score = 3

        indicators['variability'] = {
            'cycle_time_cv': cv,
            'severity': variability_severity,
            'score': variability_score
        }

        # 2. Stance Phase Distribution
        stance = analysis.stance_percent_mean
        if self.THRESHOLDS['stance_percent_normal_min'] <= stance <= self.THRESHOLDS['stance_percent_normal_max']:
            stance_severity = 'normal'
            stance_score = 0
        elif stance > 65:  # Increased stance phase (slower gait)
            stance_severity = 'increased'
            stance_score = 2
        else:
            stance_severity = 'decreased'
            stance_score = 1

        indicators['stance_phase'] = {
            'percent': stance,
            'severity': stance_severity,
            'score': stance_score
        }

        # 3. Double Support Time
        ds = analysis.double_support_percent
        if ds < self.THRESHOLDS['double_support_normal']:
            ds_severity = 'normal'
            ds_score = 0
        elif ds < self.THRESHOLDS['double_support_pd']:
            ds_severity = 'mild'
            ds_score = 1
        else:
            ds_severity = 'increased'  # Characteristic of PD
            ds_score = 2

        indicators['double_support'] = {
            'percent': ds,
            'severity': ds_severity,
            'score': ds_score
        }

        # 4. Asymmetry
        # Use maximum asymmetry across metrics
        max_asymmetry = max(
            analysis.stance_time_asymmetry,
            analysis.swing_time_asymmetry,
            analysis.step_length_asymmetry
        )

        if max_asymmetry < self.THRESHOLDS['asymmetry_normal']:
            asym_severity = 'normal'
            asym_score = 0
        elif max_asymmetry < self.THRESHOLDS['asymmetry_mild']:
            asym_severity = 'mild'
            asym_score = 1
        elif max_asymmetry < self.THRESHOLDS['asymmetry_moderate']:
            asym_severity = 'moderate'
            asym_score = 2
        else:
            asym_severity = 'severe'
            asym_score = 3

        indicators['asymmetry'] = {
            'max_percent': max_asymmetry,
            'stance_time': analysis.stance_time_asymmetry,
            'swing_time': analysis.swing_time_asymmetry,
            'step_length': analysis.step_length_asymmetry,
            'severity': asym_severity,
            'score': asym_score
        }

        # 5. Step Length Variability
        sl_cv = analysis.step_length_cv
        if sl_cv < self.THRESHOLDS['step_length_cv_normal']:
            sl_severity = 'normal'
            sl_score = 0
        elif sl_cv < self.THRESHOLDS['step_length_cv_mild']:
            sl_severity = 'mild'
            sl_score = 1
        else:
            sl_severity = 'increased'
            sl_score = 2

        indicators['step_length_variability'] = {
            'cv': sl_cv,
            'severity': sl_severity,
            'score': sl_score
        }

        # 6. Overall Severity Score
        total_score = (
            variability_score +
            stance_score +
            ds_score +
            asym_score +
            sl_score
        )
        max_score = 3 + 2 + 2 + 3 + 2  # = 12

        # Map to severity levels
        if total_score == 0:
            overall_severity = 'Normal'
        elif total_score <= 3:
            overall_severity = 'Slight'
        elif total_score <= 6:
            overall_severity = 'Mild'
        elif total_score <= 9:
            overall_severity = 'Moderate'
        else:
            overall_severity = 'Severe'

        indicators['overall_severity'] = overall_severity
        indicators['total_score'] = total_score
        indicators['max_score'] = max_score
        indicators['normalized_score'] = total_score / max_score

        # 7. Clinical Notes
        notes = []
        if variability_score >= 2:
            notes.append("High gait variability suggests freezing risk")
        if ds_score >= 2:
            notes.append("Increased double support indicates cautious gait")
        if asym_score >= 2:
            notes.append("Significant asymmetry may indicate unilateral symptoms")
        if sl_score >= 1:
            notes.append("Variable step length suggests motor control difficulty")

        indicators['clinical_notes'] = notes

        return indicators
