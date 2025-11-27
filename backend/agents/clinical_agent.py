from dataclasses import asdict
from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from services.metrics_calculator import MetricsCalculator
from services.updrs_scorer import UPDRSScorer

class ClinicalAgent(BaseAgent):
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.updrs_scorer = UPDRSScorer()

    def process(self, ctx: AnalysisContext) -> AnalysisContext:
        try:
            if ctx.status != "vision_done":
                ctx.log("clinical", "Skipping clinical analysis: Vision not completed.")
                return ctx

            ctx.log("clinical", "Clinical Agent started analysis.")

            landmarks = ctx.skeleton_data.get("landmarks", [])
            task_type = ctx.task_type
            fps = ctx.vision_meta.get("fps", 30.0)

            # 1. Calculate Kinematic Metrics
            try:
                # Map task types to metric calculators
                hand_tasks = ["finger_tapping", "hand_movement", "pronation_supination"]
                gait_tasks = ["gait", "leg_agility", "walking"]

                if task_type in hand_tasks:
                    metrics_obj = self.metrics_calculator.calculate_finger_tapping_metrics(landmarks)
                elif task_type in gait_tasks:
                    metrics_obj = self.metrics_calculator.calculate_gait_metrics(landmarks)
                else:
                    ctx.log("clinical", f"No metric calculation for task type: {task_type}")
                    metrics_obj = None

                if metrics_obj:
                    metrics = asdict(metrics_obj)
                    ctx.kinematic_metrics = metrics
                else:
                    metrics = {}
                    ctx.kinematic_metrics = {}
            except Exception as e:
                ctx.log("clinical", f"Error calculating metrics: {e}")
                metrics = {}
                ctx.kinematic_metrics = {}
                ctx.error = f"Metric calculation failed: {e}"
                ctx.status = "error"  # 상태를 error로 변경
                return ctx
            
            # Log key metrics for reasoning UI
            msg_parts = []
            if "speed" in metrics:
                msg_parts.append(f"Speed: {metrics['speed']:.2f}")
            if "amplitude" in metrics:
                msg_parts.append(f"Amp: {metrics['amplitude']:.2f}")
            
            summary_msg = ", ".join(msg_parts) if msg_parts else "Metrics calculated."
            ctx.log("clinical", f"Kinematic features extracted. {summary_msg}", meta=metrics)

            # 2. UPDRS Scoring (pass the original metrics object, not the dict)
            if task_type in hand_tasks:
                score_result_obj = self.updrs_scorer.score_finger_tapping(metrics_obj)
            elif task_type in gait_tasks:
                score_result_obj = self.updrs_scorer.score_gait(metrics_obj)
            else:
                # Default/Fallback
                ctx.log("clinical", f"No scoring model for task type: {task_type}")
                score_result_obj = None

            if score_result_obj:
                try:
                    ctx.clinical_scores = asdict(score_result_obj)
                except TypeError:
                    # Fallback if asdict fails (e.g. in tests if mocking is weird)
                    ctx.clinical_scores = {
                        "total_score": score_result_obj.total_score,
                        "severity": score_result_obj.severity,
                        "details": score_result_obj.details
                    }

                score_val = score_result_obj.total_score
                severity = score_result_obj.severity
                ctx.log("clinical", f"UPDRS Estimation: Score {score_val} ({severity})", meta=ctx.clinical_scores)
            else:
                ctx.clinical_scores = {}
                ctx.log("clinical", "UPDRS Scoring skipped.")

            ctx.status = "clinical_done"
            return ctx

        except Exception as e:
            ctx.error = str(e)
            ctx.log("clinical", f"Error in Clinical Agent: {e}")
            raise e
