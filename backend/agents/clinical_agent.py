from dataclasses import asdict
from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from services.metrics_calculator import MetricsCalculator
from services.updrs_scorer import UPDRSScorer

from agents.model_selector_agent import ModelSelectorAgent
from models.rule_based import register_rule_based_models

class ClinicalAgent(BaseAgent):
    def __init__(self):
        # Initialize Registry with default models
        register_rule_based_models()
        
        self.model_selector = ModelSelectorAgent()
        # UPDRSScorer is created per-request based on ctx.scoring_method

    def process(self, ctx: AnalysisContext) -> AnalysisContext:
        try:
            if ctx.status != "vision_done":
                ctx.log("clinical", "Skipping clinical analysis: Vision not completed.")
                return ctx

            ctx.log("clinical", "Clinical Agent started analysis.")

            landmarks = ctx.skeleton_data.get("landmarks", [])
            task_type = ctx.task_type
            fps = ctx.vision_meta.get("fps", 30.0)

            # 1. Calculate Kinematic Metrics via Model Selector
            ctx = self.model_selector.process(ctx)
            
            if ctx.error:
                ctx.log("clinical", f"Model Selector failed: {ctx.error}")
                # Status is already set to error by ModelSelector if critical
                return ctx

            metrics = ctx.kinematic_metrics
            metrics_obj = getattr(ctx, 'latest_metrics_obj', None)
            
            # Log key metrics for reasoning UI
            msg_parts = []
            if "speed" in metrics:
                msg_parts.append(f"Speed: {metrics['speed']:.2f}")
            if "amplitude" in metrics:
                msg_parts.append(f"Amp: {metrics['amplitude']:.2f}")
            
            summary_msg = ", ".join(msg_parts) if msg_parts else "Metrics calculated."
            ctx.log("clinical", f"Kinematic features extracted. {summary_msg}", meta=metrics)

            # 2. UPDRS Scoring (pass the original metrics object, not the dict)
            # We need to know if it's hand or gait for scoring
            # Ideally this should also be dynamic, but for now we keep the mapping here
            hand_tasks = ["finger_tapping", "hand_movement", "pronation_supination"]
            gait_tasks = ["gait", "leg_agility", "walking"]

            if metrics_obj:
                # Create scorer with method from context
                scorer = UPDRSScorer(method=ctx.scoring_method, ml_model_type=ctx.ml_model_type)
                ctx.log("clinical", f"Using scoring method: {ctx.scoring_method} (model: {ctx.ml_model_type})")

                if task_type in hand_tasks:
                    score_result_obj = scorer.score_finger_tapping(metrics_obj)
                elif task_type in gait_tasks:
                    score_result_obj = scorer.score_gait(metrics_obj)
                else:
                    # Default/Fallback
                    ctx.log("clinical", f"No scoring model for task type: {task_type}")
                    score_result_obj = None
            else:
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

            # 3. Generate Charts for VLM
            self.generate_charts(ctx)

            ctx.status = "clinical_done"
            return ctx

        except Exception as e:
            ctx.error = str(e)
            ctx.log("clinical", f"Error in Clinical Agent: {e}")
            raise e

    def generate_charts(self, ctx: AnalysisContext):
        """
        Format metrics into a text-based representation (Markdown) for the VLM.
        """
        try:
            metrics = ctx.kinematic_metrics
            scores = ctx.clinical_scores
            
            if not metrics:
                ctx.clinical_charts = "No metrics available."
                return

            charts = []
            
            # 1. Metrics Table
            charts.append("### Kinematic Metrics")
            charts.append("| Metric | Value | Unit |")
            charts.append("| :--- | :--- | :--- |")
            
            for key, value in metrics.items():
                # Skip complex objects if any
                if isinstance(value, (int, float)):
                    unit = ""
                    if "speed" in key: unit = "Hz/px/s"
                    elif "amplitude" in key: unit = "px"
                    elif "opening" in key: unit = "px"
                    
                    charts.append(f"| {key} | {value:.2f} | {unit} |")
                elif isinstance(value, str):
                    charts.append(f"| {key} | {value} | - |")
            
            charts.append("")
            
            # 2. UPDRS Scores
            if scores:
                charts.append("### UPDRS Estimation")
                charts.append(f"- **Total Score**: {scores.get('total_score', 'N/A')}")
                charts.append(f"- **Severity**: {scores.get('severity', 'N/A')}")
                
                details = scores.get('details', {})
                if details:
                    charts.append("\n**Detailed Scores:**")
                    for k, v in details.items():
                        charts.append(f"- {k}: {v}")
            
            ctx.clinical_charts = "\n".join(charts)
            ctx.log("clinical", "Generated clinical charts for VLM.")
            
        except Exception as e:
            ctx.log("clinical", f"Failed to generate charts: {e}")
            ctx.clinical_charts = "Error generating charts."

