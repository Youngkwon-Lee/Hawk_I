from dataclasses import asdict
from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from services.metrics_calculator import MetricsCalculator
from services.updrs_scorer import UPDRSScorer

from agents.model_selector_agent import ModelSelectorAgent
from models.rule_based import register_rule_based_models
from models.ml_wrapper import FingerTappingMLModel
from services.model_registry import ModelRegistry

# CORAL Ordinal Regression
from services.skeleton_converter import convert_landmarks_to_array, get_task_type_for_coral
from services.ml_scorer import get_ml_scorer

class ClinicalAgent(BaseAgent):
    def __init__(self):
        # Initialize Registry with default models
        register_rule_based_models()
        
        # Register ML Models
        registry = ModelRegistry()
        registry.register(FingerTappingMLModel("finger_tapping"))
        # Add other tasks if needed, e.g. hand_movement
        
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

            # ========================================
            # CORAL Ordinal Regression Path (Priority)
            # ========================================
            if ctx.scoring_method == "coral":
                coral_result = self._score_with_coral(ctx, landmarks, task_type)
                if coral_result:
                    ctx.clinical_scores = coral_result
                    ctx.log("clinical", f"CORAL Score: {coral_result['total_score']} ({coral_result['severity']})", meta=coral_result)

                    # Still calculate metrics for visualization (optional)
                    ctx = self.model_selector.process(ctx)
                    self.generate_charts(ctx)
                    ctx.status = "clinical_done"
                    return ctx
                else:
                    ctx.log("clinical", "CORAL scoring failed, falling back to rule-based.")
                    ctx.scoring_method = "rule"  # Fallback

            # 1. Calculate Kinematic Metrics via Model Selector
            ctx = self.model_selector.process(ctx)
            
            if ctx.error:
                ctx.log("clinical", f"Model Selector failed: {ctx.error}")
                # Status is already set to error by ModelSelector if critical
                return ctx

            metrics = ctx.kinematic_metrics
            metrics_obj = getattr(ctx, 'latest_metrics_obj', None)
            
            # Check if the model performed direct scoring (e.g. ML model)
            is_direct_scoring = False
            if isinstance(metrics, dict) and metrics.get("is_direct_scoring"):
                is_direct_scoring = True
                ctx.log("clinical", "Model used direct scoring. Using model outputs.")
                
                # Extract score from metrics dict
                ctx.clinical_scores = {
                    "total_score": metrics.get("total_score"),
                    "severity": metrics.get("severity"),
                    "details": metrics.get("details", {})
                }
                
                score_val = metrics.get("total_score")
                severity = metrics.get("severity")
                ctx.log("clinical", f"UPDRS Estimation (ML): Score {score_val} ({severity})", meta=ctx.clinical_scores)

            else:
                # Standard path: Metrics -> UPDRSScorer
                
                # Log key metrics for reasoning UI
                msg_parts = []
                if "speed" in metrics:
                    msg_parts.append(f"Speed: {metrics['speed']:.2f}")
                if "amplitude" in metrics:
                    msg_parts.append(f"Amp: {metrics['amplitude']:.2f}")
                
                summary_msg = ", ".join(msg_parts) if msg_parts else "Metrics calculated."
                ctx.log("clinical", f"Kinematic features extracted. {summary_msg}", meta=metrics)

                # 2. UPDRS Scoring
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
                        ctx.log("clinical", f"No scoring model for task type: {task_type}")
                        score_result_obj = None
                else:
                    score_result_obj = None

                if score_result_obj:
                    try:
                        ctx.clinical_scores = asdict(score_result_obj)
                    except TypeError:
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

    def _score_with_coral(self, ctx: AnalysisContext, landmarks: list, task_type: str) -> dict:
        """
        Score using CORAL Ordinal Regression (Mamba + K-Fold trained models)

        Best performing models:
        - Gait: Pearson 0.790
        - Finger: Pearson 0.553
        - Hand: Pearson 0.598
        - Leg: Pearson 0.238

        Args:
            ctx: Analysis context
            landmarks: Raw MediaPipe landmarks from VisionAgent
            task_type: Task type (gait, finger_tapping, hand_movement, leg_agility)

        Returns:
            dict with total_score, severity, details, method, confidence or None if failed
        """
        try:
            # Map to CORAL task type
            coral_task = get_task_type_for_coral(task_type)
            ctx.log("clinical", f"CORAL: Converting {task_type} -> {coral_task}")

            # Convert landmarks to numpy array
            skeleton_array = convert_landmarks_to_array(landmarks, coral_task)

            if skeleton_array is None:
                ctx.log("clinical", "CORAL: Failed to convert landmarks to array")
                return None

            ctx.log("clinical", f"CORAL: Skeleton shape {skeleton_array.shape}")

            # Get CORAL prediction
            ml_scorer = get_ml_scorer()
            prediction = ml_scorer.predict_coral(skeleton_array, coral_task)

            if prediction is None:
                ctx.log("clinical", "CORAL: Model prediction failed")
                return None

            # Map score to severity
            score = prediction.score
            if score < 0.5:
                severity = "Normal"
            elif score < 1.5:
                severity = "Slight"
            elif score < 2.5:
                severity = "Mild"
            elif score < 3.5:
                severity = "Moderate"
            else:
                severity = "Severe"

            return {
                "total_score": score,
                "base_score": score,
                "penalties": 0.0,
                "severity": severity,
                "details": {
                    "method": "coral_mamba",
                    "task": coral_task,
                    "expected_score": prediction.raw_prediction,
                    "model_type": prediction.model_type
                },
                "method": "coral",
                "confidence": prediction.confidence
            }

        except Exception as e:
            ctx.log("clinical", f"CORAL: Error - {e}")
            return None

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

