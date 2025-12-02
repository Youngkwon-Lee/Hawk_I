from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from services.model_registry import ModelRegistry
import traceback

class ModelSelectorAgent(BaseAgent):
    def __init__(self):
        self.registry = ModelRegistry()

    def process(self, ctx: AnalysisContext) -> AnalysisContext:
        try:
            task_type = ctx.task_type
            ctx.log("model_selector", f"Selecting model for task: {task_type}")

            # 1. Get Data for Verification
            landmarks = ctx.skeleton_data.get("landmarks", [])
            if not landmarks:
                ctx.log("model_selector", "No landmarks available for model verification.")
                return ctx

            # 2. Find Best Model
            model = self.registry.find_best_model(task_type, data=landmarks)
            
            if not model:
                ctx.log("model_selector", f"No suitable model found for task: {task_type}")
                # We don't fail hard here, we just don't calculate metrics
                # Or we could have a default "fallback" logic if needed
                return ctx

            ctx.log("model_selector", f"Selected Model: {model.metadata.name} (v{model.metadata.version}) [{model.metadata.model_type}]")

            # 3. Apply Model
            try:
                # The model returns a metrics object (dataclass) or dict
                result = model.process(landmarks, context=ctx)
                
                # Store result in context
                # We assume the result is the metrics object expected by ClinicalAgent
                # But we need to handle it generically.
                # For now, let's assume it returns the metrics object or dict that we put into kinematic_metrics
                
                from dataclasses import asdict, is_dataclass
                
                if is_dataclass(result):
                    ctx.kinematic_metrics = asdict(result)
                    # We might want to store the raw object too if needed for scoring
                    # But ClinicalAgent currently expects a dict in kinematic_metrics
                    # AND the object for UPDRS scoring.
                    # This is a bit of a coupling issue. 
                    # Ideally, UPDRS scoring should also be a "Model" or part of the pipeline.
                    # For now, let's store the object in a temporary place or standardize.
                    ctx.latest_metrics_obj = result 
                elif isinstance(result, dict):
                    ctx.kinematic_metrics = result
                    ctx.latest_metrics_obj = None # Can't score if we don't have the object? 
                    # Actually UPDRSScorer expects the object.
                    # We should probably make sure our models return the object.
                
                ctx.log("model_selector", "Model execution successful.")

            except Exception as e:
                ctx.log("model_selector", f"Model execution failed: {e}")
                ctx.error = f"Model {model.metadata.name} failed: {e}"
                # Fallback logic could go here (try next best model)

            return ctx

        except Exception as e:
            ctx.error = str(e)
            ctx.log("model_selector", f"Critical Error: {traceback.format_exc()}")
            return ctx
