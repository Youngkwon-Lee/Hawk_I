from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from agents.vision_agent import VisionAgent
from agents.clinical_agent import ClinicalAgent
from agents.report_agent import ReportAgent
import traceback

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        self.vision_agent = VisionAgent()
        self.clinical_agent = ClinicalAgent()
        self.report_agent = ReportAgent()

    def process(self, ctx: AnalysisContext, on_progress_update=None) -> AnalysisContext:
        """
        Main workflow: Vision -> Clinical -> Report
        Args:
            ctx: AnalysisContext
            on_progress_update: Callable(step_name, status) -> None
        """
        try:
            ctx.log("orchestrator", "Starting analysis workflow.")

            # Step 1: Vision
            if on_progress_update: on_progress_update("roi_detection", "in_progress")
            ctx = self.vision_agent.process(ctx)
            if ctx.error:
                ctx.log("orchestrator", f"Aborting due to Vision error: {ctx.error}")
                if on_progress_update: on_progress_update("roi_detection", "failed")
                return ctx
            
            if on_progress_update: 
                on_progress_update("roi_detection", "completed")
                on_progress_update("skeleton", "completed") # Vision agent does both

            # Step 2: Clinical
            if on_progress_update: on_progress_update("metrics", "in_progress")
            ctx = self.clinical_agent.process(ctx)
            if ctx.error:
                ctx.log("orchestrator", f"Aborting due to Clinical error: {ctx.error}")
                if on_progress_update: on_progress_update("metrics", "failed")
                return ctx
            
            if on_progress_update: 
                on_progress_update("metrics", "completed")
                on_progress_update("updrs_calculation", "completed")

            # Step 3: Report
            if on_progress_update: on_progress_update("ai_interpretation", "in_progress")
            ctx = self.report_agent.process(ctx)
            if ctx.error:
                ctx.log("orchestrator", f"Report generation failed, but analysis is available.")
                if on_progress_update: on_progress_update("ai_interpretation", "failed")
            else:
                if on_progress_update: on_progress_update("ai_interpretation", "completed")
            
            ctx.log("orchestrator", "Workflow completed.")
            return ctx

        except Exception as e:
            ctx.error = str(e)
            ctx.log("orchestrator", f"Critical Orchestrator Error: {traceback.format_exc()}")
            return ctx

    def process_video(self, video_path: str) -> AnalysisContext:
        """Convenience entry point"""
        ctx = AnalysisContext(video_path=video_path)
        return self.process(ctx)
