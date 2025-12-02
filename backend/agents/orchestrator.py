from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from agents.vision_agent import VisionAgent
from agents.clinical_agent import ClinicalAgent
from agents.gait_cycle_agent import GaitCycleAgent
from agents.report_agent import ReportAgent
import traceback

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        self.vision_agent = VisionAgent()
        self.clinical_agent = ClinicalAgent()
        self.gait_cycle_agent = GaitCycleAgent()
        self.report_agent = ReportAgent()

    def process(self, ctx: AnalysisContext, on_progress_update=None) -> AnalysisContext:
        """
        Main workflow: Vision -> Clinical -> GaitCycle (if gait) -> Report
        Supports dynamic routing and error handling.
        """
        try:
            ctx.log("orchestrator", "Starting collaborative analysis workflow.")

            # --- Step 1: Vision ---
            if on_progress_update: on_progress_update("roi_detection", "in_progress")
            ctx = self.vision_agent.process(ctx)
            
            # Check for critical vision failure
            if ctx.error and not ctx.vision_meta:
                ctx.log("orchestrator", f"Critical Vision Failure: {ctx.error}")
                if on_progress_update: on_progress_update("roi_detection", "failed")
                return ctx
            
            # Check confidence
            confidence = 0.0
            if ctx.vision_meta and "confidence" in ctx.vision_meta:
                confidence = ctx.vision_meta["confidence"]
            
            if confidence < 0.4:
                ctx.log("orchestrator", f"Warning: Low vision confidence ({confidence:.2f}). Analysis may be unreliable.")
                # We continue, but flag it
                ctx.log("orchestrator", "Proceeding with caution.")

            if on_progress_update: 
                on_progress_update("roi_detection", "completed")
                on_progress_update("skeleton", "completed")
                on_progress_update("heatmap", "completed")

            # --- Step 2: Clinical ---
            if on_progress_update: on_progress_update("metrics", "in_progress")
            
            # Dynamic Routing: If vision failed partially (no skeleton), skip clinical
            if not ctx.skeleton_data or not ctx.skeleton_data.get("landmarks"):
                ctx.log("orchestrator", "No skeleton data available. Skipping Clinical Agent.")
                if on_progress_update: on_progress_update("metrics", "failed")
            else:
                ctx = self.clinical_agent.process(ctx)
                if ctx.error:
                    ctx.log("orchestrator", f"Clinical Agent failed: {ctx.error}. Proceeding to Report to explain failure.")
                    if on_progress_update: on_progress_update("metrics", "failed")
                else:
                    if on_progress_update: 
                        on_progress_update("metrics", "completed")
                        on_progress_update("updrs_calculation", "completed")

            # --- Step 3: Gait Cycle Analysis (for gait tasks) ---
            if ctx.task_type in ['gait', 'leg_agility']:
                if on_progress_update: on_progress_update("gait_cycle", "in_progress")
                ctx = self.gait_cycle_agent.process(ctx)
                if on_progress_update: on_progress_update("gait_cycle", "completed")

            # --- Step 4: Report (Interpretation) ---
            if on_progress_update: on_progress_update("ai_interpretation", "in_progress")

            # Even if clinical failed, we might have visual data to report on
            ctx = self.report_agent.process(ctx)
            
            if ctx.error:
                ctx.log("orchestrator", f"Report generation failed.")
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
