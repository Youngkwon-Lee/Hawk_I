from agents.base_agent import BaseAgent
from domain.context import AnalysisContext
from agents.vision_agent import VisionAgent
from agents.clinical_agent import ClinicalAgent
from agents.gait_cycle_agent import GaitCycleAgent
from agents.validation_agent import ValidationAgent
from agents.report_agent import ReportAgent
import traceback
import time

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        self.vision_agent = VisionAgent()
        self.clinical_agent = ClinicalAgent()
        self.gait_cycle_agent = GaitCycleAgent()
        self.validation_agent = ValidationAgent()
        self.report_agent = ReportAgent()

    def _log_step(self, step_name: str, status: str, elapsed: float = None):
        """Print colored log for each agent step"""
        elapsed_str = f" ({elapsed:.2f}s)" if elapsed else ""
        if status == "start":
            print(f"\n[AGENT] >>> {step_name} STARTED{elapsed_str}")
        elif status == "done":
            print(f"[AGENT] <<< {step_name} COMPLETED{elapsed_str}")
        elif status == "fail":
            print(f"[AGENT] !!! {step_name} FAILED{elapsed_str}")

    def process(self, ctx: AnalysisContext, on_progress_update=None) -> AnalysisContext:
        """
        Main workflow: Vision -> Clinical -> GaitCycle (if gait) -> Validation -> Report
        Supports dynamic routing and error handling.
        """
        try:
            total_start = time.time()
            ctx.log("orchestrator", "Starting collaborative analysis workflow.")
            print(f"\n{'='*60}")
            print(f"[ORCHESTRATOR] Analysis Pipeline Started")
            print(f"{'='*60}")

            # --- Step 1: Vision ---
            self._log_step("VisionAgent", "start")
            t0 = time.time()
            if on_progress_update: on_progress_update("roi_detection", "in_progress")
            ctx = self.vision_agent.process(ctx)
            vision_time = time.time() - t0

            # Check for critical vision failure
            if ctx.error and not ctx.vision_meta:
                self._log_step("VisionAgent", "fail", vision_time)
                ctx.log("orchestrator", f"Critical Vision Failure: {ctx.error}")
                if on_progress_update: on_progress_update("roi_detection", "failed")
                return ctx

            self._log_step("VisionAgent", "done", vision_time)

            # Check confidence
            confidence = 0.0
            if ctx.vision_meta and "confidence" in ctx.vision_meta:
                confidence = ctx.vision_meta["confidence"]

            if confidence < 0.4:
                ctx.log("orchestrator", f"Warning: Low vision confidence ({confidence:.2f}). Analysis may be unreliable.")
                ctx.log("orchestrator", "Proceeding with caution.")

            if on_progress_update:
                on_progress_update("roi_detection", "completed")
                on_progress_update("skeleton", "completed")
                on_progress_update("heatmap", "completed")

            # --- Step 2: Clinical ---
            self._log_step("ClinicalAgent", "start")
            t0 = time.time()
            if on_progress_update: on_progress_update("metrics", "in_progress")

            # Dynamic Routing: If vision failed partially (no skeleton), skip clinical
            if not ctx.skeleton_data or not ctx.skeleton_data.get("landmarks"):
                ctx.log("orchestrator", "No skeleton data available. Skipping Clinical Agent.")
                self._log_step("ClinicalAgent", "fail", time.time() - t0)
                if on_progress_update: on_progress_update("metrics", "failed")
            else:
                ctx = self.clinical_agent.process(ctx)
                clinical_time = time.time() - t0
                if ctx.error:
                    self._log_step("ClinicalAgent", "fail", clinical_time)
                    ctx.log("orchestrator", f"Clinical Agent failed: {ctx.error}. Proceeding to Report to explain failure.")
                    if on_progress_update: on_progress_update("metrics", "failed")
                else:
                    self._log_step("ClinicalAgent", "done", clinical_time)
                    if on_progress_update:
                        on_progress_update("metrics", "completed")
                        on_progress_update("updrs_calculation", "completed")

            # --- Step 3: Gait Cycle Analysis (for gait tasks) ---
            if ctx.task_type in ['gait', 'leg_agility']:
                self._log_step("GaitCycleAgent", "start")
                t0 = time.time()
                if on_progress_update: on_progress_update("gait_cycle", "in_progress")
                ctx = self.gait_cycle_agent.process(ctx)
                self._log_step("GaitCycleAgent", "done", time.time() - t0)
                if on_progress_update: on_progress_update("gait_cycle", "completed")

            # --- Step 4: Validation ---
            self._log_step("ValidationAgent", "start")
            t0 = time.time()
            if on_progress_update: on_progress_update("validation", "in_progress")
            ctx = self.validation_agent.process(ctx)
            self._log_step("ValidationAgent", "done", time.time() - t0)
            if on_progress_update: on_progress_update("validation", "completed")

            # --- Step 5: Report (Interpretation) ---
            self._log_step("ReportAgent", "start")
            t0 = time.time()
            if on_progress_update: on_progress_update("ai_interpretation", "in_progress")

            # Even if clinical failed, we might have visual data to report on
            ctx = self.report_agent.process(ctx)
            report_time = time.time() - t0

            if ctx.error:
                self._log_step("ReportAgent", "fail", report_time)
                ctx.log("orchestrator", f"Report generation failed.")
                if on_progress_update: on_progress_update("ai_interpretation", "failed")
            else:
                self._log_step("ReportAgent", "done", report_time)
                if on_progress_update: on_progress_update("ai_interpretation", "completed")

            total_time = time.time() - total_start
            print(f"\n{'='*60}")
            print(f"[ORCHESTRATOR] Pipeline COMPLETED in {total_time:.2f}s")
            print(f"{'='*60}\n")
            ctx.log("orchestrator", f"Workflow completed in {total_time:.2f}s")
            return ctx

        except Exception as e:
            ctx.error = str(e)
            ctx.log("orchestrator", f"Critical Orchestrator Error: {traceback.format_exc()}")
            return ctx

    def process_video(self, video_path: str) -> AnalysisContext:
        """Convenience entry point"""
        ctx = AnalysisContext(video_path=video_path)
        return self.process(ctx)
