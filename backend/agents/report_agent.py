from agents.base_agent import BaseAgent
from domain.context import AnalysisContext, Report
from services.interpretation_agent import InterpretationAgent

class ReportAgent(BaseAgent):
    def __init__(self):
        self.interpreter = InterpretationAgent()

    def process(self, ctx: AnalysisContext) -> AnalysisContext:
        try:
            if ctx.status != "clinical_done":
                ctx.log("report", "Skipping report: Clinical analysis not completed.")
                return ctx

            ctx.log("report", "Report Agent generating summary...")

            # Prepare data for interpretation service
            # The existing service expects specific args. We map context to them.
            updrs_score = ctx.clinical_scores.get("score", 0)
            severity = ctx.clinical_scores.get("severity", "Unknown")
            metrics = ctx.kinematic_metrics
            details = ctx.clinical_scores.get("details", {})
            task_type = ctx.task_type

            # Call existing service based on task type
            # Note: InterpretationAgent has specific methods like interpret_finger_tapping
            # We might need a dispatcher here.
            
            result = None
            if task_type == "finger_tapping":
                result = self.interpreter.interpret_finger_tapping(updrs_score, severity, metrics, details)
            elif task_type == "gait":
                result = self.interpreter.interpret_gait(updrs_score, severity, metrics, details)
            else:
                # Fallback or generic
                # For now, let's try to use one of them or a generic method if available
                # Assuming finger tapping as fallback for prototype if method missing
                result = self.interpreter.interpret_finger_tapping(updrs_score, severity, metrics, details)

            if result:
                ctx.report = Report(
                    summary_for_patient=result.summary,
                    summary_for_clinician=result.explanation, # Mapping explanation to clinician summary for now
                    recommendations=result.recommendations,
                    key_findings=[] # Populate if available
                )
                ctx.log("report", "Report generated successfully.")
            
            ctx.status = "report_done"
            return ctx

        except Exception as e:
            ctx.error = str(e)
            ctx.log("report", f"Error in Report Agent: {e}")
            # Don't raise here, just log error so partial results are returned
            return ctx
