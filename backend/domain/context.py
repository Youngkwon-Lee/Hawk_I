from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
import uuid

class ReasoningStep(BaseModel):
    """A single step in the agent's reasoning process, to be displayed in the UI."""
    step: str  # e.g., "vision", "clinical", "report"
    message: str  # User-facing explanation
    timestamp: datetime = Field(default_factory=datetime.now)
    meta: Optional[Dict[str, Any]] = None  # Extra data (scores, confidence, etc.)

class Report(BaseModel):
    """Structured report output."""
    summary_for_patient: str
    summary_for_clinician: str
    recommendations: List[str]
    key_findings: List[str] = []

class AnalysisContext(BaseModel):
    """Shared state object passed between agents."""
    video_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    video_path: str
    
    # Workflow Status
    status: Literal["pending", "vision_done", "clinical_done", "report_done", "error"] = "pending"
    error: Optional[str] = None

    # Scoring Configuration
    scoring_method: Literal["rule", "ml", "ensemble"] = "ensemble"
    ml_model_type: str = "rf"

    # Data Slots (Populated by Agents)
    task_type: Optional[str] = None
    skeleton_data: Optional[Dict[str, Any]] = None
    vision_meta: Optional[Dict[str, Any]] = None  # frame_rate, quality_flag, heatmap_path, etc.
    
    kinematic_metrics: Optional[Dict[str, Any]] = None
    latest_metrics_obj: Any = None # Temporary storage for the raw metrics object (for scoring)
    clinical_scores: Optional[Dict[str, Any]] = None
    clinical_charts: Optional[str] = None # Markdown/Text representation for VLM

    
    report: Optional[Report] = None
    
    # Reasoning Log
    reasoning_log: List[ReasoningStep] = []

    def log(self, step: str, message: str, meta: Optional[Dict[str, Any]] = None):
        """Helper to add a log entry."""
        self.reasoning_log.append(ReasoningStep(
            step=step,
            message=message,
            meta=meta
        ))
