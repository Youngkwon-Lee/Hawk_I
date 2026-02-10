from typing import Any, Dict
from services.model_registry import AnalysisModel, ModelMetadata
from models.lstm_pytorch_model import get_finger_tapping_lstm_scorer
import logging

logger = logging.getLogger(__name__)

class FingerTappingMLModel(AnalysisModel):
    def __init__(self, task_type: str):
        super().__init__(ModelMetadata(
            name="LSTMFingerTapping",
            version="2.0",
            task_type=task_type,
            model_type="ml",
            priority=20,  # Higher priority than rule-based (10)
            description="Deep Learning model (LSTM+RF) for Finger Tapping analysis"
        ))
        # Lazy loading handled by the scorer itself
        self.scorer = get_finger_tapping_lstm_scorer()

    def verify(self, data: Any) -> bool:
        # Check if landmarks are provided
        if not isinstance(data, list):
            return False
        if len(data) == 0:
            return False
        return True

    def process(self, data: Any, context: Any = None) -> Any:
        # data is list of landmarks
        prediction = self.scorer.predict(data)
        
        # Determine metrics to return
        # Since this is an end-to-end scorer, it returns the final score.
        # However, ClinicalAgent might expect metrics to populate the UI.
        # The LSTM model extracts "clinical features" internally, but currently
        # doesn't expose them in a friendly way for the UI.
        
        # Ideally, we should modify the scorer to return intermediate metrics if possible.
        # For now, we return a special structure that ClinicalAgent will recognize.
        
        return {
            "is_direct_scoring": True,
            "total_score": prediction.score,
            "confidence": prediction.confidence,
            "severity": self._get_severity(prediction.score),
            "details": {
                "raw_score": prediction.raw_score,
                "lstm_score": prediction.lstm_score,
                "rf_score": prediction.rf_score,
                "model_type": prediction.model_type
            },
             # We might need to mock or calculate basic metrics for the UI table
             # depending on what the frontend expects.
             # For now, let's leave metrics empty or minimal.
             # ClinicalAgent currently logs metrics.
        }

    def _get_severity(self, score: float) -> str:
        if score < 0.5: return "Normal"
        elif score < 1.5: return "Slight"
        elif score < 2.5: return "Mild"
        elif score < 3.5: return "Moderate"
        else: return "Severe"
