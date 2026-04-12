"""
Compatibility shim for legacy FingerTappingMLModel import.

ClinicalAgent still imports `models.ml_wrapper.FingerTappingMLModel`.
We provide a lightweight registry model that reuses `MetricsCalculator` so the
backend can boot and the newer scoring paths can run.
"""

from __future__ import annotations

from services.metrics_calculator import MetricsCalculator
from services.model_registry import AnalysisModel, ModelMetadata


class FingerTappingMLModel(AnalysisModel):
    def __init__(self, task_type: str = "finger_tapping"):
        super().__init__(
            ModelMetadata(
                name="finger_ml_metrics_compat",
                version="compat-v1",
                task_type=task_type,
                model_type="ml",
                priority=20,
                description="Compatibility finger metrics extractor for legacy ClinicalAgent import.",
            )
        )
        self.calculator = MetricsCalculator()

    def process(self, data, context=None):
        return self.calculator.calculate_finger_tapping_metrics(data)

    def verify(self, data) -> bool:
        return bool(data)
