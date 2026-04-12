"""
Compatibility shim for legacy model registry imports.

This repo currently expects `models.rule_based.register_rule_based_models()`
from the older backend layout. The file was removed in the working tree, but
ClinicalAgent still imports it at startup. We provide a minimal implementation
that registers metrics-extraction models backed by `MetricsCalculator`.
"""

from __future__ import annotations

from services.metrics_calculator import MetricsCalculator
from services.model_registry import AnalysisModel, ModelMetadata, ModelRegistry


class _FingerRuleModel(AnalysisModel):
    def __init__(self):
        super().__init__(
            ModelMetadata(
                name="finger_rule_metrics",
                version="compat-v1",
                task_type="finger_tapping",
                model_type="rule_based",
                priority=10,
                description="Compatibility metrics extractor for finger tapping.",
            )
        )
        self.calculator = MetricsCalculator()

    def process(self, data, context=None):
        return self.calculator.calculate_finger_tapping_metrics(data)

    def verify(self, data) -> bool:
        return bool(data)


class _GaitRuleModel(AnalysisModel):
    def __init__(self):
        super().__init__(
            ModelMetadata(
                name="gait_rule_metrics",
                version="compat-v1",
                task_type="gait",
                model_type="rule_based",
                priority=10,
                description="Compatibility metrics extractor for gait.",
            )
        )
        self.calculator = MetricsCalculator()

    def process(self, data, context=None):
        return self.calculator.calculate_gait_metrics(data)

    def verify(self, data) -> bool:
        return bool(data)


def register_rule_based_models() -> None:
    registry = ModelRegistry()
    existing = {
        (model.metadata.task_type, model.metadata.name)
        for task_models in registry.models.values()
        for model in task_models
    }
    for model in (_FingerRuleModel(), _GaitRuleModel()):
        key = (model.metadata.task_type, model.metadata.name)
        if key not in existing:
            registry.register(model)
