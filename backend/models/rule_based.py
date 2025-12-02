from services.model_registry import AnalysisModel, ModelMetadata, ModelRegistry
from services.metrics_calculator import MetricsCalculator
from typing import Any, List

class FingerTappingRuleBasedModel(AnalysisModel):
    def __init__(self, task_type: str):
        super().__init__(ModelMetadata(
            name="RuleBasedFingerTapping",
            version="1.0",
            task_type=task_type,
            model_type="rule_based",
            priority=10,
            description="Classic heuristic-based analysis using MediaPipe hand landmarks."
        ))
        self.calculator = MetricsCalculator()

    def verify(self, data: Any) -> bool:
        # Basic validation: check if it's a list of frames
        if not isinstance(data, list):
            return False
        if len(data) == 0:
            return False
        return True

    def process(self, data: Any, context: Any = None) -> Any:
        # Delegate to existing calculator
        return self.calculator.calculate_finger_tapping_metrics(data)

class GaitRuleBasedModel(AnalysisModel):
    def __init__(self, task_type: str):
        super().__init__(ModelMetadata(
            name="RuleBasedGait",
            version="1.0",
            task_type=task_type,
            model_type="rule_based",
            priority=10,
            description="Classic heuristic-based analysis using MediaPipe pose landmarks."
        ))
        self.calculator = MetricsCalculator()

    def verify(self, data: Any) -> bool:
        if not isinstance(data, list):
            return False
        if len(data) == 0:
            return False
        return True

    def process(self, data: Any, context: Any = None) -> Any:
        return self.calculator.calculate_gait_metrics(data)

def register_rule_based_models():
    """Register all rule-based models with the registry"""
    registry = ModelRegistry()
    
    # Hand Tasks
    hand_tasks = ["finger_tapping", "hand_movement", "pronation_supination"]
    for task in hand_tasks:
        registry.register(FingerTappingRuleBasedModel(task))
        
    # Gait Tasks
    gait_tasks = ["gait", "leg_agility", "walking"]
    for task in gait_tasks:
        registry.register(GaitRuleBasedModel(task))

    print("Registered rule-based models.")
