from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelMetadata:
    name: str
    version: str
    task_type: str  # e.g., "finger_tapping", "gait"
    model_type: str # "rule_based", "ml", "hybrid"
    priority: int   # Higher number = higher priority
    description: str = ""

class AnalysisModel(ABC):
    """Abstract base class for all analysis models"""
    
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata

    @abstractmethod
    def process(self, data: Any, context: Any = None) -> Any:
        """
        Process the input data and return results.
        Args:
            data: The input data (e.g., landmarks)
            context: Optional full AnalysisContext if needed
        Returns:
            The analysis result (metrics, score, etc.)
        """
        pass

    @abstractmethod
    def verify(self, data: Any) -> bool:
        """
        Check if the model can handle this specific data.
        """
        pass

class ModelRegistry:
    """
    Central repository for analysis models.
    Singleton pattern to ensure one registry instance.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.models = {} # Dict[task_type, List[AnalysisModel]]
        return cls._instance

    def register(self, model: AnalysisModel):
        """Register a new model"""
        task = model.metadata.task_type
        if task not in self.models:
            self.models[task] = []
        
        self.models[task].append(model)
        # Sort by priority (descending)
        self.models[task].sort(key=lambda x: x.metadata.priority, reverse=True)
        print(f"[ModelRegistry] Registered model: {model.metadata.name} (v{model.metadata.version}) for {task}")

    def get_models(self, task_type: str) -> List[AnalysisModel]:
        """Get all models for a task type, sorted by priority"""
        return self.models.get(task_type, [])

    def find_best_model(self, task_type: str, data: Any = None) -> Optional[AnalysisModel]:
        """
        Find the highest priority model that verifies against the data.
        """
        candidates = self.get_models(task_type)
        for model in candidates:
            if data is not None:
                if model.verify(data):
                    return model
            else:
                return model
        return None
