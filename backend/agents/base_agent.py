from abc import ABC, abstractmethod
from domain.context import AnalysisContext

class BaseAgent(ABC):
    """Abstract base class for all HawkEye agents."""

    @abstractmethod
    def process(self, ctx: AnalysisContext) -> AnalysisContext:
        """
        Process the analysis context and return the updated context.
        Must be implemented by concrete agents.
        """
        pass
