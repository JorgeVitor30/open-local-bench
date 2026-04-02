from abc import ABC, abstractmethod
from typing import Optional, Any, Union
import sys
sys.path.insert(0, "/home/jorge/Documentos/mini-local-bench")


class TestAbstract(ABC):
    """
    Base class for all tests.

    A test runs a prompt against a model and checks the result.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the test."""
        pass

    @property
    @abstractmethod
    def prompt(self) -> str:
        """The actual prompt text to send to the model."""
        pass

    @property
    def ground_truth(self) -> Optional[Any]:
        """
        Expected result (optional).
        Can be int, str, bool, list, dict, or None if no right answer.
        """
        return None

    @abstractmethod
    def check_result(self, result: Union[str, Any]) -> tuple[bool, float, Optional[str]]:
        """
        Check if the model response is correct.
        
        Args:
            result: The model response, can be a string or a structured object (Pydantic model)
        
        Returns:
            tuple of (passed, score, message)
        """
        pass

    def _log_result(self, result: Any, passed: bool, score: float, message: Optional[str], model_name: str, category: str = "general") -> None:
        """
        Log test results using MLflow logger.
        
        This method can be overridden by subclasses to customize logging behavior
        (e.g., adding specific metrics for that test type).
        
        Args:
            result: The raw result from the model
            passed: Whether the test passed
            score: Score between 0.0 and 1.0
            message: Optional message explaining the result
            model_name: Name of the model being tested
            category: Category of the test (e.g., "reasoning", "code", "math")
        """
        try:
            from src.utils.mlflow_logger import get_benchmark_logger
            logger = get_benchmark_logger()
            logger.log_test(
                test_name=self.name,
                model_name=model_name,
                category=category,
                passed=passed,
                score=score,
                response=result,
                ground_truth=self.ground_truth,
                prompt=self.prompt,
            )
        except Exception as e:
            print(f"Warning: Failed to log to MLFlow: {e}")
