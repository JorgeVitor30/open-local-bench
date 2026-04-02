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
