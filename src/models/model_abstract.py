from abc import ABC, abstractmethod
from typing import Any


class ModelAbstract(ABC):
    """
    Base class for all models.
    
    A model runs prompts and returns responses.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model."""
        pass
    
    @abstractmethod
    def run(self, prompt: str, **kwargs) -> Any:
        """
        Run the model with a prompt.
        
        Args:
            prompt: The prompt to send
            **kwargs: Extra options like response_format, temperature, etc.
            
        Returns:
            Model response
        """
        pass
