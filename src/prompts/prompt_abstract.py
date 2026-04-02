from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class PromptAbstract(ABC):
    """
    Base class for all prompts.
    
    All prompts must inherit from this class and
    implement the required properties.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the prompt."""
        pass
    
    @property
    @abstractmethod
    def prompt(self) -> str:
        """The actual prompt text."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Version of the prompt (like '1.0.0')."""
        pass
    
    @property
    def description(self) -> Optional[str]:
        """Optional description of what the prompt does."""
        return None
    
    @property
    def tags(self) -> list[str]:
        """Optional tags for organizing prompts."""
        return []
    
    @property
    def metadata(self) -> dict[str, Any]:
        """Extra info specific to this prompt type."""
        return {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert prompt to a dictionary."""
        return {
            "name": self.name,
            "prompt": self.prompt,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        return f"{self.name}@v{self.version}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}@v{self.version}>"
