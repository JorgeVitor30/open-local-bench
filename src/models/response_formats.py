from pydantic import BaseModel, Field
from typing import List


class MathResponse(BaseModel):
    """Response format for math problems."""
    answer: int = Field(description="The numerical answer")
    explanation: str = Field(description="Step by step explanation")


class TextResponse(BaseModel):
    """Response format for text generation."""
    text: str = Field(description="The generated text")


class ConnectionsResponse(BaseModel):
    """Response format for Connections game."""
    groups: List[List[str]] = Field(description="Four groups of four words each")
    themes: List[str] = Field(description="The theme for each group")
