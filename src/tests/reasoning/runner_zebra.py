from pydantic import BaseModel, Field
from src.models.ollama_model import OllamaModel
from src.utils.prompts import load_prompt
from src.tests.test_abstract import TestAbstract


class ZebraPuzzleResponse(BaseModel):
    water_drinker: str = Field(description="The nationality of the person who drinks water")
    zebra_owner: str = Field(description="The nationality of the person who owns the zebra")


class ZebraPuzzleTest(TestAbstract[ZebraPuzzleResponse]):
    """
    Zebra Puzzle reasoning test.
    
    Tests if the model can solve the classic logic puzzle
    using deductive reasoning.
    """
    
    def __init__(self, model_name: str = "llama3.2"):
        self._model_name = model_name
        self._prompt = load_prompt("zebra_prompt", category="reasoning")
    
    @property
    def name(self) -> str:
        return "zebra_puzzle"
    
    @property
    def prompt(self) -> str:
        return self._prompt
    
    @property
    def ground_truth(self) -> dict:
        """
        Ground truth solution:
        - Norwegian drinks water
        - Japanese owns the zebra
        """
        return {
            "water_drinker": "Norwegian",
            "zebra_owner": "Japanese"
        }
    
    def check_result(self, result: ZebraPuzzleResponse) -> tuple[bool, float, str]:
        """
        Check if the model's answer is correct using ground_truth.

        Args:
            result: ZebraPuzzleResponse from the model

        Returns:
            tuple of (passed, score, message)
        """
        truth = self.ground_truth
        
        correct_water = result.water_drinker.lower() == truth["water_drinker"].lower()
        correct_zebra = result.zebra_owner.lower() == truth["zebra_owner"].lower()

        if correct_water and correct_zebra:
            return True, 1.0, f"Correct: {truth['water_drinker']} drinks water, {truth['zebra_owner']} owns zebra"
        elif correct_water or correct_zebra:
            return False, 0.5, "Partially correct: got one answer right"
        else:
            return False, 0.0, "Incorrect: neither answer is correct"
    
    def run(self) -> ZebraPuzzleResponse:
        """Execute the test and return structured result."""
        model = OllamaModel(model_name=self._model_name)
        
        result = model.run(
            self.prompt,
            response_format=ZebraPuzzleResponse,
            temperature=0.3
        )
        
        passed, score, message = self.check_result(result)
        self._log_result(result, passed, score, message, self._model_name, category="reasoning")

        return result


__all__ = ["ZebraPuzzleTest", "ZebraPuzzleResponse"]
