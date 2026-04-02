from tests.test_abstract import TestAbstract
from prompts.math import SimpleMathPrompt, WordProblemPrompt


class AdditionTest(TestAbstract[int]):
    """Test basic addition."""
    
    def __init__(self, num1: int, num2: int):
        self.num1 = num1
        self.num2 = num2
        self._prompt_obj = SimpleMathPrompt()
    
    @property
    def name(self) -> str:
        return f"addition_{self.num1}_{self.num2}"
    
    @property
    def prompt_obj(self):
        return self._prompt_obj
    
    @property
    def prompt(self) -> str:
        return f"{self.prompt_obj.prompt}\n\nWhat is {self.num1} + {self.num2}?"
    
    @property
    def ground_truth(self) -> int:
        return self.num1 + self.num2
    
    def check_result(self, result: str) -> tuple[bool, float, str]:
        expected = str(self.ground_truth)
        if expected in result:
            return True, 1.0, "Correct!"
        # Try to extract number from response
        import re
        numbers = re.findall(r'\d+', result)
        if expected in numbers:
            return True, 1.0, "Correct!"
        return False, 0.0, f"Expected {expected}, got: {result}"


class TrainProblemTest(TestAbstract[int]):
    """Classic train word problem."""
    
    def __init__(self):
        self._prompt_obj = WordProblemPrompt()
    
    @property
    def name(self) -> str:
        return "train_problem_1"
    
    @property
    def prompt_obj(self):
        return self._prompt_obj
    
    @property
    def prompt(self) -> str:
        return f"{self.prompt_obj.prompt}\n\nTrain A leaves at 60mph, Train B leaves at 80mph. They leave at same time from same station. After 2 hours, what is the distance between them?"
    
    @property
    def ground_truth(self) -> int:
        return 40  # Distance between them
    
    def check_result(self, result: str) -> tuple[bool, float, str]:
        if "40" in result:
            return True, 1.0, "Correct!"
        return False, 0.0, "Expected 40 miles distance"


# List all math tests
MATH_TESTS = [
    AdditionTest(2, 2),
    AdditionTest(10, 15),
    TrainProblemTest(),
]
