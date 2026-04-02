from tests.test_abstract import TestAbstract
from prompts.language_comprehension import ConnectionsPrompt, SummarizationPrompt


class ConnectionsTest(TestAbstract[dict]):
    """Test the Connections word game."""
    
    def __init__(self, words: list[str], expected_groups: list[list[str]]):
        self.words = words
        self._expected_groups = expected_groups
        self._prompt_obj = ConnectionsPrompt()
    
    @property
    def name(self) -> str:
        return "connections_easy_1"
    
    @property
    def prompt_obj(self):
        return self._prompt_obj
    
    @property
    def prompt(self) -> str:
        words_str = ", ".join(self.words)
        return f"{self.prompt_obj.prompt}\n\nWords: {words_str}"
    
    @property
    def ground_truth(self) -> dict:
        return {"groups": self._expected_groups}
    
    def check_result(self, result: str) -> tuple[bool, float, str]:
        # Check if expected groups appear in response
        found_groups = 0
        for group in self._expected_groups:
            group_words = [w.lower() for w in group]
            if all(word in result.lower() for word in group_words):
                found_groups += 1
        
        score = found_groups / len(self._expected_groups)
        passed = found_groups == len(self._expected_groups)
        return passed, score, f"Found {found_groups}/{len(self._expected_groups)} groups"


class SummarizeTest(TestAbstract[str]):
    """Test text summarization."""
    
    def __init__(self, text: str, key_points: list[str]):
        self.text = text
        self.key_points = key_points
        self._prompt_obj = SummarizationPrompt()
    
    @property
    def name(self) -> str:
        return "summarize_news_1"
    
    @property
    def prompt_obj(self):
        return self._prompt_obj
    
    @property
    def prompt(self) -> str:
        return f"{self.prompt_obj.prompt}\n\nText: {self.text}"
    
    @property
    def ground_truth(self) -> None:
        return None  # No single right answer
    
    def check_result(self, result: str) -> tuple[bool, float, str]:
        # Check if summary covers key points
        result_lower = result.lower()
        found_points = sum(1 for point in self.key_points if point.lower() in result_lower)
        score = found_points / len(self.key_points)
        
        # Also check length
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        if len(sentences) > 5:
            score = score * 0.7  # Penalty for too long
        
        passed = score >= 0.7
        return passed, score, f"Covered {found_points}/{len(self.key_points)} key points"


# Example test data
CONNECTIONS_TEST_1 = ConnectionsTest(
    words=["RED", "BLUE", "GREEN", "YELLOW", "DOG", "CAT", "BIRD", "FISH", 
           "ONE", "TWO", "THREE", "FOUR", "APPLE", "BANANA", "ORANGE", "GRAPE"],
    expected_groups=[
        ["RED", "BLUE", "GREEN", "YELLOW"],  # Colors
        ["DOG", "CAT", "BIRD", "FISH"],      # Animals
        ["ONE", "TWO", "THREE", "FOUR"],     # Numbers
        ["APPLE", "BANANA", "ORANGE", "GRAPE"]  # Fruits
    ]
)

SUMMARIZE_TEST_1 = SummarizeTest(
    text="Scientists discovered a new planet 100 light years away. The planet is in the habitable zone and might have water. This discovery was made using the James Webb telescope. Researchers are excited about finding signs of life.",
    key_points=["new planet", "habitable zone", "water", "James Webb"]
)

LANGUAGE_TESTS = [
    CONNECTIONS_TEST_1,
    SUMMARIZE_TEST_1,
]
