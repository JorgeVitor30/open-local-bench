import json
from typing import Optional
from pydantic import BaseModel, Field
from src.models.ollama_model import OllamaModel
from src.utils.prompts import load_prompt
from src.tests.test_abstract import TestAbstract


class GPQAResponse(BaseModel):
    """Response structure for GPQA questions."""
    reasoning: str = Field(description="The step-by-step reasoning process")
    final_answer: str = Field(description="The final answer to the question")


class GPQATest(TestAbstract[GPQAResponse]):
    """
    GPQA (Google-Proof Q&A) reasoning test.
    
    Tests if the model can answer challenging questions that require
    deep reasoning and domain expertise. Iterates over all questions
    in the GPQA dataset.
    """
    
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.0):
        """
        Initialize GPQA test.
        
        Args:
            model_name: Name of the model to test
            temperature: Temperature for model generation
        """
        self._model_name = model_name
        self._template = load_prompt("gpqa_prompt", category="reasoning")
        self._temperature = temperature

        with open("src/datasets/gpqa_dataset.json", "r", encoding="utf-8") as f:
            self._dataset = json.load(f)
        
        self._current_index = 0
        self._current_item: Optional[dict] = None
    
    def _load_question(self, index: int) -> None:
        """Load a specific question from the dataset."""
        if index < len(self._dataset):
            self._current_item = self._dataset[index]
            self._current_index = index
    
    @property
    def name(self) -> str:
        return "gpqa_test"
    
    @property
    def prompt(self) -> str:
        if not self._current_item:
            return ""
        
        domain = self._current_item.get("subdomain", "")
        question = self._current_item.get("question", "")
        
        return self._template.replace("{{domain}}", domain).replace("{{question}}", question)
    
    @property
    def ground_truth(self) -> dict:
        """
        Ground truth answer from the dataset.
        """
        if not self._current_item:
            return {"final_answer": "", "domain": ""}
        
        return {
            "final_answer": self._current_item.get("correct_answer", ""),
            "domain": self._current_item.get("subdomain", ""),
            "explanation": self._current_item.get("explanation", "")
        }
    
    def check_result(self, result: GPQAResponse) -> tuple[bool, float, str]:
        """
        Check if the model's answer is correct.

        Args:
            result: GPQAResponse from the model

        Returns:
            tuple of (passed, score, message)
        """
        truth = self.ground_truth
        expected = truth["final_answer"].lower().strip()
        actual = result.final_answer.lower().strip()
        
        expected = expected.replace("\n", " ").replace("  ", " ").strip()
        actual = actual.replace("\n", " ").replace("  ", " ").strip()
        
        if actual == expected or expected in actual or actual in expected:
            return True, 1.0, f"Correct! Expected: {truth['final_answer']}"
        else:
            return False, 0.0, f"Incorrect. Expected: {truth['final_answer']}, Got: {result.final_answer}"
    
    def run_single(self, index: int) -> Optional[GPQAResponse]:
        """
        Run test for a single question at the given index.
        
        Args:
            index: Index of the question in the dataset
            
        Returns:
            GPQAResponse or None if index is out of range
        """
        if index >= len(self._dataset):
            print(f"Index {index} out of range. Dataset has {len(self._dataset)} questions.")
            return None
        
        self._load_question(index)
        
        if not self._current_item:
            print(f"Failed to load question at index {index}")
            return None
        
        print(f"\n{'='*60}")
        print(f"Question {index + 1}/{len(self._dataset)}")
        print(f"Domain: {self._current_item.get('subdomain', 'unknown')}")
        print(f"{'='*60}")

        model = OllamaModel(model_name=self._model_name)

        result = model.run(
            self.prompt,
            response_format=GPQAResponse,
            temperature=self._temperature
        )
        
        passed, score, message = self.check_result(result)
        self._log_result(result, passed, score, message, self._model_name, category="reasoning")
        
        print(f"Answer: {result.final_answer}")
        print(f"Expected: {self.ground_truth['final_answer']}")
        print(f"Status: {'✅ PASS' if passed else '❌ FAIL'} - {message}")
        
        return result
    
    def run(self, limit: int = 5) -> tuple[list[GPQAResponse], float]:
        """
        Execute the test for questions in the dataset.
        
        Args:
            limit: Maximum number of questions to run (default: 10)
            
        Returns:
            Tuple of (list of GPQAResponse, average score)
        """
        results: list[GPQAResponse] = []
        scores: list[float] = []
        total = min(limit, len(self._dataset))
        
        print(f"\nRunning GPQA test on {total} questions...")
        print(f"Dataset size: {len(self._dataset)} questions")
        
        for i in range(total):
            result = self.run_single(i)
            if result:
                results.append(result)
                passed, score, _ = self.check_result(result)
                scores.append(score)
        
        passed_count = sum(1 for s in scores if s > 0)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: {passed_count}/{len(results)} questions answered correctly")
        print(f"Average Score: {avg_score:.2%}")
        print(f"{'='*60}")
        
        return results, avg_score
    
    def get_dataset_info(self) -> dict:
        """Return information about the loaded dataset."""
        domains: list[str] = []
        for item in self._dataset:
            if isinstance(item, dict):
                domains.append(item.get("subdomain", "unknown"))
        return {
            "total_questions": len(self._dataset),
            "domains": list(set(domains))
        }


__all__ = ["GPQATest", "GPQAResponse"]
