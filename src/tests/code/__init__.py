from tests.test_abstract import TestAbstract
from prompts.code import CodeGenerationPrompt, CodeExplanationPrompt


class CodeGenerationTest(TestAbstract[str]):
    """Test generating code."""
    
    def __init__(self, task: str, test_cases: list[tuple]):
        self.task = task
        self.test_cases = test_cases
        self._prompt_obj = CodeGenerationPrompt()
    
    @property
    def name(self) -> str:
        return f"code_gen_{self.task.replace(' ', '_')[:20]}"
    
    @property
    def prompt_obj(self):
        return self._prompt_obj
    
    @property
    def prompt(self) -> str:
        return f"{self.prompt_obj.prompt}\n\nTask: {self.task}"
    
    @property
    def ground_truth(self) -> None:
        return None  # Code can be correct in many ways
    
    def check_result(self, result: str) -> tuple[bool, float, str]:
        # Check if code contains Python function
        if "def " not in result:
            return False, 0.0, "No function definition found"
        
        # Try to extract and execute the code
        try:
            import re
            # Extract code block
            code_match = re.search(r'```python\n(.*?)```', result, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = result
            
            # Execute in safe environment
            local_ns = {}
            exec(code, {}, local_ns)
            
            # Find the function
            func = None
            for name, obj in local_ns.items():
                if callable(obj):
                    func = obj
                    break
            
            if func is None:
                return False, 0.3, "Code compiled but no function found"
            
            # Run test cases
            passed = 0
            for inputs, expected in self.test_cases:
                try:
                    if func(*inputs) == expected:
                        passed += 1
                except:
                    pass
            
            score = passed / len(self.test_cases)
            return score >= 1.0, score, f"Passed {passed}/{len(self.test_cases)} test cases"
            
        except Exception as e:
            return False, 0.1, f"Code error: {str(e)}"


class CodeExplanationTest(TestAbstract[str]):
    """Test explaining code."""
    
    def __init__(self, code: str, expected_concepts: list[str]):
        self.code = code
        self.expected_concepts = expected_concepts
        self._prompt_obj = CodeExplanationPrompt()
    
    @property
    def name(self) -> str:
        return "code_explain_loop"
    
    @property
    def prompt_obj(self):
        return self._prompt_obj
    
    @property
    def prompt(self) -> str:
        return f"{self.prompt_obj.prompt}\n\n```python\n{self.code}\n```"
    
    @property
    def ground_truth(self) -> None:
        return None
    
    def check_result(self, result: str) -> tuple[bool, float, str]:
        result_lower = result.lower()
        found = sum(1 for concept in self.expected_concepts if concept.lower() in result_lower)
        score = found / len(self.expected_concepts)
        passed = score >= 0.7
        return passed, score, f"Explained {found}/{len(self.expected_concepts)} key concepts"


# Example tests
FACTORIAL_TEST = CodeGenerationTest(
    task="Write a function to calculate factorial of a number",
    test_cases=[
        ((5,), 120),
        ((0,), 1),
        ((3,), 6),
    ]
)

EXPLAIN_LOOP = CodeExplanationTest(
    code="for i in range(10): print(i)",
    expected_concepts=["loop", "iteration", "print", "range"]
)

CODE_TESTS = [
    FACTORIAL_TEST,
    EXPLAIN_LOOP,
]
