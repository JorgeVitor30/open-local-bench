"""
Runner for all Reasoning tests.

Executes all reasoning benchmark tests and aggregates results.
Each test class must have:
- __init__(model_name: str)
- run() -> result
- check_result(result) -> tuple[bool, float, str]
"""

from typing import List, Any
from src.tests.reasoning.runner_zebra import ZebraPuzzleTest


def run_reasoning_tests(model_name: str) -> List[dict]:
    """
    Run all reasoning tests.
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        List of test results with status, score, and error info
    """
    print("=" * 60)
    print("REASONING TESTS")
    print("=" * 60)
    
    test_classes: list[tuple[str, Any]] = [
        ("zebra_puzzle", ZebraPuzzleTest),
    ]
    
    results = []
    total = len(test_classes)
    
    for i, (test_name, TestClass) in enumerate(test_classes, 1):
        print(f"\n[{i}/{total}] {test_name}")
        print("-" * 40)
        
        try:
            test = TestClass(model_name=model_name)
            result = test.run()  
            
            passed, score, message = test.check_result(result)
            print(f"✓ Result: {message}")
            print(f"✓ Score: {score:.2%}")
            
            results.append({
                "test": test_name,
                "passed": passed,
                "score": score,
                "message": message,
                "status": "completed"
            })
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results.append({
                "test": test_name,
                "error": str(e),
                "status": "failed"
            })
    
    print("\n" + "=" * 60)
    print("REASONING SUMMARY")
    print("=" * 60)
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    
    print(f"Completed: {completed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Average Score: {avg_score:.2%}")
    
    return results
