from src.runners.runner_reasoning import run_reasoning_tests
from src.utils.mlflow_logger import get_benchmark_logger


def run_benchmark(
    categories: list[str] | None = None,
    model_name: str = "llama3.2",
    temperature: float = 0.0
):
    """
    Run benchmark tests for specified categories.

    Args:
        categories: List of categories to run (e.g., ["reasoning", "math"])
                   If None, runs all categories
        model_name: Name of the Ollama model to use
        temperature: Temperature for model generation (0.0 = deterministic)
    """
    print("=" * 60)
    print("Open Local Bench")
    print("=" * 60)
    
    all_results = {}
    
    available_categories = {
        "reasoning": run_reasoning_tests,
        # "math": run_math_tests,  # TODO: add when implemented
        # "code": run_code_tests,  # TODO: add when implemented
        # "language": run_language_tests,  # TODO: add when implemented
    }
    
    if categories is None:
        categories = list(available_categories.keys())
    
    print(f"\nRunning categories: {', '.join(categories)}")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print()
    
    for category in categories:
        if category not in available_categories:
            print(f"⚠️  Category '{category}' not available")
            continue
        
        logger = get_benchmark_logger()
        logger.set_experiment(category)
        
        runner = available_categories[category]
        results = runner(model_name=model_name, temperature=temperature)
        all_results[category] = results
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    for category, results in all_results.items():
        completed = sum(1 for r in results if r.get("status") == "completed")
        total = len(results)
        print(f"  {category}: {completed}/{total} tests passed")
    
    return all_results


if __name__ == "__main__":
    run_benchmark(model_name='llama3.2', temperature=0.0)
