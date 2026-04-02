from src.tests.reasoning.runner_zebra import run_zebra_puzzle_test
from src.utils.mlflow_logger import get_benchmark_logger

print("=" * 60)
print("Open Local Bench - Running Tests")
print("=" * 60)

logger = get_benchmark_logger()
logger.set_experiment("reasoning")  # Creates experiment 'benchmark_reasoning'
print("=" * 60)

print("\n Running Zebra Puzzle test...\n")
result = run_zebra_puzzle_test("llama3.1")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
