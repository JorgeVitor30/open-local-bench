import os
import mlflow
import time
from datetime import datetime
from typing import Any, Optional, Dict


class MLFlowBenchmarkLogger:
    """
    Professional logger for LLM benchmark tracking.
    
    Structure:
    - Experiment = Test category (reasoning, code, math, language)
    - Run = Individual test execution
    - Tags = model_name, test_name for filtering and grouping
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLFlow logger.
        
        Args:
            tracking_uri: Optional URI for MLFlow tracking server. 
                         If None, uses local ./mlflow.db
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            db_path = os.path.join(project_root, "mlflow.db")
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
        self._current_experiment: Optional[str] = None
    
    def set_experiment(self, category: str) -> None:
        """
        Set the experiment (test category).
        
        Args:
            category: Test category like 'reasoning', 'code', 'math', 'language'
        """
        experiment_name = f"benchmark_{category}"
        mlflow.set_experiment(experiment_name)
        self._current_experiment = experiment_name
        print(f"✓ Experiment set: '{experiment_name}'")
    
    def log_test(self,
                 test_name: str,
                 model_name: str,
                 category: str,
                 passed: bool,
                 score: float,
                 response: Any,
                 ground_truth: Optional[Any] = None,
                 prompt: Optional[str] = None,
                 temperature: float = 0.3,
                 latency_ms: Optional[float] = None,
                 extra_params: Optional[Dict[str, Any]] = None):
        """
        Log a test run to MLFlow.
        
        Args:
            test_name: Specific test name (e.g., 'zebra_puzzle')
            model_name: Model used (e.g., 'llama3.2', 'gpt-4')
            category: Test category for experiment grouping
            passed: Whether test passed
            score: Score from 0.0 to 1.0
            response: Model response
            ground_truth: Expected result
            prompt: Prompt sent to model
            temperature: Temperature used
            latency_ms: Response time in milliseconds
            extra_params: Additional parameters to log
        """
        if not self._current_experiment or category not in self._current_experiment:
            self.set_experiment(category)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}_{test_name}_{timestamp}"
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_name", test_name)
            mlflow.log_param("category", category)
            mlflow.log_param("temperature", temperature)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            if extra_params:
                for key, value in extra_params.items():
                    mlflow.log_param(key, value)
            
            mlflow.log_metric("score", score) 
            mlflow.log_metric("passed", 1.0 if passed else 0.0)  
            mlflow.log_metric("accuracy", score * 100) 
            
            if latency_ms:
                mlflow.log_metric("latency_ms", latency_ms)
            
            tags = {
                "model": model_name,
                "test": test_name,
                "category": category,
                "result": "passed" if passed else "failed",
                "benchmark_version": "1.0"
            }
            mlflow.set_tags(tags)
            
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
                response_text = str(response_dict)

                for key, value in response_dict.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"response_{key}", value)
                    elif isinstance(value, str) and len(value) < 100:
                        mlflow.log_param(f"response_{key}", value)
            else:
                response_text = str(response)
            
            mlflow.log_text(response_text, "response.json")
            
            if ground_truth:
                mlflow.log_text(str(ground_truth), "ground_truth.json")
            
            if prompt:
                prompt_truncated = prompt[:5000] + "..." if len(prompt) > 5000 else prompt
                mlflow.log_text(prompt_truncated, "prompt.txt")
            
            summary = f"""Test: {test_name}
            Model: {model_name}
            Score: {score:.2%}
            Passed: {passed}
            Timestamp: {datetime.now().isoformat()}
            """
            mlflow.log_text(summary, "summary.txt")
            
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run else "unknown"
            print(f"✓ Logged to MLFlow: run_id={run_id[:8]}...")
            print(f"  Experiment: {self._current_experiment}")
            print(f"  Model: {model_name} | Score: {score:.2%} | Passed: {passed}")
            
            return run_id
    
    def get_results_by_model(self, model_name: str, category: Optional[str] = None):
        """
        Get all results for a specific model.
        Useful for calculating averages.
        
        Returns:
            List of runs with metrics
        """
        import mlflow.tracking
        
        client = mlflow.tracking.MlflowClient()
        
        from mlflow.entities import Experiment, Run
        
        runs_list: list[Run] = []
        
        if category:
            experiment_name = f"benchmark_{category}"
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                return runs_list
            runs: list[Run] = list(client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.model = '{model_name}'"
            ))
            runs_list = runs
        else:
            experiments = client.search_experiments()
            for exp in experiments:
                exp_runs: list[Run] = list(client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"tags.model = '{model_name}'"
                ))
                runs_list.extend(exp_runs)
        
        return runs_list
    
    def get_average_score(self, model_name: str, category: Optional[str] = None) -> float:
        """
        Calculate average score for a model.
        
        Returns:
            Average score (0.0 to 1.0)
        """
        runs = self.get_results_by_model(model_name, category)
        if not runs:
            return 0.0
        
        scores = [r.data.metrics.get("score", 0) for r in runs]
        return sum(scores) / len(scores) if scores else 0.0


_benchmark_logger: Optional[MLFlowBenchmarkLogger] = None


def get_benchmark_logger() -> MLFlowBenchmarkLogger:
    """Get or create MLFlow benchmark logger singleton."""
    global _benchmark_logger
    if _benchmark_logger is None:
        _benchmark_logger = MLFlowBenchmarkLogger()
    return _benchmark_logger


def reset_logger():
    """Reset logger (useful for testing)."""
    global _benchmark_logger
    _benchmark_logger = None
