import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

TRACKING_FILE = "models/experiment_history.json"
REGISTRY_FILE = "models/champion_registry.json"

class ExperimentTracker:
    """
    A lightweight MLOps tracker to replace full servers like MLflow for this portable project.
    Logs experiment metrics, parameters, and model versions to a local JSON file.
    """
    
    def __init__(self, tracking_file: str = TRACKING_FILE):
        self.tracking_file = tracking_file
        self.logger = logging.getLogger(__name__)
        
    def _load_history(self) -> list:
        if not os.path.exists(self.tracking_file):
            return []
        try:
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def _save_history(self, history: list):
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(history, f, indent=2)

    def log_run(self, 
                model_name: str, 
                metrics: Dict[str, float], 
                params: Dict[str, Any], 
                tags: Optional[Dict[str, str]] = None):
        """
        Log a training run.
        
        Args:
            model_name: Name of the model artifact (e.g., "price_predictor_nn_v1").
            metrics: Dictionary of result metrics (e.g., {"mae": 50000, "r2": 0.85}).
            params: Dictionary of hyperparameters (e.g., {"epochs": 100, "lr": 0.001}).
            tags: Optional metadata.
        """
        history = self._load_history()
        
        run_entry = {
            "run_id": len(history) + 1,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "metrics": metrics,
            "parameters": params,
            "tags": tags or {}
        }
        
        history.append(run_entry)
        self._save_history(history)
        self.logger.info(f"Experiment logged: Run {run_entry['run_id']} for {model_name}")
        return run_entry['run_id']

    def _load_registry(self) -> Dict[str, Any]:
        if not os.path.exists(REGISTRY_FILE):
            return {}
        try:
            with open(REGISTRY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save_registry(self, registry: Dict[str, Any]):
        os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=2)

    def promote_to_champion(self, run_id: int):
        """
        Promotes a specific run to 'Champion' for its model type.
        """
        history = self._load_history()
        run_entry = next((r for r in history if r['run_id'] == run_id), None)
        if not run_entry:
            raise ValueError(f"Run {run_id} not found in history.")
        
        registry = self._load_registry()
        model_name = run_entry['model_name']
        registry[model_name] = run_entry
        self._save_registry(registry)
        self.logger.info(f"Model {model_name} (Run {run_id}) promoted to champion.")

    def get_champion(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the current champion run for a given model type.
        """
        registry = self._load_registry()
        return registry.get(model_name)

# Singleton instance for easy import
tracker = ExperimentTracker()
