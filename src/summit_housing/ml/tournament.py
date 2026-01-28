import os
import shutil
import json
import yaml
from summit_housing.ml.pipeline import MLPipeline

def cleanup():
    """Clears previous training results and validation logs."""
    print("--- Cleaning up previous ML artifacts ---")
    
    # Files to remove
    files_to_remove = [
        "models/experiment_history.json",
        "models/champion_registry.json",
    ]
    
    # Extensions to clear in models/
    extensions = [".pkl", ".pth", ".json"]
    if os.path.exists("models"):
        for f in os.listdir("models"):
            if any(f.endswith(ext) for ext in extensions):
                files_to_remove.append(os.path.join("models", f))
                
    for f_path in files_to_remove:
        if os.path.exists(f_path):
            os.remove(f_path)
            print(f"Deleted: {f_path}")

    # Clear validation docs
    if os.path.exists("docs/validation"):
        shutil.rmtree("docs/validation")
        os.makedirs("docs/validation")
        print("Cleared: docs/validation/")

def run_tournament(runs_per_model=5):
    """Performs a hyperparameter sweep for both GBM and NN models."""
    pipeline = MLPipeline()
    
    # 1. GBM Sweep Grid
    gbm_grid = [
        {"learning_rate": 0.01, "max_iter": 1000, "max_depth": 5},
        {"learning_rate": 0.01, "max_iter": 2000, "max_depth": 6},
        {"learning_rate": 0.05, "max_iter": 500,  "max_depth": 8},
        {"learning_rate": 0.02, "max_iter": 1500, "max_depth": 7},
        {"learning_rate": 0.1,  "max_iter": 300,  "max_depth": 5},
    ][:runs_per_model]

    # 2. NN Sweep Grid
    nn_grid = [
        {"lr": 0.001,  "epochs": 200, "batch_size": 128, "hidden_dims": [128, 64, 32]},
        {"lr": 0.0005, "epochs": 400, "batch_size": 64,  "hidden_dims": [256, 128, 64]},
        {"lr": 0.001,  "epochs": 300, "batch_size": 128, "hidden_dims": [64, 32]},
        {"lr": 0.002,  "epochs": 150, "batch_size": 256, "hidden_dims": [128, 128]},
        {"lr": 0.0001, "epochs": 500, "batch_size": 128, "hidden_dims": [128, 64, 32]},
    ][:runs_per_model]

    print(f"\n--- Starting GBM Tournament ({len(gbm_grid)} runs) ---")
    for i, params in enumerate(gbm_grid):
        print(f"\nRun {i+1}/{len(gbm_grid)}: {params}")
        pipeline.train_and_promote(model_type="gbm", params_override=params)

    print(f"\n--- Starting NN Tournament ({len(nn_grid)} runs) ---")
    for i, params in enumerate(nn_grid):
        print(f"\nRun {i+1}/{len(nn_grid)}: {params}")
        pipeline.train_and_promote(model_type="nn", params_override=params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated Model Tournament")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per model type")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip the initial cleanup phase")
    args = parser.parse_args()

    if not args.skip_cleanup:
        cleanup()
        
    run_tournament(runs_per_model=args.runs)
    print("\n--- Tournament Complete. Best models have been promoted to Champions! ---")
