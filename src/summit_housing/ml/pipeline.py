import yaml
import os
from summit_housing.ml.gbm import train_gbm, backtest_gbm
from summit_housing.ml.nn import train_macro_nn, backtest_nn
from summit_housing.tracking import tracker

CONFIG_PATH = "config/ml_config.yaml"

class MLPipeline:
    def __init__(self, config_path=CONFIG_PATH):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def run_full_evaluation(self, model_type="gbm"):
        """
        Runs backtesting for a model type and prints results.
        """
        if model_type == "gbm":
            metrics = backtest_gbm(n_folds=self.config['training']['temporal_split']['n_folds'])
        elif model_type == "nn":
            metrics = backtest_nn(n_folds=self.config['training']['temporal_split']['n_folds'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        avg_mae = sum(m['mae'] for m in metrics) / len(metrics)
        print(f"Final Evaluation for {model_type}: Avg MAE = ${avg_mae:,.0f}")
        return avg_mae

    def train_and_promote(self, model_type="gbm", params_override=None):
        """
        Trains a new model, logs it, and promotes it to champion if it meets criteria.
        """
        print(f"--- Starting Production Pipeline for {model_type} ---")
        
        if model_type == "gbm":
            pipeline, mae = train_gbm(params_override=params_override)
            # Fetch the run_id from history (latest)
            history = tracker._load_history()
            run_id = history[-1]['run_id']
        elif model_type == "nn":
            model, pre, scaler, mae, feat, ip, params = train_macro_nn(params_override=params_override)
            history = tracker._load_history()
            run_id = history[-1]['run_id']
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"Success: Model {model_type} v{run_id} trained with MAE ${mae:,.0f}")
        
        # Promotion Logic: Check if better than current champion
        champion = tracker.get_champion(model_type)
        if not champion or mae < champion['metrics']['mae']:
            print(f"New Champion! Promoting Run {run_id}...")
            tracker.promote_to_champion(run_id)
        else:
            print(f"Challenger (Run {run_id}) did not beat Champion (Run {champion['run_id']}).")
        
        return run_id

if __name__ == "__main__":
    import sys
    pipeline = MLPipeline()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    m_type = sys.argv[2] if len(sys.argv) > 2 else "gbm"
    
    if cmd == "train":
        pipeline.train_and_promote(m_type)
    else:
        pipeline.run_full_evaluation(m_type)
