import pandas as pd
import numpy as np
import joblib
import os
from summit_housing.ml.data import load_and_prep_data
from summit_housing.ml.validation import ResidualAnalyzer, TemporalFoldSplitter
from summit_housing.tracking import tracker

def evaluate_gbm(run_id=None):
    """
    Evaluates a specific GBM run on the most recent data.
    """
    if run_id is None:
        # Load latest gbm from history
        history = tracker._load_history()
        gbm_runs = [r for r in history if r['model_name'] == 'gbm']
        if not gbm_runs:
            print("No GBM runs found.")
            return
        run_id = gbm_runs[-1]['run_id']

    model_path = f"models/price_predictor_gbm_v{run_id}.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = joblib.load(model_path)
    df = load_and_prep_data()
    
    # Use the last 6 months as the evaluation set
    splitter = TemporalFoldSplitter(test_months=6, n_folds=1)
    _, test_df = splitter.split(df)[0]
    
    X_test = test_df
    y_test = test_df['price'].values
    
    y_pred = model.predict(X_test)
    
    analyzer = ResidualAnalyzer(results_dir="docs/validation")
    analyzer.analyze(y_test, y_pred, test_df, run_name=f"gbm_v{run_id}_val")
    print(f"Evaluation complete for GBM v{run_id}. Report saved to docs/validation/")

def evaluate_nn(run_id=None):
    """
    Evaluates a specific NN run.
    """
    if run_id is None:
        history = tracker._load_history()
        nn_runs = [r for r in history if r['model_name'] == 'price_net_macro']
        if not nn_runs:
            print("No NN runs found.")
            return
        run_id = nn_runs[-1]['run_id']

    from summit_housing.ml.nn import load_historical_nn
    import torch
    
    model, preprocessor, y_scaler, _, _, _, _ = load_historical_nn(run_id)
    if not model:
        print(f"NN Model v{run_id} not found.")
        return

    df = load_and_prep_data()
    splitter = TemporalFoldSplitter(test_months=6, n_folds=1)
    _, test_df = splitter.split(df)[0]

    X_test_proc = preprocessor.transform(test_df)
    X_test_t = torch.FloatTensor(X_test_proc)
    
    with torch.no_grad():
        y_scaled_pred = model(X_test_t).numpy()
    
    y_pred = np.expm1(y_scaler.inverse_transform(y_scaled_pred)).flatten()
    y_true = test_df['price'].values
    
    analyzer = ResidualAnalyzer(results_dir="docs/validation")
    analyzer.analyze(y_true, y_pred, test_df, run_name=f"nn_v{run_id}_val")
    print(f"Evaluation complete for NN v{run_id}. Report saved to docs/validation/")

if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "gbm"
    rid = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if model_type == "gbm":
        evaluate_gbm(rid)
    elif model_type == "nn":
        evaluate_nn(rid)
