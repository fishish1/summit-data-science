import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from summit_housing.tracking import tracker
from .data import load_and_prep_data
from .validation import TemporalFoldSplitter, ResidualAnalyzer
from .uncertainty import IntervalPredictor
import yaml

with open("config/ml_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

class PriceNet(nn.Module):
    def __init__(self, input_dim, params=None):
        super(PriceNet, self).__init__()
        nn_cfg = params if params else CONFIG['training']['nn']
        dims = nn_cfg['hidden_dims']
        layers = []
        prev_dim = input_dim
        for dim in dims:
            linear = nn.Linear(prev_dim, dim)
            # Kaiming initialization for ReLUs
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='leaky_relu')
            layers.extend([linear, nn.LeakyReLU(0.1)])
            if dim == dims[0]: layers.append(nn.Dropout(0.2))
            prev_dim = dim
        
        final_layer = nn.Linear(prev_dim, 1)
        nn.init.xavier_normal_(final_layer.weight)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

def train_macro_nn(df=None, params_override=None):
    if df is None: df = load_and_prep_data()
    numeric_features = CONFIG['features']['numeric']
    categorical_features = CONFIG['features']['categorical']
    # Filter to existing columns
    numeric_features = [f for f in numeric_features if f in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_features + categorical_features + ['price'])
    df = df[df['price'] > 100000]
    X, y = df[numeric_features + categorical_features], df['price'].values
    preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
    X_processed = preprocessor.fit_transform(X)
    y_log = np.log1p(y).reshape(-1, 1)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_log)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_scaled, test_size=0.2, random_state=42)
    
    # DYNAMIC CONSTRAINTS DETECTION
    nn_params = params_override if params_override else CONFIG['training']['nn']
    mono_cfg = nn_params.get('monotonicity', {})
    constraints = mono_cfg.get('constraints', {})
    
    pos_indices = []
    neg_indices = []
    
    for feat_name, direction in constraints.items():
        if feat_name in numeric_features:
            idx = numeric_features.index(feat_name)
            if direction == 1:
                pos_indices.append(idx)
            elif direction == -1:
                neg_indices.append(idx)
    
    X_train_t, y_train_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    
    print(f"Monotonicity: {len(pos_indices)} positive, {len(neg_indices)} negative constraints detected.")
    
    # LUXURY CALIBRATION: Sample Weights (aligned with training set)
    # We need to map y_scaled (which corresponds to X_processed) back to the training subset of y
    y_raw_train = np.expm1(y_scaler.inverse_transform(y_train)).flatten()
    weights_t = torch.FloatTensor(np.where(y_raw_train > 2000000, 2.0, 1.0))
    
    model = PriceNet(X_processed.shape[1], params=nn_params)
    optimizer = optim.Adam(model.parameters(), lr=nn_params['lr'])
    criterion = nn.SmoothL1Loss(reduction='none', beta=0.1) # Smooth gradients near zero
    
    dataloader = DataLoader(TensorDataset(X_train_t, y_train_t, weights_t), batch_size=nn_params['batch_size'], shuffle=True)
    model.train()
    print(f"Training NN with Luxury Calibration (Sample Weighting)...")
    for epoch in range(nn_params['epochs']):
        epoch_loss = 0
        for batch_X, batch_y, batch_w in dataloader:
            optimizer.zero_grad()
            
            # Monotonicity requires gradients w.r.t input
            mono_cfg = nn_params.get('monotonicity', {})
            is_mono = mono_cfg.get('enabled', False)
            if is_mono:
                batch_X.requires_grad = True

            outputs = model(batch_X)
            
            # Apply weights to the loss
            losses = criterion(outputs, batch_y)
            mae_loss = (losses * batch_w).mean()
            
            if is_mono:
                penalty_weight = mono_cfg.get('penalty_weight', 0.01)
                grads = torch.autograd.grad(outputs, batch_X, grad_outputs=torch.ones_like(outputs), create_graph=True, allow_unused=True)[0]
                
                # If a feature is unused (e.g. constant/masked), grads[0] might be None or zeros
                if grads is not None:
                    penalty = sum(torch.relu(-grads[:, idx]).mean() * penalty_weight for idx in pos_indices) + \
                              sum(torch.relu(grads[:, idx]).mean() * penalty_weight for idx in neg_indices)
                    (mae_loss + penalty).backward()
                else:
                    mae_loss.backward()
            else:
                mae_loss.backward()
                
            optimizer.step()
            epoch_loss += mae_loss.item()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{nn_params['epochs']} | Loss: {epoch_loss/len(dataloader):.4f}")
            
    # Train Intervals (using GBM-based predictor as a confidence layer)
    ip = IntervalPredictor()
    ip.train_intervals(X, y, numeric_features, categorical_features)
    
    model.eval()
    with torch.no_grad():
        test_preds = np.expm1(y_scaler.inverse_transform(model(X_test_t).numpy()))
        y_test_actual = np.expm1(y_scaler.inverse_transform(y_test))
        mae_actual = np.mean(np.abs(test_preds - y_test_actual))
        r2_actual = r2_score(y_test_actual, test_preds)
    
    # Calculate SHAP for NN
    shap_list = None
    try:
        import shap
        import pandas as pd
        # Use simple permutation importance if SHAP is too complex, but let's try GradientExplainer first
        # GradientExplainer requires model to be in training mode? No, eval is fine usually but gradients needed.
        # Actually KernelExplainer is safer for generic PyTorch models if dataset is small
        
        # Use KernelExplainer with a summarized background
        # It's slower but robust
        # Limit background and test samples strictly
        background = X_train[:25] # numpy array
        test_samples = X_test[:25] # numpy array
        
        # Wrapper for model to take numpy
        def model_wrapper(x_numpy):
            with torch.no_grad():
                tensor_x = torch.FloatTensor(x_numpy)
                return model(tensor_x).numpy()
        
        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(test_samples, nsamples=100)
        
        if isinstance(shap_values, list): shap_values = shap_values[0]
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Get feature names
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
             if hasattr(trans, 'get_feature_names_out'):
                 feature_names.extend(trans.get_feature_names_out())
             else:
                 feature_names.extend(cols)
                 
        # Ensure length match
        if len(feature_names) != len(mean_shap):
            feature_names = [f"Feature {i}" for i in range(len(mean_shap))]
            
        shap_df = pd.DataFrame({'feature': feature_names, 'importance': mean_shap})
        shap_df = shap_df.sort_values('importance', ascending=False).head(15)
        shap_list = shap_df.to_dict('records')
        
    except Exception as e:
        print(f"⚠️ Warning: Could not generate SHAP for NN: {e}")

    run_id = tracker.log_run(
        model_name="price_net_macro", 
        metrics={"mae": float(mae_actual), "r2": float(r2_actual)}, 
        params=nn_params,
        shap_summary=shap_list
    )
    torch.save(model.state_dict(), f"models/price_predictor_nn_v{run_id}.pth")
    joblib.dump(preprocessor, f"models/nn_preprocessor_v{run_id}.pkl")
    joblib.dump(y_scaler, f"models/nn_y_scaler_v{run_id}.pkl")
    ip.save(run_id)
    print(f"NN v{run_id} trained and saved with intervals.")
    return model, preprocessor, y_scaler, mae_actual, numeric_features + categorical_features, ip, nn_params

def load_historical_nn(run_id):
    """
    Loads a historical NN run by reconstructing its exact architecture from the log.
    """
    try:
        history = tracker._load_history()
        run_data = next((r for r in history if r['run_id'] == run_id), None)
        if not run_data:
            print(f"Error: Run {run_id} not found in history.")
            return None, None, None, None, None, None
            
        params = run_data['parameters']
        preprocessor = joblib.load(f"models/nn_preprocessor_v{run_id}.pkl")
        y_scaler = joblib.load(f"models/nn_y_scaler_v{run_id}.pkl")
        
        # Determine input dim from preprocessor
        input_dim = preprocessor.named_transformers_['num'].get_feature_names_out().shape[0] + \
                    preprocessor.named_transformers_['cat'].get_feature_names_out().shape[0]
                    
        model = PriceNet(input_dim, params=params)
        state_dict = torch.load(f"models/price_predictor_nn_v{run_id}.pth", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        num_cols = []
        cat_cols = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num': num_cols = list(cols)
            elif name == 'cat': cat_cols = list(cols)
            
        return model, preprocessor, y_scaler, run_data['metrics'].get('mae'), num_cols + cat_cols, None, params
    except Exception as e:
        print(f"Error loading historical model v{run_id}: {e}")
        return None, None, None, None, None, None, None
def backtest_nn(n_folds=3):
    """
    Runs a full temporal backtest for the Neural Network.
    """
    df = load_and_prep_data()
    splitter = TemporalFoldSplitter(test_months=6, n_folds=n_folds)
    folds = splitter.split(df)
    
    analyzer = ResidualAnalyzer(results_dir="docs/validation")
    fold_metrics = []
    
    for i, (train_df, test_df) in enumerate(folds):
        print(f"--- Training NN Fold {i} ---")
        
        numeric_features = CONFIG['features']['numeric']
        categorical_features = CONFIG['features']['categorical']
        numeric_features = [f for f in numeric_features if f in train_df.columns]
        
        train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_features + categorical_features + ['price'])
        test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_features + categorical_features + ['price'])
        
        X_train_raw, y_train_raw = train_df[numeric_features + categorical_features], train_df['price'].values
        X_test_raw, y_test_raw = test_df[numeric_features + categorical_features], test_df['price'].values
        
        preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)
        
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(np.log1p(y_train_raw).reshape(-1, 1))
        
        X_train_t, y_train_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        
        model = PriceNet(X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        dataloader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
        model.train()
        for epoch in range(50): # Fewer epochs for backtest speed
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred = np.expm1(y_scaler.inverse_transform(model(X_test_t).numpy())).flatten()
            
        mae = np.mean(np.abs(y_pred - y_test_raw))
        fold_metrics.append({'mae': mae})
        
        analyzer.analyze(y_test_raw, y_pred, test_df, run_name=f"nn_backtest_fold_{i}")
        
    avg_mae = np.mean([f['mae'] for f in fold_metrics])
    print(f"\nNN Backtest Complete. Avg MAE: ${avg_mae:,.0f}")
    return fold_metrics
