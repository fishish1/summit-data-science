import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import mean_absolute_error, r2_score
from summit_housing.tracking import tracker
from .data import load_and_prep_data
from .validation import TemporalFoldSplitter, ResidualAnalyzer
from .uncertainty import IntervalPredictor
import yaml

with open("config/ml_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

def analyze_features(df=None):
    if df is None:
        df = load_and_prep_data()
    lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
    valid_lifts = [c for c in lift_cols if c in df.columns]
    extra_features = []
    if valid_lifts:
        df['dist_to_lift'] = df[valid_lifts].min(axis=1)
        df['dist_to_lift'] = df['dist_to_lift'].replace([np.inf, -np.inf], np.nan)
        extra_features.append('dist_to_lift')
    numeric_features = ['sfla', 'year_blt', 'beds', 'baths', 'garage_size', 'acres', 'mortgage_rate', 'cpi', 'sp500', 'summit_pop'] + extra_features
    corr_df = df[numeric_features + ['price']].corr()
    X = df[numeric_features].fillna(0) 
    y = df['price']
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X, y)
    r = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({'feature': numeric_features, 'importance': r.importances_mean, 'std': r.importances_std}).sort_values('importance', ascending=False)
    return importance_df, corr_df

def train_macro_model(df=None, params_override=None):
    if df is None:
        df = load_and_prep_data()
    
    numeric_features = CONFIG['features']['numeric']
    categorical_features = CONFIG['features']['categorical']
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = ['city', 'prop_type']
    target = 'price'
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_features + categorical_features + [target])
    df = df[df['price'] > 100000]
    if 'tx_date' in df.columns: df = df.sort_values('tx_date')
    X, y = df[numeric_features + categorical_features], df[target]
    
    # CALCULATE SAMPLE WEIGHTS (Luxury Calibration)
    # Give 2x weight to luxury properties (> $2M) to reduce larger absolute errors
    sample_weights = np.where(y > 2000000, 2.0, 1.0)
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    w_train = sample_weights[:split_idx]
    
    preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
    model_params = params_override if params_override else CONFIG['training']['gbm']
    
    # Preprocess X_train before fitting TransformedTargetRegressor
    X_train_proc = preprocessor.fit_transform(X_train)
    
    model = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(random_state=42, **model_params), 
        func=np.log1p, 
        inverse_func=np.expm1
    )
    
    print(f"Fitting GBM with Luxury Calibration (N_luxury={sum(y_train > 2000000)})...")
    # Note: Regressor expects preprocessed X
    model.fit(X_train_proc, y_train, sample_weight=w_train)
    
    # For inference, we need the pipeline to handle the raw data
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # TRAIN INTERVALS
    ip = IntervalPredictor()
    ip.train_intervals(X_train, y_train, numeric_features, categorical_features)
    
    ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    return pipeline, ip, X_test, y_test, numeric_features + categorical_features, numeric_features + list(ohe_names)

def get_shap_values(model_pipeline, X_sample):
    import shap
    if 'model' in model_pipeline.named_steps:
        wrapped_model = model_pipeline.named_steps['model']
    else:
        wrapped_model = model_pipeline.named_steps['regressor']
        
    actual_model = wrapped_model.regressor_ if hasattr(wrapped_model, 'regressor_') else wrapped_model
    preprocessor = model_pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X_sample)
    try:
        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer.shap_values(X_transformed)
    except Exception as e:
        explainer = shap.PermutationExplainer(actual_model.predict, X_transformed)
        shap_values = explainer(X_transformed).values
    return explainer, shap_values

def get_available_pdp_features():
    """Return a list of feature names that can be used for PDP calculations.
    This includes all numeric columns from the prepared dataframe plus any
    derived features such as `dist_to_lift` that are created on‑the‑fly.
    """
    df = load_and_prep_data()
    # Identify numeric columns (int or float) – exclude the target column.
    numeric_cols = [c for c, dtype in df.dtypes.items() if dtype.kind in "iuf" and c != "price"]
    # Add derived lift distance if possible.
    lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
    valid_lifts = [c for c in lift_cols if c in df.columns]
    if valid_lifts:
        numeric_cols.append('dist_to_lift')
    return numeric_cols


def get_pdp_data(feature_name):
    """Calculate PDP data for a given feature.
    The function now dynamically determines the set of numeric features
    (including any derived features) rather than relying on a hard‑coded list.
    If the requested feature is not available, an empty DataFrame is returned.
    
    Outliers are filtered using the IQR method to provide cleaner, more interpretable plots.
    """
    df = load_and_prep_data()
    # Ensure derived lift distance is present when needed.
    lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
    valid_lifts = [c for c in lift_cols if c in df.columns]
    extra_features = []
    if valid_lifts:
        df['dist_to_lift'] = df[valid_lifts].min(axis=1)
        df['dist_to_lift'] = df['dist_to_lift'].replace([np.inf, -np.inf], np.nan)
        extra_features.append('dist_to_lift')
    # Dynamically collect numeric columns.
    numeric_features = [c for c, dtype in df.dtypes.items() if dtype.kind in "iuf" and c != "price"]
    # Ensure any extra derived features are included.
    numeric_features = list(set(numeric_features + extra_features))
    if feature_name not in numeric_features:
        return pd.DataFrame()
    
    # Filter outliers using IQR method for the target feature
    # This removes extreme values that distort the PDP plot
    Q1 = df[feature_name].quantile(0.25)
    Q3 = df[feature_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Keep only non-outlier rows for this feature
    df_filtered = df[(df[feature_name] >= lower_bound) & (df[feature_name] <= upper_bound)].copy()
    
    # If filtering removed too much data, fall back to percentile filtering (1st to 99th)
    if len(df_filtered) < 100:
        lower_pct = df[feature_name].quantile(0.01)
        upper_pct = df[feature_name].quantile(0.99)
        df_filtered = df[(df[feature_name] >= lower_pct) & (df[feature_name] <= upper_pct)].copy()
    
    X, y = df_filtered[numeric_features].fillna(0), df_filtered['price']
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    try:
        pdp_results = partial_dependence(model, X, features=[feature_name], kind='average', grid_resolution=50)
        grid = pdp_results['grid_values'][0] if 'grid_values' in pdp_results else pdp_results['values'][0]
        preds = pdp_results['average'][0]
        return pd.DataFrame({'value': grid, 'average_prediction': preds})
    except Exception:
        return pd.DataFrame()

def backtest_gbm(n_folds=3):
    """
    Runs a full temporal backtest and returns aggregated metrics.
    """
    df = load_and_prep_data()
    splitter = TemporalFoldSplitter(test_months=6, n_folds=n_folds)
    folds = splitter.split(df)
    
    analyzer = ResidualAnalyzer(results_dir="docs/validation")
    fold_metrics = []
    
    for i, (train_df, test_df) in enumerate(folds):
        print(f"--- Training Fold {i} (Test Window: {test_df['tx_date'].min().date()} to {test_df['tx_date'].max().date()}) ---")
        
        numeric_features = ['sfla', 'beds', 'baths', 'year_blt', 'garage_size', 'acres', 'mortgage_rate', 'sp500', 'cpi', 'summit_pop']
        categorical_features = ['city', 'prop_type']
        target = 'price'
        
        train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_features + categorical_features + [target])
        test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_features + categorical_features + [target])
        
        X_train, y_train = train_df[numeric_features + categorical_features], train_df[target]
        X_test, y_test = test_df[numeric_features + categorical_features], test_df[target]
        
        preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
        model = TransformedTargetRegressor(regressor=HistGradientBoostingRegressor(random_state=42, max_iter=300, learning_rate=0.05), func=np.log1p, inverse_func=np.expm1)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        fold_metrics.append({'mae': mae, 'r2': r2})
        
        analyzer.analyze(y_test, y_pred, test_df, run_name=f"gbm_backtest_fold_{i}")
        
    avg_mae = np.mean([f['mae'] for f in fold_metrics])
    print(f"\nBacktest Complete. Avg MAE over {n_folds} folds: ${avg_mae:,.0f}")
    return fold_metrics

def train_gbm(params_override=None):
    df = load_and_prep_data()
    pipeline, ip, X_test, y_test, features, _ = train_macro_model(df, params_override=params_override)
    
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate SHAP summary
    import numpy as np
    import pandas as pd
    shap_list = None
    try:
        # We need a sample for SHAP (X_test is already split)
        # Use first 100 rows to speed it up
        X_sample = X_test.iloc[:100].copy()
        
        preprocessor = pipeline.named_steps['preprocessor']
        # The model step is named 'model' (see train_macro_model), but it's a TransformedTargetRegressor
        # We need the inner fitted regressor which is stored in .regressor_
        regressor = pipeline.named_steps['model'].regressor_
        
        X_transformed = preprocessor.transform(X_sample)
        
        # Use TreeExplainer for GBM
        import shap
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_transformed)
        
        # Handle potential list output from shap
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # Calculate mean absolute importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Map to features
        # Note: features list from train_macro_model should match transformed columns order ideally
        # But OneHotEncoder might expand features.
        # Let's try to get feature names from preprocessor if possible
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            # Fallback if get_feature_names_out not supported or complex
            feature_names = features 
            # If lengths don't match, we might have an issue, but let's try
            if len(feature_names) != len(mean_shap):
                # Try to generate names
                feature_names = [f"Feature {i}" for i in range(len(mean_shap))]

        shap_df = pd.DataFrame({'feature': feature_names, 'importance': mean_shap})
        shap_df = shap_df.sort_values('importance', ascending=False).head(15)
        shap_list = shap_df.to_dict('records')
        
    except Exception as e:
        print(f"⚠️ Warning: Could not generate SHAP summary: {e}")

    # Consolidate parameters for logging
    log_params = params_override if params_override else CONFIG['training']['gbm'].copy()
    log_params['features'] = features
    
    run_id = tracker.log_run(
        model_name="gbm", 
        metrics={"mae": float(mae), "r2": float(r2)}, 
        params=log_params,
        shap_summary=shap_list
    )
    
    joblib.dump(pipeline, f"models/price_predictor_gbm_v{run_id}.pkl")
    ip.save(run_id)
    print(f"GBM v{run_id} trained and saved with intervals.")
    return pipeline, mae

def load_historical_gbm(run_id):
    """Loads a specific historical GBM run by ID."""
    model_path = f"models/price_predictor_gbm_v{run_id}.pkl"
    if not os.path.exists(model_path):
        return None, None, None, None, None
        
    pipeline = joblib.load(model_path)
    
    # Try to load params from experiment history
    history_path = "models/experiment_history.json"
    params = {}
    mae = None
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            run_data = next((r for r in history if r['run_id'] == run_id), None)
            if run_data:
                params = run_data.get('parameters', {})
                mae = run_data.get('metrics', {}).get('mae')
                
    # Extract features from pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    features = []
    for _, _, cols in preprocessor.transformers_:
        features.extend(list(cols))
        
    return pipeline, mae, features, params
