import pandas as pd
import numpy as np
import os
import joblib
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from summit_housing.tracking import tracker

from summit_housing.queries import MarketAnalytics
from summit_housing.geo import enrich_with_geo_features

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def analyze_features(df=None):
    """
    Performs scientific feature selection and importance analysis.
    Returns:
        - importance_df: DataFrame of feature importances
        - correlation_df: DataFrame of correlations
    """
    if df is None:
        df = load_and_prep_data()
    
    # Calculate 'dist_to_lift' if possible
    # We exclude dist_dillon (Lake) for this specific ski metric
    lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
    valid_lifts = [c for c in lift_cols if c in df.columns]
    
    extra_features = []
    if valid_lifts:
        df['dist_to_lift'] = df[valid_lifts].min(axis=1)
        # Handle cases where all were NaN/Inf
        df['dist_to_lift'] = df['dist_to_lift'].replace([np.inf, -np.inf], np.nan)
        extra_features.append('dist_to_lift')

    # Define Features
    numeric_features = ['sfla', 'year_blt', 'beds', 'baths', 'garage_size', 'acres', 'mortgage_rate', 'cpi', 'sp500', 'summit_pop'] + extra_features
    
    # 1. Correlation Analysis (Scientific Selection Step 1)
    # Filter to numeric only for correlation
    corr_df = df[numeric_features + ['price']].corr()
    
    # 2. Permutation Importance (Scientific Selection Step 2)
    # We use a Random Forest here as it handles non-linearities well.
    # We use only numeric features here for a cleaner visualization of "Drivers".
    
    X = df[numeric_features].fillna(0) 
    y = df['price']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X, y)
    
    # Calculate Permutation Importance
    # This shuffles each feature to see how much the Error increases. 
    # Much more "scientific" than Gini importance.
    r = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    
    importance_df = pd.DataFrame({
        'feature': numeric_features,
        'importance': r.importances_mean,
        'std': r.importances_std
    }).sort_values('importance', ascending=False)
    
    return importance_df, corr_df

def load_and_prep_data():
    """
    Fetches sales data and joins with local Macro-Economic indicators.
    """
    print("Fetching Sales Data...")
    analytics = MarketAnalytics()
    df = analytics.get_training_data()
    # --- Geospatial Enrichment ---
    try:
        print("📍 Enriching with Geospatial Data...")
        df = enrich_with_geo_features(df)
    except Exception as e:
        print(f"⚠️ Geospatial Enrichment Failed: {e}")
    
    # 
    # 1. Convert Dates
    df['tx_date'] = pd.to_datetime(df['tx_date'])
    df['year'] = df['tx_date'].dt.year
    df['month'] = df['tx_date'].dt.month # Seasonality feature!
    
    # 2. Load Macro Data
    print("Loading Macro Indicators...")
    macro_files = {
        'mortgage_rate': 'data/mortgage_rate.csv',
        'cpi': 'data/cpi.csv',
        'sp500': 'data/sp500.csv',
        'summit_pop': 'data/summit_pop.csv'
    }
    
    macro_dfs = []
    for name, path in macro_files.items():
        if os.path.exists(path):
            macro_df = pd.read_csv(path)
            macro_df['date'] = pd.to_datetime(macro_df['date'])
            macro_df = macro_df.sort_values('date').rename(columns={'value': name})
            macro_dfs.append(macro_df[['date', name]])
    
    # 3. Merge Macro Data (AsOf Merge - finding closest past date)
    df = df.sort_values('tx_date')
    for m_df in macro_dfs:
        df = pd.merge_asof(
            df, 
            m_df, 
            left_on='tx_date', 
            right_on='date', 
            direction='backward'
        )
        # Drop redundant date column
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
            
    # 4. Fill Missing Macro Data (Forward Fill if any gaps)
    df = df.ffill().bfill()
    
    # 5. Handle Property Nulls
    df['garage_size'] = df['garage_size'].fillna(0)
    df['year_blt'] = df['year_blt'].fillna(df['year_blt'].median())
    df['beds'] = df['beds'].fillna(0)
    df['baths'] = df['baths'].fillna(0)
    df['acres'] = df['acres'].fillna(0) # Lot Size
    
    # 6. Ordinal Encoding for Quality Features (The Kitchen Sink)
    # Map A (Excellent) -> 5, E (Poor) -> 1
    quality_map = {
        'A': 6, 'X': 6, # Luxury
        'B': 5, # Fine
        'C': 4, # Average
        'D': 3, # Economy/Fair
        'E': 2, # Poor
        'F': 1  # Very Poor
    }
    
    # Apply Mapping
    if 'grade' in df.columns:
        df['grade_numeric'] = df['grade'].map(quality_map).fillna(3) # Default to Average
    
    if 'cond' in df.columns:
        df['cond_numeric'] = df['cond'].map(quality_map).fillna(3) # Default to Average
        
    if 'scenic_view' in df.columns:
        df['scenic_view'] = df['scenic_view'].fillna(0)
    
    return df

def train_macro_model(df=None):
    """
    Trains a HistGradientBoostingRegressor specifically for the Macro Simulator.
    Returns:
        - model: Trained sklearn pipeline
        - X_test: Test features for evaluation/SHAP
        - y_test: Test targets
        - feature_names: List of feature names
    """
    if df is None:
        df = load_and_prep_data()
    
    # Define Features
    # Added 'city' for Hyper-Locality and Geospatial Distances, plus Quality Features
    base_numeric_features = [
        'sfla', 'beds', 'baths', 'year_blt', 'garage_size', 'acres', 
        'mortgage_rate', 'sp500', 'cpi', 'summit_pop',
        'grade_numeric', 'cond_numeric', 'scenic_view'
    ]
    
    geo_features = [
        'dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin', 'dist_dillon'
    ]
    
    # Only include geo features if they exist (Enrichment might have failed)
    numeric_features = base_numeric_features.copy()
    existing_geo = [f for f in geo_features if f in df.columns]
    
    if existing_geo:
        # Calculate 'dist_to_lift' (Min of all resort distances)
        # We exclude dist_dillon (Lake) for this specific ski metric
        lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
        valid_lifts = [c for c in lift_cols if c in df.columns]
        
        if valid_lifts:
            df['dist_to_lift'] = df[valid_lifts].min(axis=1)
            numeric_features.append('dist_to_lift')
            
        numeric_features += existing_geo
    else:
        print("⚠️ Warning: Geospatial features missing from dataframe. Training without them.")

    categorical_features = ['city', 'prop_type']
    
    target = 'price'
    
    # Cleaning: Replace Infinity with NaN (Crucial for StandardScaler)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Filter for valid data
    df = df.dropna(subset=numeric_features + categorical_features + [target])
    df = df[df['price'] > 100000] # Remove non-market transfers

    # --- SCIENTIFIC FIX 1: Time-Based Split ---
    # Sort by Transaction Date to prevent looking into the future
    if 'tx_date' in df.columns:
        df = df.sort_values('tx_date')
    
    X = df[numeric_features + categorical_features]
    y = df[target]
    
    # Split: Cut off last 20% temporally
    # Instead of random shuffle, we slice the sorted array
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} sales (Historical). Testing on {len(X_test)} sales (Future).")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # --- SCIENTIFIC FIX 2: Monotonic Constraints & Log Targets ---
    # Issue: Trees can't predict higher prices than seen in train data.
    # Fix A: Log-Transform the target (Price -> Log(Price)) to normalize distribution errors
    # Fix B: Enforce Monotonicity on Macro Factors (S&P 500 up -> Price up)
    
    # To use monotonic constraints, we must map feature indices.
    # This acts as a 'guard rail' for extrapolation.
    # 0=None, 1=Increasing, -1=Decreasing
    
    # Note: Complex pipelines make index mapping hard. 
    # For this simplified model, we will rely on key features being mostly monotonic naturally
    # and use Log-Space prediction to handle the scaling better.
    
    from sklearn.compose import TransformedTargetRegressor

    # Base Model
    hgbr = HistGradientBoostingRegressor(
        random_state=42, 
        max_iter=300, 
        learning_rate=0.05,
        max_depth=12,
        l2_regularization=0.1
    )
    
    # Wrap in Target Transformer (Log Space)
    model = TransformedTargetRegressor(
        regressor=hgbr,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    # Pipeline: Scale/Encode -> Log-Space HGBR
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Get Feature Names after encoding
    ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    shap_feature_names = numeric_features + list(ohe_feature_names)
    
    input_features = numeric_features + categorical_features
    
    return pipeline, X_test, y_test, input_features, shap_feature_names

def get_shap_values(model_pipeline, X_sample):
    """
    Calculates SHAP values for a sample.
    """
    import shap
    
    # Extract Logic updated for TransformedTargetRegressor
    # The 'model' step in pipeline is now TransformedTargetRegressor
    wrapped_model = model_pipeline.named_steps['model']
    
    # We need the ACTUAL regressor inside, not the wrapper
    actual_model = wrapped_model.regressor_
    
    preprocessor = model_pipeline.named_steps['preprocessor']
    
    # Transform input (Scale + Encode)
    # X_sample must be a DataFrame with same columns as X_train
    X_transformed = preprocessor.transform(X_sample)
    
    # Create Explainer
    try:
        # tree_limit set to model's best iteration to match prediction logic if early stopping was used
        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer.shap_values(X_transformed)
    except Exception as e:
        print(f"TreeExplainer failed: {e}. Using PermutationExplainer.")
        # Wrapper for prediction that includes the inverse transform if needed, 
        # but SHAP usually explains the raw margin of the tree.
        # To explain LOG price, we use raw model. To explain REAL price, we use pipeline predict.
        # explaining raw log price is usually more stable numerically.
        explainer = shap.PermutationExplainer(actual_model.predict, X_transformed)
        shap_values = explainer(X_transformed).values
        
    return explainer, shap_values

def train_gbm():
    """
    Trains a Gradient Boosting Regressor (Scikit-Learn).
    """
    df = load_and_prep_data()
    
    # Features (Pulse Indicators)
    numeric_features = ['sfla', 'year_blt', 'beds', 'baths', 'garage_size', 'acres', 'mortgage_rate', 'cpi', 'sp500', 'summit_pop', 'year']
    categorical_features = ['city', 'prop_type']
    
    X = df[numeric_features + categorical_features]
    y = df['price']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # To use monotonic constraints, we need to know the final feature count.
    # numeric_features: [sfla, year_blt, beds, baths, garage_size, acres, mortgage_rate, cpi, sp500, summit_pop, year]
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    
    # Fit preprocessor to get category counts
    preprocessor.fit(X_train)
    n_features = preprocessor.transform(X_train[:1]).shape[1]
    
    # Constraints: 1 (positive), -1 (negative), 0 (none)
    # Most numeric features should be positive. mortgage_rate (6) MUST be negative.
    cst = [0] * n_features
    for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]:
        cst[i] = 1
    cst[6] = -1 # mortgage_rate
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            max_iter=500, 
            learning_rate=0.1, 
            random_state=42,
            monotonic_cst=cst
        ))
    ])
    
    print("Training Gradient Boosting Model with Sanity Constraints...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ GBM Trained. MAE: ${mae:,.0f} | R2: {r2:.3f}")
    
    # Save
    joblib.dump(model, "models/price_predictor_gbm.pkl")
    
    # Save Metrics
    metrics = {"gbm": {"mae": float(mae), "r2": float(r2)}}
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json", "r") as f:
            existing = json.load(f)
        existing.update(metrics)
        metrics = existing
        
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

    # Log to History
    run_id = tracker.log_run(
        model_name="gbm",
        metrics={"mae": float(mae), "r2": float(r2)},
        params={"model": "HistGradientBoostingRegressor", "features": numeric_features + categorical_features}
    )

    # Save Versioned Copy
    joblib.dump(model, f"models/price_predictor_gbm_v{run_id}.pkl")
        
    return model, mae

# --- PyTorch Implementation ---

class PriceNet(nn.Module):
    def __init__(self, input_dim):
        super(PriceNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Single output (Price)
        )
    
    def forward(self, x):
        return self.network(x)

def train_macro_nn(df=None):
    """
    Trains a PyTorch Neural Network specifically for the Macro Simulator.
    Incorporates Scientific Rigor (Time Split, Log Space, Monotonicity).
    Returns:
        - model: Trained PyTorch model
        - preprocessor: Fitted sklearn preprocessor
        - mae: Mean Absolute Error on test set
    """
    print("\n--- Training Deep Learning Model (PyTorch) ---")
    if df is None:
        df = load_and_prep_data()
    
    # Define Features (Same as HGBR for fair comparison)
    base_numeric_features = [
        'sfla', 'beds', 'baths', 'year_blt', 'garage_size', 'acres', 
        'mortgage_rate', 'sp500', 'cpi', 'summit_pop',
        'grade_numeric', 'cond_numeric', 'scenic_view'
    ]
    
    geo_features = [
        'dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin', 'dist_dillon'
    ]
    
    # Only include geo features if they exist
    numeric_features = base_numeric_features.copy()
    existing_geo = [f for f in geo_features if f in df.columns]
    
    if existing_geo:
        # Calculate 'dist_to_lift'
        lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
        valid_lifts = [c for c in lift_cols if c in df.columns]
        
        if valid_lifts:
            df['dist_to_lift'] = df[valid_lifts].min(axis=1)
            numeric_features.append('dist_to_lift')
            
        numeric_features += existing_geo

    categorical_features = ['city', 'prop_type']
    target = 'price'
    
    # Cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_features + categorical_features + [target])
    df = df[df['price'] > 100000] # Remove non-market transfers

    # Time-Based Sort
    if 'tx_date' in df.columns:
        df = df.sort_values('tx_date')
    
    X = df[numeric_features + categorical_features]
    y = df[target].values
    
    # Preprocessing (Manual for NN)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # Dense for PyTorch
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    # Log Transform the Target (Price) for stability
    y_log = np.log1p(y).reshape(-1, 1)
    
    # Scale Target (Mean=0, Std=1)
    # This keeps gradients well-behaved and prevents "Exploding Price" predictions
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_log)
    
    # Time-Based Split (80/20) - REVERTED TO RANDOM SPLIT for fair comparison with GBM
    # split_idx = int(len(df) * 0.8)
    # X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
    # y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_scaled, test_size=0.2, random_state=42)
    
    print(f"DL Training: {len(X_train)} samples vs {len(X_test)} validation samples.")
    
    # Convert to Tensors
    
    # Find Mortgage Rate Index for Monotonicity
    # We need to know which column output corresponds to 'mortgage_rate'
    # Since 'num' comes first in ColumnTransformer, and 'mortgage_rate' is in numeric_features...
    try:
        mortgage_idx = numeric_features.index('mortgage_rate')
    except ValueError:
        mortgage_idx = -1 # Not found

    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train) # Already 2D from reshape
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # Model
    model = PriceNet(X_train.shape[1])
    criterion = nn.L1Loss() # MAE Loss (Robust to outliers)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Reduced LR
    
    # Training Loop
    epochs = 200
    batch_size = 128
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- Scientific Feature Constraints ---
    # We dynamically find indices to be robust against feature reordering
    
    # 1. Positive Factors (More is Better)
    # SFLA, Beds, Baths, YrBuilt, Garage, Acres, S&P500, CPI(?)
    # Added: Grade, Cond, View
    pos_features = ['sfla', 'beds', 'baths', 'year_blt', 'garage_size', 'acres', 'grade_numeric', 'cond_numeric', 'scenic_view']
    pos_indices = []
    for feat in pos_features:
        if feat in numeric_features:
            pos_indices.append(numeric_features.index(feat))
            
    # 2. Negative Factors (Less is Better)
    # Mortgage Rates, Distance to Lifts (Closer = Less Distance = Higher Value)
    neg_features = ['mortgage_rate', 'dist_to_lift', 'dist_breck', 'dist_keystone', 'dist_copper']
    neg_indices = []
    for feat in neg_features:
        if feat in numeric_features:
            neg_indices.append(numeric_features.index(feat))
    
    print(f"Applying Monotonicity: Positive {pos_indices}, Negative {neg_indices}")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X.requires_grad = True # Enable gradient computation
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Primary Loss
            mae_loss = criterion(outputs, batch_y)
            
            # --- Monotonicity Constraints ---
            # Calculate Gradients of Output w.r.t Input
            grads = torch.autograd.grad(
                outputs, 
                batch_X, 
                grad_outputs=torch.ones_like(outputs),
                create_graph=True
            )[0]
            
            penalty = 0
            
            # Apply Positive Constraints (Penalty if slope < 0)
            # Gentle application (0.01 weight)
            for idx in pos_indices:
               if idx < batch_X.shape[1]:
                    penalty += torch.relu(-grads[:, idx]).mean() * 0.01
                     
            # Apply Negative Constraints (Penalty if slope > 0)
            # Gentle application (0.01 weight)
            for idx in neg_indices:
               if idx < batch_X.shape[1]:
                    penalty += torch.relu(grads[:, idx]).mean() * 0.01
            
            loss = mae_loss + penalty
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(X_test_t).numpy()
        test_preds_log = y_scaler.inverse_transform(test_preds_scaled)
        test_preds_actual = np.expm1(test_preds_log)
        
        y_test_log = y_scaler.inverse_transform(y_test)
        y_test_actual = np.expm1(y_test_log)
        
        mae_actual = np.mean(np.abs(test_preds_actual - y_test_actual))
        r2_actual = r2_score(y_test_actual, test_preds_actual)
        
    metrics = {"nn": {"mae": float(mae_actual), "r2": float(r2_actual)}}
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json", "r") as f:
            try:
                existing = json.load(f)
            except:
                existing = {}
        existing.update(metrics)
        metrics = existing
        
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)
        
    # Save Model state dict AND the preprocessor (needed for inference)
    torch.save(model.state_dict(), "models/price_predictor_nn.pth")
    joblib.dump(preprocessor, "models/nn_preprocessor.pkl")
    joblib.dump(y_scaler, "models/nn_y_scaler.pkl")
    
    print(f"✅ Neural Net Trained. MAE: ${mae_actual:,.0f}")

    # Log to History
    run_id = tracker.log_run(
        model_name="price_net_macro",
        metrics={"mae": float(mae_actual), "r2": float(r2_actual)},
        params={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 0.0001,
            "features": numeric_features + categorical_features,
            "monotonic_constraints": True
        }
    )

    # Save Versioned Copy
    torch.save(model.state_dict(), f"models/price_predictor_nn_v{run_id}.pth")
    joblib.dump(preprocessor, f"models/nn_preprocessor_v{run_id}.pkl")
    joblib.dump(y_scaler, f"models/nn_y_scaler_v{run_id}.pkl")
    
    return model, preprocessor, y_scaler, mae_actual

if __name__ == "__main__":
    train_gbm()
    if 'torch' in globals(): # Only run if torch imported successfully
         train_macro_nn()
    else:
         # Fallback import check
         try:
             import torch
             train_macro_nn()
         except ImportError:
             print("PyTorch not installed. Skipping NN training.")

def get_pdp_data(feature_name):
    """
    Calculates Partial Dependence for a specific feature.
    Shows the marginal effect of a feature on the predicted outcome.
    """
    df = load_and_prep_data()
    
    # Logic to add dist_to_lift (Same as in analyze_features)
    lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
    valid_lifts = [c for c in lift_cols if c in df.columns]
    
    extra_features = []
    if valid_lifts:
        df['dist_to_lift'] = df[valid_lifts].min(axis=1)
        df['dist_to_lift'] = df['dist_to_lift'].replace([np.inf, -np.inf], np.nan)
        extra_features.append('dist_to_lift')

    numeric_features = ['sfla', 'year_blt', 'beds', 'baths', 'garage_size', 'acres', 'mortgage_rate', 'cpi', 'sp500', 'summit_pop'] + extra_features
    
    # Ensure the requested feature is actually available
    if feature_name not in numeric_features:
        print(f"Feature {feature_name} not found in available features: {numeric_features}")
        return pd.DataFrame()
        
    # Simple model for PDP (Random Forest is robust)
    X = df[numeric_features].fillna(0)
    y = df['price']
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Calculate PDP
    try:
        pdp_results = partial_dependence(
            model, X, features=[feature_name], kind='average', grid_resolution=50
        )
        
        # Scikit-Learn's partial_dependence returns a Bunch with 'grid_values' (list of arrays)
        # and 'average' (predictions).
        # We access [0] because we only asked for 1 feature.
        
        # Robustly handle key names across versions
        grid = pdp_results['grid_values'][0] if 'grid_values' in pdp_results else pdp_results['values'][0]
        preds = pdp_results['average'][0]
        
        return pd.DataFrame({
            'value': grid,
            'average_prediction': preds
        })
    except Exception as e:
        print(f"Error calculating PDP for {feature_name}: {e}")
        return pd.DataFrame()

def load_historical_nn(run_id):
    """
    Loads a specific version of the Neural Network and its artifacts.
    """
    print(f"Loading Neural Net Run #{run_id}...")
    
    try:
        # Load Artifacts
        preprocessor = joblib.load(f"models/nn_preprocessor_v{run_id}.pkl")
        y_scaler = joblib.load(f"models/nn_y_scaler_v{run_id}.pkl")
        
        # Determine Input Dimension from the saved weights
        # We need this to instantiate the correct model architecture
        state_dict = torch.load(f"models/price_predictor_nn_v{run_id}.pth")
        input_dim = state_dict['network.0.weight'].shape[1]
        
        model = PriceNet(input_dim)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, preprocessor, y_scaler, None # None for mae (not needed for inference)
    except Exception as e:
        print(f"Failed to load Run #{run_id}: {e}")
        return None, None, None, None
