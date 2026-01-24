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

from summit_housing.queries import MarketAnalytics

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def analyze_features():
    """
    Performs scientific feature selection and importance analysis.
    Returns:
        - importance_df: DataFrame of feature importances
        - correlation_df: DataFrame of correlations
    """
    df = load_and_prep_data()
    
    # Define Features
    numeric_features = ['sfla', 'year_blt', 'beds', 'baths', 'garage_size', 'acres', 'mortgage_rate', 'cpi', 'sp500', 'summit_pop']
    
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
    
    return df

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

def train_nn():
    """
    Trains a PyTorch Neural Network.
    """
    print("\n--- Training Neural Network ---")
    df = load_and_prep_data()
    
    # Same Features (Pulse Indicators)
    numeric_features = ['sfla', 'year_blt', 'beds', 'baths', 'garage_size', 'acres', 'mortgage_rate', 'cpi', 'sp500', 'summit_pop', 'year']
    categorical_features = ['city', 'prop_type']
    
    X = df[numeric_features + categorical_features]
    y = df['price'].values
    
    # Preprocessing (Manual for NN)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # Dense for PyTorch
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    # Log Transform the Target (Price) for stability
    y_log = np.log1p(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_log, test_size=0.1, random_state=42)
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1) # Ensure Shape matches output
    
    # Model
    model = PriceNet(X_train.shape[1])
    criterion = nn.L1Loss() # MAE Loss (Robust to outliers)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 100
    batch_size = 64
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X.requires_grad = True # Enable gradient computation w.r.t input for monotonicity check
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Monotonicity Constraint: Mortgage Rate (Index 6) should be negative
            # Calculate gradients of output w.r.t inputs
            grads = torch.autograd.grad(outputs, batch_X, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
            rate_grads = grads[:, 6] # Index 6 is mortgage_rate
            
            # Penalize positive gradients (we want rate_grads <= 0)
            # ReLU(x) is x if x>0 else 0. So this penalizes only when gradient is positive.
            mono_penalty = torch.mean(torch.relu(rate_grads)) 
            
            # Total Loss
            total_loss = loss + 100.0 * mono_penalty # Strong penalty to enforce economic logic
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):,.0f}")
            
    # Save Metrics
    # Convert back from log space for MAE reporting
    with torch.no_grad():
        test_preds_actual = np.expm1(model(X_test_t).numpy())
        y_test_actual = np.expm1(y_test_t.numpy())
        mae_actual = np.mean(np.abs(test_preds_actual - y_test_actual))
        
    metrics = {"nn": {"mae": float(mae_actual)}}
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json", "r") as f:
            existing = json.load(f)
        existing.update(metrics)
        metrics = existing
        
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)
        
    # Save Model state dict AND the preprocessor (needed for inference)
    torch.save(model.state_dict(), "models/price_predictor_nn.pth")
    joblib.dump(preprocessor, "models/nn_preprocessor.pkl")
    
    print(f"✅ Neural Net Trained. MAE: ${mae_actual:,.0f}")
    
    return model, mae_actual

if __name__ == "__main__":
    train_gbm()
    if 'torch' in globals(): # Only run if torch imported successfully
         train_nn()
    else:
         # Fallback import check
         try:
             import torch
             train_nn()
         except ImportError:
             print("PyTorch not installed. Skipping NN training.")

def get_pdp_data(feature_name):
    """
    Calculates Partial Dependence for a specific feature.
    Shows the marginal effect of a feature on the predicted outcome.
    """
    df = load_and_prep_data()
    numeric_features = ['sfla', 'year_blt', 'beds', 'baths', 'garage_size', 'acres', 'mortgage_rate', 'cpi', 'sp500', 'summit_pop']
    
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
