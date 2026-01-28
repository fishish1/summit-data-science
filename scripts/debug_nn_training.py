import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from summit_housing.ml.data import load_and_prep_data
import matplotlib.pyplot as plt

def debug_nn():
    with open("config/ml_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    df = load_and_prep_data()
    numeric_features = config['features']['numeric']
    categorical_features = config['features']['categorical']
    
    # Preprocessing identical to nn.py
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
    X_train_t, y_train_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_test_t, y_test_t = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    
    nn_params = config['training']['nn']
    
    from summit_housing.ml.nn import PriceNet
    model = PriceNet(X_processed.shape[1], params=nn_params)
    optimizer = optim.Adam(model.parameters(), lr=nn_params['lr'])
    criterion = nn.L1Loss()
    
    dataloader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=nn_params['batch_size'], shuffle=True)
    
    print(f"Starting Debug Training: {nn_params['epochs']} epochs")
    history = []
    for epoch in range(1, nn_params['epochs'] + 1):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds_scaled = model(X_test_t).numpy()
                r2 = r2_score(y_test, preds_scaled)
                preds = np.expm1(y_scaler.inverse_transform(preds_scaled)).flatten()
                y_true = np.expm1(y_scaler.inverse_transform(y_test)).flatten()
                mae = mean_absolute_error(y_true, preds)
                print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | R2: {r2:.4f} | MAE: ${mae:,.0f} | StdPred: {preds.std():.0f}")
                history.append(preds.std())

    print("\nTraining Final Results:")
    print(f"Prediction Std Dev: {preds.std():.2f}")
    if preds.std() < 5000:
        print("WARNING: Model predictions are indeed extremely flat!")
    
    # Histogram of predictions vs truth
    plt.figure(figsize=(10, 5))
    plt.hist(preds, bins=50, alpha=0.5, label='Predictions')
    plt.hist(y_true, bins=50, alpha=0.5, label='Actual')
    plt.legend()
    plt.title("Prediction Distribution")
    plt.savefig("prediction_debug.png")
    print("Saved distribution plot to prediction_debug.png")

if __name__ == '__main__':
    debug_nn()
