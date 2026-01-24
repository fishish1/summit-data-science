import joblib
import pandas as pd
import numpy as np
import torch
import os
import sys

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from summit_housing.ml import PriceNet

def test_sensitivity():
    # Base Case
    base_input = {
        'sfla': 2000,
        'year_blt': 1995,
        'beds': 3,
        'baths': 2.5,
        'garage_size': 400,
        'acres': 0.1,
        'mortgage_rate': 6.8,
        'cpi': 320,
        'sp500': 6000,
        'summit_pop': 31.0,
        'year': 2026, 
        'city': 'BRECKENRIDGE',
        'prop_type': 'Single Family'
    }

    # Load Preprocessor
    preprocessor = joblib.load("models/nn_preprocessor.pkl")
    # Load Model
    model_nn = PriceNet(preprocessor.transform(pd.DataFrame([base_input])).shape[1])
    model_nn.load_state_dict(torch.load("models/price_predictor_nn.pth"))
    model_nn.eval()

    rates = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for r in rates:
        data = base_input.copy()
        data['mortgage_rate'] = r
        input_data = pd.DataFrame([data])
        X_processed = preprocessor.transform(input_data)
        X_tensor = torch.FloatTensor(X_processed)
        with torch.no_grad():
            pred_log = model_nn(X_tensor).item()
            pred_actual = np.expm1(pred_log)
        print(f"Rate: {r}% -> Price: ${pred_actual:,.2f}")

if __name__ == "__main__":
    test_sensitivity()
