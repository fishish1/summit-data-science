import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def test_sensitivity_gbm():
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

    # Load Model
    model_gbm = joblib.load("models/price_predictor_gbm.pkl")

    rates = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for r in rates:
        data = base_input.copy()
        data['mortgage_rate'] = r
        input_data = pd.DataFrame([data])
        pred = model_gbm.predict(input_data)[0]
        print(f"Rate: {r}% -> Price: ${pred:,.2f}")

if __name__ == "__main__":
    test_sensitivity_gbm()
