#!/bin/bash
# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate Venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Install dependencies if needed (fast check)
# We need streamlit, joblib, torch, pandas, plotting, shap/statsmodels
echo "Checking dependencies..."
pip install streamlit joblib scikit-learn pandas plotly torch shap statsmodels matplotlib --quiet

# Run Dashboard
echo "Launching Summit Housing Dashboard..."
export PYTHONPATH=src
streamlit run src/summit_housing/dashboard/app.py
