import streamlit as st
from summit_housing.ml import train_macro_model, train_macro_nn

# --- Model Caching (Shared) ---
@st.cache_resource
def get_trained_model_v2():
    """
    Trains (or loads) the ML model once and caches it in memory.
    """
    return train_macro_model()

@st.cache_resource
def get_trained_nn_model_v4(version=None):
    """
    Trains the Neural Network and caches it.
    If version is provided, loads that specific run ID from disk.
    """
    if version:
        from summit_housing.ml import load_historical_nn
        return load_historical_nn(version)

    return train_macro_nn()
