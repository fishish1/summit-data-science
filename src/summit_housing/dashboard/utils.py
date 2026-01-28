import streamlit as st
import pandas as pd
import numpy as np
from summit_housing.ml import train_macro_model, train_macro_nn, load_historical_nn, load_historical_gbm
from summit_housing.tracking import tracker
from summit_housing.ml.uncertainty import IntervalPredictor
from summit_housing.queries import MarketAnalytics
from summit_housing.geo import enrich_with_geo_features

@st.cache_resource
def get_trained_model_v2(version=None):
    """Returns the champion GBM model and its interval predictor."""
    # 1. Determine which run_id to load
    run_id = version
    if not run_id:
        champ = tracker.get_champion("gbm")
        run_id = champ['run_id'] if champ else None
        
    if run_id:
        import joblib
        import yaml
        
        # Use helper to load model + info
        pipeline, mae, features, params = load_historical_gbm(run_id)
        if not pipeline:
            return None, None, None, None, None, None
            
        ip = IntervalPredictor()
        try:
            ip.load(run_id)
        except Exception as e:
            # Only warn if it's the requested version (fallback logic handled by caller)
            if version: st.warning(f"Intervals not available for GBM Run {run_id}.")
            ip = None
        
        # Load sample data for SHAP
        from summit_housing.ml import load_and_prep_data
        df = load_and_prep_data()
        
        preprocessor = pipeline.named_steps['preprocessor']
        num_cols = []
        cat_cols = []
        
        # Recover columns from ColumnTransformer
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num': num_cols = list(cols)
            elif name == 'cat': cat_cols = list(cols)
        
        X_test = df[num_cols + cat_cols].tail(50)
        y_test = df['price'].tail(50)
        
        ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        shap_cols = num_cols + list(ohe_names)
        
        return pipeline, ip, X_test, y_test, num_cols + cat_cols, shap_cols
        
    return train_macro_model()

@st.cache_resource
def get_trained_nn_model_v4(version=None):
    """Returns the champion NN model and its interval predictor."""
    if version:
        # Load historical 7-tuple
        model, pre, scaler, mae, feat, _, params = load_historical_nn(version)
        ip = IntervalPredictor()
        try: ip.load(version)
        except: ip = None
        return model, pre, scaler, mae, feat, ip, params
        
    champ = tracker.get_champion("price_net_macro")
    if champ:
        model, pre, scaler, mae, feat, _, params = load_historical_nn(champ['run_id'])
        ip = IntervalPredictor()
        try: ip.load(champ['run_id'])
        except: ip = None
        return model, pre, scaler, mae, feat, ip, params
        
    # Fallback to training (returns 6-tuple)
    return train_macro_nn()

# --- Data Caching ---
@st.cache_data
def get_analytics_data(method_name: str, *args, **kwargs):
    """Helper to cache any MarketAnalytics method call."""
    analytics = MarketAnalytics()
    method = getattr(analytics, method_name)
    return method(*args, **kwargs)

@st.cache_data
def get_ml_feature_analysis():
    """Cached wrapper for analyze_features from ML package."""
    from summit_housing.ml import analyze_features
    return analyze_features()

@st.cache_data
def get_shap_analysis_data():
    """Calculates and caches SHAP values for the main GBM model."""
    from summit_housing.ml import get_shap_values
    pipeline, _, X_test, _, _, shap_cols = get_trained_model_v2()
    _, shap_values = get_shap_values(pipeline, X_test)
    preprocessor = pipeline.named_steps['preprocessor']
    X_test_transformed = preprocessor.transform(X_test)
    return shap_values, X_test_transformed, shap_cols

@st.cache_data
def get_pdp_data_cached(feature_name: str):
    """Calculates and caches PDP data for specific features."""
    from summit_housing.ml import get_pdp_data
    return get_pdp_data(feature_name)

@st.cache_data
def get_map_dataset():
    """Loads unique properties and calculates distances for visualization."""
    analytics = MarketAnalytics()
    df = analytics.get_dataset_sample(limit=5000) # Use sample for map performance
    
    if df.empty:
        return df
        
    if 'schno' in df.columns:
        df = df.drop_duplicates(subset=['schno'])
        
    try:
        df = enrich_with_geo_features(df)
        lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
        valid_lifts = [c for c in lift_cols if c in df.columns]
        
        if valid_lifts:
            df['dist_to_lift'] = df[valid_lifts].min(axis=1)
            df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'dist_to_lift'])
            MAX_VAL = 10.0
            df['dist_clamped'] = df['dist_to_lift'].clip(upper=MAX_VAL)
            df['norm'] = df['dist_clamped'] / MAX_VAL 
            df['r'] = (1 - df['norm']) * 255
            df['g'] = 50 
            df['b'] = df['norm'] * 255
        
        if 'address' not in df.columns:
            df['address'] = "Property View"

        return df[['LATITUDE', 'LONGITUDE', 'dist_to_lift', 'address', 'r', 'g', 'b']]
        
    except Exception as e:
        st.error(f"Geo enrichment failed: {e}")
        return pd.DataFrame()
