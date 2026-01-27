import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from summit_housing.dashboard.utils import get_trained_model_v2, get_trained_nn_model_v4

st.set_page_config(page_title="Price Predictor | Summit Housing", layout="wide")

def main():
    st.title("Interactive Price Predictor")
    st.markdown("---")

    # Use a Form to prevent reload on every slider
    with st.form("prediction_form"):
        # Model Selection
        model_choice = st.radio(
            "Model Architecture", 
            ["Gradient Boosting (Standard)", "Deep Learning (Experimental)"], 
            horizontal=True,
            help="Choose between a traditional tree-based model (GBM) or a Neural Network."
        )
        
        # 0. Load Model
        model_loaded = False
        pipeline = None
        nn_model = None
        nn_preprocessor = None
        nn_y_scaler = None
        
        # Note: spinner inside form might look weird, but functional
        try:
            if "Gradient" in model_choice:
                pipeline, _, _, _, _ = get_trained_model_v2()
            else:
                # Check for override version from Experiments page
                override_version = st.session_state.get('active_model_version', None)
                if override_version:
                    st.info(f"💾 Using Specific Experiment Version: Run #{override_version}")
                    nn_model, nn_preprocessor, nn_y_scaler, _ = get_trained_nn_model_v4(version=override_version)
                    if not nn_model: # Fallback if load fails
                         st.error(f"Failed to load Run #{override_version}. Falling back to default.")
                         nn_model, nn_preprocessor, nn_y_scaler, _ = get_trained_nn_model_v4()
                else:
                    nn_model, nn_preprocessor, nn_y_scaler, _ = get_trained_nn_model_v4()
            model_loaded = True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            model_loaded = False

        # 1. Input Form
        # Group 1: Property Features
        st.markdown("#### 🏠 Property Features")
        c1, c2, c3 = st.columns(3)
        with c1:
            sim_sqft = st.number_input("Square Footage", 400, 10000, 1500, step=50)
            sim_beds = st.slider("Bedrooms", 0, 8, 2)
            sim_type = st.selectbox("Property Type", ["Single Family", "Condo", "Townhouse", "Other"])
        with c2:
            sim_year = st.number_input("Year Built", 1900, 2025, 1995)
            sim_baths = st.slider("Bathrooms", 1.0, 8.0, 2.0, step=0.5)
        with c3:
            sim_loc = st.selectbox("Location", ["BRECKENRIDGE", "FRISCO", "SILVERTHORNE", "DILLON", "KEYSTONE", "COPPER MOUNTAIN"])
            sim_garage_choice = st.selectbox("Garage Spaces (Approx)", ["None", "1 Car", "2 Car", "3+ Car"])
            sim_garage = 0
            if sim_garage_choice == "1 Car": sim_garage = 250
            if sim_garage_choice == "2 Car": sim_garage = 500
            if sim_garage_choice == "3+ Car": sim_garage = 800
            
            sim_acres = st.number_input("Lot Size (Acres)", 0.0, 10.0, 0.0, step=0.1)

        # --- Feature Expansion: Quality & View ---
        st.markdown("##### ✨ Quality & Condition Steps")
        q1, q2, q3 = st.columns(3)
        with q1:
            sim_grade = st.selectbox("Construction Grade", ["Luxury (A)", "Custom/Fine (B)", "Average (C)", "Economy (D)"], index=2)
            # Map back to 1-6 scale used in training
            grade_map = {"Luxury (A)": 6, "Custom/Fine (B)": 5, "Average (C)": 4, "Economy (D)": 3}
            val_grade = grade_map[sim_grade]
            
        with q2:
            sim_cond = st.selectbox("Condition", ["Excellent", "Good", "Average", "Fair"], index=2)
            cond_map = {"Excellent": 6, "Good": 5, "Average": 4, "Fair": 3}
            val_cond = cond_map[sim_cond]
            
        with q3:
            sim_view = st.slider("Scenic View Score (0-5)", 0, 5, 0, help="0=None, 5=Panoramic Mountain Views")

        # Dynamic Distance Input
        # approximate min/max distances to a lift for each town

        # town: (min, max, default)
        dist_ranges = {
            "BRECKENRIDGE": (0.1, 8.0, 2.0), # Can be ski-in/out up to Blue River
            "FRISCO": (6.0, 15.0, 9.0),      # ~9 miles to Copper or Breck
            "SILVERTHORNE": (8.0, 20.0, 12.0),
            "DILLON": (5.0, 15.0, 6.0),      # Close to Keystone
            "KEYSTONE": (0.1, 5.0, 1.0),
            "COPPER MOUNTAIN": (0.0, 2.0, 0.5)
        }
        d_min, d_max, d_def = dist_ranges.get(sim_loc, (0.0, 20.0, 5.0))
        
        sim_dist_lift = st.slider("Distance to Ski Lift (Miles)", min_value=d_min, max_value=d_max, value=d_def, step=0.1, help="Distance to the nearest resort base.")

        st.divider()

        # Group 2: Macro Conditions
        st.markdown("#### 📉 Macro Economic Conditions")
        st.caption("How much is this house worth *in a different economy*?")
        
        m1, m2 = st.columns(2)
        with m1:
            # Defaults to current aprox values
            sim_rate = st.slider("Mortgage Rate (%)", 2.5, 9.0, 6.5, step=0.1)
        with m2:
            sim_sp500 = st.slider("S&P 500 Price ($)", 1000, 10000, 4800, step=100)
            
        submitted = st.form_submit_button("💰 Get Valuation")

    # 2. Prediction Logic (Outside Form if triggered)
    # Actually, easiest if inside logic runs on submit, but variables need to be captured.
    # Streamlit execution flow runs top to bottom. If submitted is True, runs block.
    
    if submitted and model_loaded:
        # A. Distance Lookup (Approximate driving miles from town center)
        # Dist order: [breck, keystone, copper, abasin, dillon]
        dist_map = {
            "BRECKENRIDGE":    {'dist_breck': 0,  'dist_keystone': 12, 'dist_copper': 16, 'dist_abasin': 18, 'dist_dillon': 10},
            "FRISCO":          {'dist_breck': 9,  'dist_keystone': 9,  'dist_copper': 6,  'dist_abasin': 13, 'dist_dillon': 4},
            "SILVERTHORNE":    {'dist_breck': 12, 'dist_keystone': 7,  'dist_copper': 11, 'dist_abasin': 10, 'dist_dillon': 1},
            "DILLON":          {'dist_breck': 11, 'dist_keystone': 5,  'dist_copper': 12, 'dist_abasin': 9,  'dist_dillon': 0},
            "KEYSTONE":        {'dist_breck': 14, 'dist_keystone': 0,  'dist_copper': 15, 'dist_abasin': 6,  'dist_dillon': 6},
            "COPPER MOUNTAIN": {'dist_breck': 17, 'dist_keystone': 18, 'dist_copper': 0,  'dist_abasin': 21, 'dist_dillon': 12}
        }
        
        # B. Hidden Defaults
        # Assuming current day CPI/Pop for the simulation context
        # In a real app, we might project these based on 'Simulation Year'
        default_cpi = 310.0 # Approx 2024 CPI
        default_pop = 31.0 # In thousands (Matches training data scale)
        
        # C. Construct Input DataFrame
        # Must match training feature names exactly
        # Numeric: ['sfla', 'beds', 'baths', 'year_blt', 'garage_size', 'acres', 'mortgage_rate', 'sp500', 'cpi', 'summit_pop', 'dist_to_lift', dist_breck...]
        
        dists = dist_map.get(sim_loc, {'dist_breck': 10, 'dist_keystone': 10, 'dist_copper': 10, 'dist_abasin': 10, 'dist_dillon': 10})
        
        # Override the calculated min distance with User Input
        # Logic: If user says they are 0.5 miles from lift, we rely on that feature primarily.
        # But we still need the other distance columns for the model shape.
        dist_to_lift = sim_dist_lift

        input_dict = {
            'sfla': [sim_sqft],
            'beds': [sim_beds],
            'baths': [sim_baths],
            'year_blt': [sim_year],
            'garage_size': [sim_garage],
            'acres': [sim_acres],
            'mortgage_rate': [sim_rate],
            'sp500': [sim_sp500],
            'cpi': [default_cpi],
            'summit_pop': [default_pop],
            'dist_to_lift': [dist_to_lift],
            'grade_numeric': [val_grade],
            'cond_numeric': [val_cond],
            'scenic_view': [sim_view],
            # Add explicit distance columns
            'dist_breck': [dists.get('dist_breck', 10)],
            'dist_keystone': [dists.get('dist_keystone', 10)],
            'dist_copper': [dists.get('dist_copper', 10)],
            'dist_abasin': [dists.get('dist_abasin', 10)],
            'dist_dillon': [dists.get('dist_dillon', 10)],
            
            # Categorical (OHE columns will be generated by pipeline or we pass raw if pipeline handles it)
            'city': [sim_loc],
            'prop_type': [sim_type]
        }
        
        # Add individual distances
        for k, v in dists.items():
            input_dict[k] = [v]
            
        input_df = pd.DataFrame(input_dict)
        
        # Predict Wrapper
        def get_prediction(model_type, df_in):
            if "Gradient" in model_type:
                return pipeline.predict(df_in)[0]
            else:
                # NN Prediction
                try:
                    X_proc = nn_preprocessor.transform(df_in)
                    
                    X_t = torch.FloatTensor(X_proc)
                    with torch.no_grad():
                        log_pred_scaled = nn_model(X_t).item()
                    
                    # Inverse Scale (Z-Score -> Log Price)
                    log_pred = nn_y_scaler.inverse_transform([[log_pred_scaled]])[0][0]
                    
                    # Sane Clipping
                    if log_pred > 20: log_pred = 20
                    if log_pred < 0: log_pred = 0
                    
                    return np.expm1(log_pred)
                except Exception as e:
                    st.error(f"NN Error: {e}")
                    return 0.0

        # Predict
        try:
            prediction = get_prediction(model_choice, input_df)
            
            st.divider()
            
            # Display Result
            c_res1, c_res2 = st.columns([1, 2])
            with c_res1:
                st.metric(label="Estimated Value", value=f"${prediction:,.0f}", delta=None)
            with c_res2:
                st.info(f"""
                **Simulation Values:**  
                {sim_sqft} sqft | {sim_loc} | {sim_type}  
                {sim_year} Built | {sim_rate}% Rates | ${sim_sp500} S&P 500
                """)
                
            # D. Sensitivity Mini-Check
            # What if rates dropped 1%?
            input_dict_low_rate = input_dict.copy()
            input_dict_low_rate['mortgage_rate'] = [sim_rate - 1.0]
            df_low = pd.DataFrame(input_dict_low_rate)
            pred_low = get_prediction(model_choice, df_low)
            diff = pred_low - prediction
            
            st.caption(f"📉 **Financial Leverage Effect:** If rates dropped by 1.0%, this home's value might increase by **${diff:,.0f}**.")
            
        except Exception as pred_e:
            st.warning(f"Prediction Error: {pred_e}")

if __name__ == "__main__":
    main()
