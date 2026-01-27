import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from datetime import datetime

st.set_page_config(page_title="Experiment Tracker", page_icon="🧬", layout="wide")

st.title("🧬 ML Experiment Registry")
st.markdown("""
This dashboard visualizes the training history from `models/experiment_history.json`.  
It serves as a lightweight **MLOps** store to track model performance over time.
""")

HISTORY_FILE = "models/experiment_history.json"

def load_experiments():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    
    with open(HISTORY_FILE, 'r') as f:
        data = json.load(f)
        
    # Flatten the JSON
    rows = []
    for run in data:
        row = {
            "Run ID": run['run_id'],
            "Timestamp": pd.to_datetime(run['timestamp']),
            "Model": run['model_name'],
            "MAE": run['metrics'].get('mae'),
            "R2": run['metrics'].get('r2', None),
            # Flatten parameters carefully
            "Epochs": run['parameters'].get('epochs'),
            "Learning Rate": run['parameters'].get('lr'),
            "Batch Size": run['parameters'].get('batch_size'),
            "Constraints": run['parameters'].get('monotonic_constraints', False)
        }
        rows.append(row)
        
    return pd.DataFrame(rows)

# Load Data
df = load_experiments()

if df.empty:
    st.warning("No experiment history found yet. Train a model first!")
else:
    # Ensure MAE is numeric and drop invalid rows
    df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')
    df = df.dropna(subset=['MAE'])
    
    if df.empty:
        st.warning("Experiments found, but no valid MAE metrics recorded.")
    else:
        # --- Summary Metrics ---
        try:
            best_run_idx = df['MAE'].idxmin()
            best_run = df.loc[best_run_idx]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Runs", len(df))
            c2.metric("Best MAE (Lowest)", f"${best_run['MAE']:,.0f}")
            c3.metric("Best Model v", f"Run #{best_run['Run ID']} ({best_run['Model']})")
        except Exception as e:
            st.error(f"Error calculating summary metrics: {e}")
            
        st.divider()
        
        # --- Leaderboard ---
        st.subheader("🏆 Leaderboard")
        
        try:
            # Highlight best run
            def highlight_best(s):
                is_min = s == s.min()
                return ['background-color: #dcfce7' if v else '' for v in is_min]

            # Use a copy to avoid SettingWithCopy warnings on style
            display_df = df.sort_values("MAE").copy()
            
            st.dataframe(
                display_df.style.format({
                    "MAE": "${:,.0f}",
                    "R2": "{:.3f}",
                    "Timestamp": "{:%Y-%m-%d %H:%M}"
                }, na_rep="-").apply(highlight_best, subset=['MAE']),
                use_container_width=True,
                hide_index=True
            )
            
            # --- Model Loading ---
            st.subheader("💾 Load Specific Version")
            col_load_1, col_load_2 = st.columns([3, 1])
            with col_load_1:
                 # Only show runs that match "price_net" or similar because we only implemented loading for NN
                 # Filter based on model name column
                 if 'Model' in df.columns:
                     nn_runs = df[df['Model'].astype(str).str.contains('net|nn', case=False, regex=True)]
                 else:
                     nn_runs = pd.DataFrame(columns=['Run ID'])
                     
                 if not nn_runs.empty:
                     run_options = nn_runs['Run ID'].sort_values(ascending=False).tolist()
                     
                     selected_run = st.selectbox(
                         "Select Run ID to Load (Neural Networks Only)", 
                         options=run_options,
                         format_func=lambda x: f"Run #{x} (MAE: ${df[df['Run ID']==x]['MAE'].iloc[0]:,.0f})"
                     )
                 else:
                     st.info("No Neural Network runs available to load (GBM loading not supported yet).")
                     selected_run = None
                 
            with col_load_2:
                st.write("") # Spacer
                st.write("") 
                if selected_run and st.button("Load this Model", type="primary"):
                    # Set session state which Predictor.py will read
                    st.session_state['active_model_version'] = int(selected_run)
                    st.success(f"Run #{selected_run} Activated! Go to 'Predictor' page to use it.")

        except Exception as e:
            st.error(f"Error rendering leaderboard: {e}")
            st.dataframe(df)

        # --- Visualizations ---
        c_chart1, c_chart2 = st.columns(2)
        
        with c_chart1:
            st.subheader("📉 Error Reduction Over Time")
            fig_time = px.line(
                df, 
                x='Timestamp', 
                y='MAE', 
                color='Model', 
                markers=True,
                title="Model Convergence History",
                hover_data=['Run ID', 'Epochs']
            )
            st.plotly_chart(fig_time, use_container_width=True)
            
        with c_chart2:
            st.subheader("🐌 Learning Rate Impact")
            # Filter for NN only as GBM doesn't have LR
            # Check if Model column exists and is string
            if 'Model' in df.columns:
                 df_nn = df[df['Model'].astype(str).str.contains('net|nn', case=False, regex=True)]
            else:
                 df_nn = pd.DataFrame()
            
            if not df_nn.empty and 'Learning Rate' in df_nn.columns:
                fig_lr = px.scatter(
                    df_nn,
                    x='Learning Rate',
                    y='MAE',
                    size='Epochs',
                    color='Batch Size',
                    title="Hyperparameter Impact (Neural Net)",
                    log_x=True
                )
                st.plotly_chart(fig_lr, use_container_width=True)
            else:
                st.caption("No Neural Network runs to analyze hyperparameters for.")
