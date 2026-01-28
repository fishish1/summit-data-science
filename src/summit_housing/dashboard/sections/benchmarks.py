import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import torch
from summit_housing.dashboard.utils import get_trained_model_v2, get_trained_nn_model_v4
from summit_housing.tracking import tracker

def section_benchmarks():
    st.subheader("Model Status & Benchmarks")
    
    gbm_champ = tracker.get_champion("gbm")
    nn_champ = tracker.get_champion("price_net_macro")
    
    if gbm_champ and nn_champ:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style="background-color: #f8fafc; padding: 10px; border-radius: 8px; border-left: 5px solid #10b981;">
                <small>🌲 <b>GBM CHAMPION</b></small><br>
                <b>Run #{gbm_champ['run_id']}</b><br>
                <small>MAE: ${gbm_champ['metrics']['mae']:,.0f}</small>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background-color: #f8fafc; padding: 10px; border-radius: 8px; border-left: 5px solid #6366f1;">
                <small>🧠 <b>NEURAL NET CHAMPION</b></small><br>
                <b>Run #{nn_champ['run_id']}</b><br>
                <small>MAE: ${nn_champ['metrics']['mae']:,.0f}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.caption("Automated comparison of model architectures across standard property types.")
    
    gbm_pipeline = None
    nn_model = None
    nn_preprocessor = None
    nn_y_scaler = None
    
    try:
        gbm_pipeline, _, _, _, _, _ = get_trained_model_v2()
        nn_model, nn_preprocessor, nn_y_scaler, _, _, _, _ = get_trained_nn_model_v4()
    except Exception as e:
        st.error(f"Failed to load models for benchmarks: {e}")
        return

    with st.expander("📊 Benchmark Scenarios", expanded=True):
        st.write("Comparing GBM vs Neural Network.")
        locations = ['BRECKENRIDGE', 'FRISCO', 'SILVERTHORNE', 'DILLON']
        scenarios = []
        for loc in locations:
            scenarios.append({'desc': f"Entry Condo ({loc.title()})", 'city': loc, 'prop_type': 'Condo', 'sfla': 600, 'beds': 1, 'baths': 1, 'year_blt': 1980, 'garage_size': 0, 'acres': 0, 'grade_numeric': 4, 'cond_numeric': 4, 'scenic_view': 0})
        for loc in locations:
            scenarios.append({'desc': f"Family Home ({loc.title()})", 'city': loc, 'prop_type': 'Single Family', 'sfla': 2200, 'beds': 3, 'baths': 2.5, 'year_blt': 1995, 'garage_size': 400, 'acres': 0.25, 'grade_numeric': 5, 'cond_numeric': 5, 'scenic_view': 2})
        
        df_bench = pd.DataFrame(scenarios)
        df_bench['mortgage_rate'] = 6.5
        df_bench['sp500'] = 5000
        df_bench['cpi'] = 310.0
        df_bench['summit_pop'] = 31.0
        
        dist_map = {
            "BRECKENRIDGE":    {'dist_breck': 0.5, 'dist_keystone': 12, 'dist_copper': 16, 'dist_abasin': 18, 'dist_dillon': 10},
            "FRISCO":          {'dist_breck': 9,   'dist_keystone': 9,  'dist_copper': 6,  'dist_abasin': 13, 'dist_dillon': 4},
            "SILVERTHORNE":    {'dist_breck': 12,  'dist_keystone': 7,  'dist_copper': 11, 'dist_abasin': 10, 'dist_dillon': 1},
            "DILLON":          {'dist_breck': 11,  'dist_keystone': 5,  'dist_copper': 12, 'dist_abasin': 9,  'dist_dillon': 0.5}
        }
        
        for col in ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin', 'dist_dillon']:
            df_bench[col] = df_bench['city'].apply(lambda c: dist_map.get(c, {'dist_breck':10}).get(col, 10))
        df_bench['dist_to_lift'] = df_bench[['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']].min(axis=1)

        try:
            df_bench['GBM Estimate'] = gbm_pipeline.predict(df_bench)
        except:
             df_bench['GBM Estimate'] = 0

        try:
            X_bench_prepped = nn_preprocessor.transform(df_bench)
            X_bench_t = torch.FloatTensor(X_bench_prepped)
            nn_model.eval()
            with torch.no_grad():
                y_bench_scaled = nn_model(X_bench_t).numpy()
                y_bench_log = nn_y_scaler.inverse_transform(y_bench_scaled)
                df_bench['NN Estimate'] = np.expm1(y_bench_log).flatten()
        except:
            df_bench['NN Estimate'] = 0

        df_display = df_bench[['desc', 'sfla', 'year_blt', 'GBM Estimate', 'NN Estimate']].copy()
        df_display['Diff %'] = ((df_display['NN Estimate'] - df_display['GBM Estimate']) / df_display['GBM Estimate']) * 100
        st.dataframe(df_display.style.format({'GBM Estimate': '${:,.0f}', 'NN Estimate': '${:,.0f}', 'Diff %': '{:+.1f}%'}))
        
        fig = px.scatter(df_display, x='GBM Estimate', y='NN Estimate', hover_data=['desc'], title="GBM vs Neural Net (Linear=Match)")
        max_val = max(df_display['GBM Estimate'].max(), df_display['NN Estimate'].max())
        fig.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=max_val, y0=0, y1=max_val)
        st.plotly_chart(fig, use_container_width=True)
