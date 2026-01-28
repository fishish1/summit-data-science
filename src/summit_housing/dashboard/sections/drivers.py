import streamlit as st
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from summit_housing.dashboard.utils import (
    get_trained_model_v2, 
    get_analytics_data, 
    get_ml_feature_analysis,
    get_shap_analysis_data,
    get_pdp_data_cached
)

def section_3_chart():
    tab_features, tab_shap, tab_corr, tab_pdp = st.tabs(["Variable Selection", "SHAP Importance", "Correlation Matrix", "Partial Dependence"])
    
    with tab_shap:
        shap_placeholder = st.empty()
        shap_placeholder.markdown("""
        <div style="padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background-color: #fafafa; color: #666;">
            ⏳ <b>Calculating SHAP Values...</b>
        </div>
        """, unsafe_allow_html=True)
        
    with tab_corr:
        corr_placeholder = st.empty()
        corr_placeholder.markdown("""
        <div style="padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background-color: #fafafa; color: #666;">
            ⏳ <b>Computing Correlation Matrix...</b>
        </div>
        """, unsafe_allow_html=True)

    with tab_pdp:
        pdp_placeholder = st.empty()

    with tab_features:
        st.markdown("#### 🛠️ Model Feature Selection")
        c1, c2 = st.columns(2)
        with c1:
            st.success("**✅ INCLUDED Variables**")
            st.markdown("* Structure, Quality, Location, Macro")
        with c2:
            st.error("**❌ EXCLUDED Variables**")
            st.markdown("* Tax Assessor Value, Transaction Date, Subdivision Name, Street Address")
    
    with tab_shap:
        try:
            shap_values, X_test_transformed, shap_cols = get_shap_analysis_data()
            shap_placeholder.empty()
            
            # More stable plotting for Streamlit
            shap.summary_plot(shap_values, X_test_transformed, feature_names=shap_cols, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close()
        except Exception as e:
            shap_placeholder.empty()
            st.error(f"Could not load ML Model: {e}")
            
    with tab_corr:
        try:
            _, corr_df = get_ml_feature_analysis()
            corr_placeholder.empty()
            fig_corr = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            corr_placeholder.empty()
            st.error(f"Could not load Correlation Matrix: {e}")
            
    with tab_pdp:
        pdp_placeholder.empty()
        st.caption("Marginal effect of specific features on price.")
        feature = st.selectbox("Select Feature for PDP", ["sfla", "year_blt", "dist_to_lift"])
        with st.spinner(f"Computing PDP for {feature}..."):
            df_pdp = get_pdp_data_cached(feature)
            if not df_pdp.empty:
                fig_pdp = px.line(df_pdp, x='value', y='average_prediction', title=f"Partial Dependence: {feature}")
                st.plotly_chart(fig_pdp, use_container_width=True)

def section_3_text():
    st.write("""
    **What Actually Drives the Price?**
    We trained a **Gradient Boosting Regressor** to predict sales prices.
    * **SHAP:** Shows global feature importance.
    * **Correlation:** Shows relationships between features.
    * **PDP:** Shows marginal effects of features on price.
    """)
    st.info("""
    **Safeguards:**
    1. Time-Based Split
    2. Log-Space Target Transformation
    """)
