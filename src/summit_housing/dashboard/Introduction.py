import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Fix for Streamlit Cloud imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from summit_housing.queries import MarketAnalytics

st.set_page_config(
    page_title="Introduction | Summit Housing",
    page_icon="🏔️",
    layout="wide"
)

def main():
    # --- Hero Section ---
    st.title("🏔️ Summit County Housing Analysis")
    st.subheader("Data Science & MLOps Portfolio Project")
    
    st.markdown("""
    ### Why this project?
    Summit County's real estate market operates like a high-beta financial asset. Unlike standard suburbs, prices here are driven by external capital, resort proximity, and strict geographical constraints. I built this platform to demonstrate how **Advanced SQL**, **Web Scraping**, and **Deep Learning** can be unified to model these complex dynamics.
    """)
    
    st.divider()
    
    # --- Navigation Guide ---
    st.markdown("### 🗺️ Module Guide")
    st.write("The dashboard is organized into three sequential modules:")
    
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    with col_nav1:
        st.markdown("#### 📊 1. Data Story")
        st.write("""
        **The Analysis**: Explores geographical pillars of value, macro-indicators (Interest Rates vs. Price), and 20-year demographic shifts.
        """)
        
    with col_nav2:
        st.markdown("#### 🧬 2. ML Experiments")
        st.write("""
        **The Registry**: View historical tournament results, compare GBM vs. NN metrics, and select specific model versions to load.
        """)
        
    with col_nav3:
        st.markdown("#### 🔮 3. Price Predictor")
        st.write("""
        **The Product**: An interactive "What-If" simulator for real-time price estimates from our current production Champion.
        """)

    st.divider()

    # --- Technical / MLOps Section ---
    st.markdown("### 🛠️ How to use the Product")
    st.write("This platform is built for continuous iteration. You can trigger the core workflows from the terminal:")

    c_ml1, c_ml2, c_ml3 = st.columns(3)
    
    with c_ml1:
        st.markdown("**1. Data Collection**")
        st.code("make scrape", language="bash")
        st.caption("Runs the asynchronous scraper to pull the latest property records.")

    with c_ml2:
        st.markdown("**2. ETL Pipeline**")
        st.code("make ingest", language="bash")
        st.caption("Resets the local SQLite warehouse and performs complex SQL feature engineering.")

    with c_ml3:
        st.markdown("**3. Model Training**")
        st.code("make tournament", language="bash")
        st.caption("Triggers a parameter sweep tournament. The best model is automatically promoted to production.")

    st.markdown("---")
    st.caption("Built with PyTorch, Scikit-Learn, Streamlit, and SQLite. | Summit Housing Portfolio v24.1")

if __name__ == "__main__":
    main()
