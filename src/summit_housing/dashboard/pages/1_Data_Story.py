import streamlit as st
import plotly.express as px
import plotly.io as pio

from summit_housing.dashboard.sections.overview import section_1_chart, section_1_text
from summit_housing.dashboard.sections.market import section_2_chart, section_2_text
from summit_housing.dashboard.sections.drivers import section_3_chart, section_3_text
from summit_housing.dashboard.sections.benchmarks import section_benchmarks

# --- Configuration ---
st.set_page_config(
    page_title="Summit Housing Data Story",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Global Plotly defaults
pio.templates.default = "plotly_white"
px.defaults.color_discrete_sequence = ["#0f1b2c", "#12b3b6", "#f59e0b", "#8aa5bf"]

# --- Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
    :root {
        --ink: #0f1b2c;
        --panel: #ffffff;
        --bg: #f8fafc;
        --accent: #3b82f6;
    }
    html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
        background-color: var(--bg);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max_width: 1400px;
    }
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
    h3 { font-size: 1.25rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0; }
    
    .story-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #334155;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helper for Layout ---
def story_section(title, subtitle, chart_func, text_func):
    st.markdown(f"### {title}")
    st.markdown(f"# {subtitle}")
    st.write("") 
    
    col_chart, col_text = st.columns([1.5, 1], gap="large")
    
    with col_chart:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        chart_func()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_text:
        st.markdown('<div class="story-text">', unsafe_allow_html=True)
        text_func()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()

# --- Main App Skeleton ---
def main():
    st.title("Summit County Housing Analysis")
    st.caption("A Data Science Portfolio Project")
    st.markdown("---")
    
    container_1 = st.container()
    container_2 = st.container()
    container_3 = st.container()
    container_4 = st.container()
    container_cta = st.container()
    
    with container_4:
        pending_4 = st.empty()
        pending_4.markdown("""
        ### 4. Model Status & Benchmarks
        <div style="padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background-color: #fafafa; color: #666;">
            ⏳ <b>Waiting for upstream analysis...</b>
        </div>
        """, unsafe_allow_html=True)
        
    with container_1:
        story_section("1. The Data", "What are we looking at?", section_1_chart, section_1_text)
    
    with container_2:
        story_section("2. The Context", "Market Forces & Trends", section_2_chart, section_2_text)
    
    with container_3:
        story_section("3. The Drivers", "Feature Importance Analysis", section_3_chart, section_3_text)
    
    pending_4.empty()
    with container_4:
        section_benchmarks()
    
    with container_cta:
        st.markdown("---")
        col_cta1, col_cta2 = st.columns([2, 1])
        with col_cta1:
            st.markdown("### 🔮 Ready to predict future prices?")
            st.write("Use the interactive inference engine to simulate property values.")
        with col_cta2:
            st.info("👈 **Select 'Predictor' in the sidebar** to start.")

if __name__ == "__main__":
    main()
