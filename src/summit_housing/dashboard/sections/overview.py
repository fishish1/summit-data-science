import streamlit as st
import pandas as pd
import pydeck as pdk
from summit_housing.queries import MarketAnalytics
from summit_housing.dashboard.utils import get_map_dataset, get_analytics_data

@st.cache_data
def get_raw_sample(all_cols=False):
    """Helper to sample the raw records CSV for the introduction."""
    try:
        # data/records.csv is the primary raw pull
        df = pd.read_csv("data/records.csv", nrows=50)
        if all_cols:
            return df.head(10)
        # Key human-readable columns
        cols = ['schno', 'address', 'city', 'year_blt', 'sfla', 'beds', 'f_baths', 'totactval']
        return df[cols].head(10)
    except:
        return pd.DataFrame()

def section_1_chart():
    analytics = MarketAnalytics()
    data_resorts = [
        {"name": "Breckenridge Ski Resort", "lat": 39.4805, "lon": -106.0666, "type": "Ski Resort", "color": [239, 68, 68, 200]}, # Red
        {"name": "Keystone Resort", "lat": 39.605, "lon": -105.9439, "type": "Ski Resort", "color": [239, 68, 68, 200]},
        {"name": "Copper Mountain", "lat": 39.5022, "lon": -106.1506, "type": "Ski Resort", "color": [239, 68, 68, 200]},
        {"name": "A-Basin", "lat": 39.6425, "lon": -105.8719, "type": "Ski Resort", "color": [239, 68, 68, 200]},
    ]
    
    data_towns = [
        {"name": "Breckenridge", "lat": 39.4817, "lon": -106.0384, "type": "Town Center", "color": [59, 130, 246, 200]}, # Blue
        {"name": "Frisco", "lat": 39.5744, "lon": -106.0975, "type": "Town Center", "color": [59, 130, 246, 200]},
        {"name": "Silverthorne", "lat": 39.6296, "lon": -106.0713, "type": "Town Center", "color": [59, 130, 246, 200]},
        {"name": "Dillon", "lat": 39.6303, "lon": -106.0434, "type": "Town Center", "color": [59, 130, 246, 200]},
    ]
    
    data_landmarks = [
        {"name": "Dillon Reservoir", "lat": 39.615, "lon": -106.05, "type": "Feature", "color": [6, 182, 212, 200]}, # Cyan
        {"name": "Summit High School", "lat": 39.553, "lon": -106.062, "type": "School", "color": [234, 179, 8, 200]}, # Yellow
    ]
    
    all_points = data_resorts + data_towns + data_landmarks
    df_map = pd.DataFrame(all_points)
    
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        df_map,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=800,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        get_line_width=50,
    )
    
    df_text = df_map.copy()
    
    text_layer = pdk.Layer(
        "TextLayer",
        df_text,
        get_position=["lon", "lat"],
        get_text="name",
        get_color=[60, 60, 60],
        get_size=16,
        get_angle=0,
        get_text_anchor="middle",
        get_alignment_baseline="center",
        get_pixel_offset=[0, -20],
        font_family="Space Grotesk, sans-serif",
        font_weight=600,
    )

    # ANIMATION LOGIC: US-wide to Summit County swoop
    if 'overview_map_state' not in st.session_state:
        # Step 1: Broad US View
        view_state = pdk.ViewState(
            latitude=39.8, 
            longitude=-98.5, 
            zoom=3.5, 
            pitch=0
        )
        st.session_state.overview_map_state = 'zooming'
        
        # Render the initial far-out view
        deck = pdk.Deck(layers=[scatter_layer, text_layer], initial_view_state=view_state, map_style=None)
        st.pydeck_chart(deck)
        
        # Let user see the US view for a second
        import time
        time.sleep(1.2)
        st.rerun()
        
    elif st.session_state.overview_map_state == 'zooming':
        # Step 2: Final Summit View with transition
        view_state = pdk.ViewState(
            latitude=39.55, 
            longitude=-106.05, 
            zoom=9.5, 
            pitch=0,
            transition_duration=3500
        )
        # Mark as done so we don't re-animate on every interaction
        st.session_state.overview_map_state = 'done'
        
        deck = pdk.Deck(layers=[scatter_layer, text_layer], initial_view_state=view_state, map_style=None)
        st.pydeck_chart(deck)
    else:
        # Step 3: Sustained Final View (no animation on subsequent reruns)
        view_state = pdk.ViewState(
            latitude=39.55, 
            longitude=-106.05, 
            zoom=9.5, 
            pitch=0
        )
        deck = pdk.Deck(layers=[scatter_layer, text_layer], initial_view_state=view_state, map_style=None)
        st.pydeck_chart(deck)
    
    
    try:
        trends = get_analytics_data("get_market_trends", exclude_multiunit=True)
        if not trends.empty:
            trends = trends.copy() # Avoid mutation of cached object
            trends['tx_year'] = pd.to_numeric(trends['tx_year'], errors='coerce')
            yearly_vol = trends.groupby('tx_year')['sales_count'].sum()
            valid_years = yearly_vol[yearly_vol > 50].index
            target_year = int(valid_years.max()) if not valid_years.empty else 2023
            
            year_data = trends[trends['tx_year'] == target_year]
            avg_price = year_data['avg_price'].mean()
            vol = year_data['sales_count'].sum()
            
            c1, c2 = st.columns(2)
            c1.metric(f"{target_year} Avg Price", f"${avg_price:,.0f}")
            c2.metric(f"{target_year} Volume", f"{vol} sales")
    except Exception as e:
        st.caption(f"Metrics unavailable: {e}")

    st.write("")
    with st.expander("📝 View Raw Data Sample (records.csv)"):
        show_all = st.checkbox("Show all columns (100+)", value=False)
        df_sample = get_raw_sample(all_cols=show_all)
        if not df_sample.empty:
            st.dataframe(df_sample, use_container_width=True)
            st.caption(f"Showing {'all' if show_all else 'curated'} columns from the raw Clerk & Recorder pull.")
        else:
            st.info("Raw CSV sample currently unavailable.")

def section_1_text():
    st.markdown("""
    ### 1. This is the Area
    **Summit County, Colorado** is known as "Colorado's Playground". It sits high in the Rockies (9,000+ ft elevation) and is defined by a unique geography that strictly limits where housing can be built.
    
    ### 2. The Major Features
    The map to the left highlights the four pillars of value in this market:
    
    *   🔴 **The Resorts (Ski Lifts):**  
        **Breckenridge, Keystone, Copper, and A-Basin.** Proximity to these lifts is the single biggest multiplier of property value.
        
    *   🔵 **The Town Centers:**  
        **Breckenridge & Frisco** are the historic, walkable hubs with restaurants and nightlife. **Silverthorne & Dillon** are the commercial hubs.
        
    *   🌊 **The Lake (Dillon Reservoir):**  
        The massive reservoir in the center creates a natural barrier between towns and offers summer value (views/boating), not just winter skiing.
        
    *   🟡 **The Local Zones:**  
        Areas like **Summit High School** (between Frisco and Breck) represent the "Local's Triangle"—neighborhoods where full-time residents actually live, distinct from the short-term rental zones.
        
    ---
    **The Dataset:**
    This analysis is built by connecting three disparate data sources:
    
    1.  **Current Ownership Records:**  
        The baseline dataset from the Summit County Clerk & Recorder, providing the *current* state of every parcel (Owner, Address, Legal Description).
        
    2.  **Historical Sales Pivot:**  
        We reconstructed the market history by pivoting the recording dates and doc fees from the ownership table into a time-series of sales events.
        
    3.  **Supplemental Market Data:**  
        Enriched with external economic indicators (S&P 500, Interest Rates) and geospatial engineering (Distances to ski lifts) to provide context to the raw transactions.
    """, unsafe_allow_html=True)
