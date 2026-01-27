import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shap
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

from summit_housing.queries import MarketAnalytics
from summit_housing.ml import get_shap_values, analyze_features, get_pdp_data
from summit_housing.dashboard.utils import get_trained_model_v2, get_trained_nn_model_v4
from summit_housing.geo import RESORT_LIFTS, enrich_with_geo_features
import torch # Needed for inference

@st.cache_data
def get_map_dataset():
    """
    Loads unique properties and calculates distance distances for visualization.
    """
    # 1. Get raw property table (using training data fetcher for simplicity)
    analytics = MarketAnalytics()
    df = analytics.get_training_data()
    
    # 2. Key by Schedule Number (Unique Property) to avoid plotting duplicates
    if 'schno' in df.columns:
        df = df.drop_duplicates(subset=['schno'])
    elif 'schedule_number' in df.columns:
        df = df.drop_duplicates(subset=['schedule_number'])
        
    # 3. Enrich with Lat/Lon/Dists
    # Note: This relies on Address_Points.csv being present in workspace root or data/
    try:
        df = enrich_with_geo_features(df)
        
        # 4. Calculate 'dist_to_lift' (Min of ski resorts only)
        # Excludes Dillon (Lake) for the coloring logic
        lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
        valid_lifts = [c for c in lift_cols if c in df.columns]
        
        if valid_lifts:
            df['dist_to_lift'] = df[valid_lifts].min(axis=1)
            # Remove properties without valid geo data
            df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'dist_to_lift'])
            
            # Normalize for color mapping (0 to 1) for PyDeck
            # We want RED = Close, BLUE = Far
            # Clip at 10 miles to keep contrast
            MAX_VAL = 10.0
            df['dist_clamped'] = df['dist_to_lift'].clip(upper=MAX_VAL)
            df['norm'] = df['dist_clamped'] / MAX_VAL 
            
            # Simple Heatmap Gradient: (R, G, B)
            # Close (0.0) -> Red (255, 0, 0)
            # Far (1.0) -> Blue (0, 0, 255)
            # We'll use a simple linear interpolation
            df['r'] = (1 - df['norm']) * 255
            df['g'] = 50 # slight constant green
            df['b'] = df['norm'] * 255
        
        # Mapping Address
        # 'FullAddress' comes from the Geo Join, 'address' typically from SQL
        if 'FullAddress' in df.columns:
            df['address'] = df['FullAddress']
        elif 'address' not in df.columns:
            df['address'] = "Unknown Address"

        return df[['LATITUDE', 'LONGITUDE', 'dist_to_lift', 'address', 'r', 'g', 'b']]
        
    except Exception as e:
        st.error(f"Geo enrichment failed: {e}")
        return pd.DataFrame()

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
    .highlight {
        background: linear-gradient(120deg, #dbeafe 0%, #dbeafe 100%);
        background-repeat: no-repeat;
        background-size: 100% 0.3em;
        background-position: 0 88%;
        font-weight: 500;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--ink);
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Data Loading (Cached) ---
@st.cache_resource
def get_analytics():
    return MarketAnalytics()

with st.spinner("Connecting to Data Warehouse..."):
    analytics = get_analytics()

# --- Helper for Layout ---
def story_section(title, subtitle, chart_func, text_func, height=600):
    st.markdown(f"### {title}")
    st.markdown(f"# {subtitle}")
    st.write("") # Spacer
    
    col_chart, col_text = st.columns([1.5, 1], gap="large")
    
    with col_chart:
        # We removed the outer spinner to prevent layout shifts during interaction
        st.markdown('<div class="card">', unsafe_allow_html=True)
        chart_func()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_text:
        st.markdown('<div class="story-text">', unsafe_allow_html=True)
        text_func()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()

# --- Section 1: Data Overview ---
def section_1_chart():
    # Define our custom layers of interest
    # We manually define these to ensure the "Story" is clear, rather than just raw data.
    
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
    
    # Layer 1: Colored Dots
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
    
    # Layer 2: Text Labels (Offset slightly)
    # create a slightly offset dataframe for text
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
        get_pixel_offset=[0, -20], # Move text up above the dot
        font_family="Space Grotesk, sans-serif",
        font_weight=600,
    )

    view_state = pdk.ViewState(
        latitude=39.55, 
        longitude=-106.05, 
        zoom=9.5, 
        pitch=0 # Flat view is better for reading labels
    )
    
    deck = pdk.Deck(
        layers=[scatter_layer, text_layer],
        initial_view_state=view_state,
        map_style=None, # Use default map style to ensure it loads
        tooltip={"text": "{name} ({type})"}
    )
    st.pydeck_chart(deck)
    
    # 2. Key Metrics Row (Mini)
    with st.spinner("Loading key metrics..."):
        try:
            trends = analytics.get_market_trends(exclude_multiunit=True)
            if not trends.empty:
                # Ensure year is numeric for filtering
                trends['tx_year'] = pd.to_numeric(trends['tx_year'], errors='coerce')
                
                # Get latest full year (filtering out incomplete future/current years with low volume if needed)
                # For this dataset, we'll look for the most recent year with substantial data (>50 sales)
                yearly_vol = trends.groupby('tx_year')['sales_count'].sum()
                valid_years = yearly_vol[yearly_vol > 50].index
                target_year = int(valid_years.max()) if not valid_years.empty else 2023
                
                # Filter
                year_data = trends[trends['tx_year'] == target_year]
                
                avg_price = year_data['avg_price'].mean()
                vol = year_data['sales_count'].sum()
                
                c1, c2 = st.columns(2)
                c1.metric(f"{target_year} Avg Price", f"${avg_price:,.0f}")
                c2.metric(f"{target_year} Volume", f"{vol} sales")
        except Exception as e:
            st.caption(f"Metrics unavailable: {e}")

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

# --- Section 2: Market Factors ---
def section_2_chart():
    # We will use tabs to show multiple dimensions of the market
    tab1, tab2, tab3, tab4 = st.tabs(["Price vs Rates", "Buyer Origins", "Supply Growth", "Lift Proximity Matrix"])
    
    # --- PRE-LOAD DATA INTO SESSION STATE ---
    # This ensures that when the user interacts with dropdowns (Tab 2), 
    # the entire section doesn't re-fetch or flash spinners for other tabs.
    
    if 'sec2_trends' not in st.session_state:
        with st.spinner("Analyzing Price vs Rates..."):
            st.session_state['sec2_trends'] = analytics.get_market_trends(exclude_multiunit=True)

    if 'sec2_owners' not in st.session_state:
        with st.spinner("Segmenting Buyer Demographics..."):
            st.session_state['sec2_owners'] = analytics.get_owner_purchase_trends()

    if 'sec2_props' not in st.session_state:
        with st.spinner("Generating GEOSPATIAL layers..."):
             st.session_state['sec2_props'] = get_map_dataset()

    with tab4:
        # Distance to Lift Map
        st.markdown("#### 🎿 Ski Resort Proximity & Town Centers")
        st.caption("A simplified 2D view. **Dots** = Properties. **Color** = Distance to Lift (Red=Close, Blue=Far).")

        # 1. Load Property Data
        df_props = st.session_state['sec2_props']
        
        if not df_props.empty:
            # 2. Prepare Map Data
            
            # Layer A: Properties
            layer_props = pdk.Layer(
                "ScatterplotLayer",
                df_props,
                get_position='[LONGITUDE, LATITUDE]',
                get_fill_color='[r, g, b, 140]', # Semi-transparent
                get_radius=80, # Small radius for density
                pickable=True,
            )
            
            # Layer B: Resort Bases (White with Red Stroke)
            resort_data = []
            for resort, coords in RESORT_LIFTS.items():
                if resort == 'dist_dillon': continue # Skip Dillon for Ski Map
                name = resort.replace('dist_', '').title()
                resort_data.append({'name': f"{name} Base", 'lat': coords[0][0], 'lon': coords[0][1]})
            
            layer_resorts = pdk.Layer(
                "ScatterplotLayer",
                pd.DataFrame(resort_data),
                get_position='[lon, lat]',
                get_color=[255, 255, 255, 255], 
                get_line_color=[0, 0, 0, 200],
                get_radius=500,
                stroked=True,
                line_width_min_pixels=2,
                pickable=True,
            )
            
            # Layer C: Labels
            layer_text = pdk.Layer(
                "TextLayer",
                pd.DataFrame(resort_data),
                get_position='[lon, lat]',
                get_text='name',
                get_color=[0, 0, 0, 255],
                get_size=14,
                get_alignment_baseline="'top'", # Text below dot
                get_text_anchor="'middle'",
                get_background_color=[255, 255, 255, 200],
                get_background_padding=[4, 4]
            )

            # Top-Down 2D View
            view_state = pdk.ViewState(
                latitude=39.55,
                longitude=-106.05,
                zoom=10,
                pitch=0
            )

            st.pydeck_chart(pdk.Deck(
                map_style=None, # Use default streamlit map style which works without token
                initial_view_state=view_state,
                layers=[layer_props, layer_resorts, layer_text],
                tooltip={"text": "{address}\nDist: {dist_to_lift:.1f} miles"}
            ))
        else:
            st.warning("Could not load property data for map.")
    
    with tab1:
        # Price vs Interest Rates Bubble Chart
        trends = st.session_state['sec2_trends']
        if not trends.empty:
            trends['tx_year'] = pd.to_numeric(trends['tx_year'], errors='coerce')
            
            all_cities = ["BRECKENRIDGE", "FRISCO", "SILVERTHORNE", "DILLON", "KEYSTONE"]
            df_filtered = trends[trends['city'].isin(all_cities)]
            
            fig = go.Figure()
            
            # Add Mortgage Rates (Context)
            try:
                df_mortgage = pd.read_csv("data/mortgage_rate.csv")
                df_mortgage['year'] = pd.to_datetime(df_mortgage['date']).dt.year
                df_rates = df_mortgage.groupby('year')['value'].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=df_rates['year'], y=df_rates['value'],
                    name="30y Mortgage Rate",
                    line=dict(color='red', width=2, dash='dot'),
                    yaxis='y2'
                ))
            except:
                pass
                
            for city in all_cities:
                city_data = df_filtered[df_filtered['city'] == city]
                fig.add_trace(go.Scatter(
                    x=city_data['tx_year'], y=city_data['avg_price_3yr_ma'],
                    mode='lines', name=city,
                    line=dict(width=3)
                ))
                
            fig.update_layout(
                title="Price Trends vs. Interest Rates",
                xaxis_title="Year",
                yaxis_title="3-Year Moving Avg Price ($)",
                yaxis2=dict(title="Interest Rate (%)", overlaying='y', side='right', showgrid=False, range=[0, 15]),
                legend=dict(orientation="h", y=1.1, x=0),
                height=450,
                margin=dict(l=20, r=20, t=80, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Buyer Origins (In-State vs Out-of-State)
        try:
            df_owners = st.session_state['sec2_owners']
            
            # View Controls
            c_view, c_stat = st.columns([2,1])
            with c_view:
                view_mode = st.radio("Display Mode", ["Volume (Count)", "Market Share (%)"], horizontal=True, label_visibility="collapsed")
            
            # Prepare Data & Config
            if view_mode == "Market Share (%)":
                groupnorm = 'percent'
                y_title = "Market Share (%)"
                title = "Buyer Origin: Market Share Evolution"
            else:
                groupnorm = None
                y_title = "Number of Buyers"
                title = "Buyer Origin: Volume Trends"

            fig_owners = px.area(
                df_owners, 
                x='purchase_year', 
                y='buyer_count', 
                color='location_type',
                title=title,
                labels={'buyer_count': y_title, 'purchase_year': 'Year Bought'},
                groupnorm=groupnorm,
                color_discrete_map={
                    'Local (In-County)': '#10b981', # Green
                    'In-State (Non-Local)': '#3b82f6', # Blue
                    'Out-of-State': '#ef4444' # Red
                },
                # Enforce stacking order: Bottom -> Top
                category_orders={'location_type': ['Local (In-County)', 'In-State (Non-Local)', 'Out-of-State']}
            )
            
            if view_mode == "Market Share (%)":
                fig_owners.update_layout(yaxis=dict(ticksuffix="%"))
            
            st.plotly_chart(fig_owners, use_container_width=True)
                
            # Dynamic Analysis Text
            st.markdown("#### 💡 Trend Analysis")
            
            # Helper to get share
            def get_oos_share(year):
                year_data = df_owners[pd.to_numeric(df_owners['purchase_year']) == year]
                if year_data.empty: return 0
                total = year_data['buyer_count'].sum()
                oos = year_data[year_data['location_type'] == 'Out-of-State']['buyer_count'].sum()
                return (oos / total) * 100 if total > 0 else 0

            share_2000 = get_oos_share(2000)
            share_2021 = get_oos_share(2021)
            
            st.write(f"""
            **Market Shift:**
            *   **2000:** Out-of-State buyers were **{share_2000:.1f}%** of the market.
            *   **2021 (Peak):** This reached **{share_2021:.1f}%**.
            """)

            st.info("""
            **Buyer Personas (Based on Data):**
            *   **🏔️ Local Buyers:** *Rate Agnostic*. Driven by necessity, not interest rates.
            *   **🚙 In-State Buyers (Front Range):** *Rate Sensitive*. Deal hunters who flood the market when rates are low.
            *   **✈️ Out-of-State Buyers:** *The Safety Net*. Capital-rich investors who often step in when rates are high and others retreat.
            """)
            
            st.divider()
            
            # --- CORRELATION LAB ---
            st.subheader("🧪 Hypothesis Tester")
            st.caption("Don't just take our word for it. Test if external factors actually drive buyer behavior.")
            
            # 1. Prepare Data for Correlation
            # Flatten Owner Data to Annual Share/Volume
            df_pivot = df_owners.pivot_table(index='purchase_year', columns='location_type', values='buyer_count', aggfunc='sum').fillna(0)
            df_pivot.reset_index(inplace=True)
            df_pivot['year'] = pd.to_numeric(df_pivot['purchase_year'])
            df_pivot['Total Volume'] = df_pivot.sum(axis=1, numeric_only=True) - df_pivot['year'] # subt year col
            
            # Calculate Shares for ALL groups
            df_pivot['Out-of-State Share (%)'] = (df_pivot['Out-of-State'] / df_pivot['Total Volume']) * 100
            if 'In-State (Non-Local)' in df_pivot.columns:
                df_pivot['In-State Share (%)'] = (df_pivot['In-State (Non-Local)'] / df_pivot['Total Volume']) * 100
            if 'Local (In-County)' in df_pivot.columns:
                df_pivot['Local Share (%)'] = (df_pivot['Local (In-County)'] / df_pivot['Total Volume']) * 100
            
            # Load Drivers
            try:
                sp500 = pd.read_csv("data/sp500_annual.csv") # year, avg_value, return_pct
                rates = pd.read_csv("data/mortgage_rate.csv")
                rates['year'] = pd.to_datetime(rates['date']).dt.year
                rates_annual = rates.groupby('year')['value'].mean().reset_index(name='Mortgage Rate (%)')
                
                # Calculate Rate Velocity (Acceleration/Deceleration)
                rates_annual['Mortgage Rate Change (YoY)'] = rates_annual['Mortgage Rate (%)'].diff()
                
                # Merge All
                df_macro = pd.merge(df_pivot, sp500, on='year', how='inner')
                df_macro = pd.merge(df_macro, rates_annual, on='year', how='inner')
                
                # Rename for UI
                df_macro.rename(columns={
                    'return_pct': 'S&P 500 Return (%)',
                    'avg_value': 'S&P 500 Price ($)',
                    'Total Volume': 'Total Sales Volume'
                }, inplace=True)
                
                # --- 1. CORRELATION MATRIX (The Heatmap) ---
                # Define relevant columns for the matrix
                matrix_cols = [
                    'Out-of-State Share (%)', 
                    'In-State Share (%)',
                    'Local Share (%)',
                    'Total Sales Volume', 
                    'S&P 500 Return (%)', 
                    'Mortgage Rate (%)',
                    'Mortgage Rate Change (YoY)'
                ]
                # Filter only columns that exist (in case of rename issues)
                valid_cols = [c for c in matrix_cols if c in df_macro.columns]
                
                corr_matrix = df_macro[valid_cols].corr()
                
                fig_matrix = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Buyer Sensitivity Matrix"
                )
                st.plotly_chart(fig_matrix, use_container_width=True)
                st.caption("""
                **How to read this matrix:**
                *   **Color Scale:** 🔴 **Red** = Positive Correlation, 🔵 **Blue** = Negative Correlation.
                *   **Out-of-State Share vs Rates:** Likely **Negative (Blue)** (Investors pull back when rates rise).
                *   **Local Share vs Rates:** Likely **Positive (Red)** (Locals gain relative market share when investors leave).
                """)
                
                st.info("""
                **Observation on In-State Buyers:**  
                You asked about In-State buyers and low rates.  
                If you check the intersection of `In-State Share (%)` and `Mortgage Rate (%)`, you will likely see a **Negative Correlation** (Blue). 
                This confirms your intuition: As rates go **Up**, In-State share goes **Down**. Therefore, In-State buyers are most active (relative to others) when rates are **Low**.
                """)

                st.divider()

                # --- 2. SCATTER PLOT (Deep Dive) ---
                # UI Controls
                c_x, c_y = st.columns(2)
                with c_x:
                    x_axis = st.selectbox("Market Driver (X-Axis)", ['S&P 500 Return (%)', 'S&P 500 Price ($)', 'Mortgage Rate (%)', 'Mortgage Rate Change (YoY)'])
                with c_y:
                    y_axis = st.selectbox("Market Response (Y-Axis)", ['Out-of-State Share (%)', 'In-State Share (%)', 'Local Share (%)', 'Total Sales Volume'])

                # Scatter Plot (The Truth)
                fig_scatter = px.scatter(
                    df_macro, 
                    x=x_axis, 
                    y=y_axis, 
                    hover_data=['year'],
                    trendline="ols", # Add regression line
                    title=f"Correlation Plot: {x_axis} vs {y_axis}",
                    color='year' # Color by year to show time progression
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Stats (Scientific Proof)
                valid_data = df_macro[[x_axis, y_axis]].dropna()
                if len(valid_data) > 2:
                    corr, p_value = pearsonr(valid_data[x_axis], valid_data[y_axis])
                    
                    # Interpretation
                    strength = "Weak"
                    if abs(corr) > 0.3: strength = "Moderate"
                    if abs(corr) > 0.6: strength = "Strong"
                    
                    sig_label = "✅ Statistically Significant" if p_value < 0.05 else "❌ Not Significant"
                    
                    st.success(f"""
                    **Statistical Verdict: {sig_label}**
                    
                    *   **Correlation (r):** `{corr:.2f}` ({strength} relationship)
                    *   **P-Value:** `{p_value:.4f}`
                    
                    **Proving the Link:**
                    With a p-value of **{p_value:.4f}**, there is a **{(1-p_value)*100:.1f}%** confidence level that this relationship is not random chance.
                    """)
                else:
                    st.warning("Not enough data points for statistical significance.")

            except Exception as e:
                st.warning(f"Could not calc correlations: {e}")

        except Exception as e:
            st.write(f"Could not load owner trends: {e}")

    with tab3:
        # Supply vs Population (SqFt per Resident)
        try:
            # 1. Get Cumulative Supply (Total SqFt Built)
            df_supply = analytics.get_cumulative_supply()
            
            # 2. Get Population
            df_pop = pd.read_csv("data/summit_pop.csv")
            df_pop['population'] = df_pop['value'] * 1000
            df_pop['year'] = df_pop['year'].astype(int)
            
            # 3. Merge
            df_merged = pd.merge(df_supply, df_pop, left_on='year_blt', right_on='year', how='inner')
            df_merged['sqft_per_resident'] = df_merged['cumulative_sqft'] / df_merged['population']
            
            # Create Dual Axis Chart
            fig = go.Figure()
            
            # Trace 1: Total Supply (Area)
            fig.add_trace(go.Scatter(
                x=df_merged['year'],
                y=df_merged['cumulative_sqft'],
                name="Total Residential SqFt",
                fill='tozeroy',
                line=dict(color='#cbd5e1', width=0), # Light gray
                hovertemplate="<b>%{x}</b><br>Total Supply: %{y:,.0f} sqft<extra></extra>"
            ))
            
            # Trace 2: Per Resident (Line) - Secondary Axis
            fig.add_trace(go.Scatter(
                x=df_merged['year'],
                y=df_merged['sqft_per_resident'],
                name="SqFt per Resident",
                line=dict(color='#8b5cf6', width=3),
                yaxis='y2',
                hovertemplate="Density: %{y:,.0f} sqft/person<extra></extra>"
            ))
            
            fig.update_layout(
                title="Supply Growth vs. Population Density",
                xaxis_title="Year",
                yaxis=dict(title="Total Built SqFt (Supply)", showgrid=False),
                yaxis2=dict(
                    title="SqFt per Resident",
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                legend=dict(x=0, y=1.1, orientation='h'),
                hovermode="x unified",
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("""
            **The "Empty Home" Paradox:**
            The Grey area shows the massive accumulation of built space in the county. 
            The Purple line shows how much space theoretically exists for each resident.
            The fact that the purple line is **rising** means we are building housing *faster* than the population is growing. 
            This excess capacity is the "Second Home" inventory.
            """)
            
        except Exception as e:
            st.write(f"Could not load supply/pop comparison: {e}")

def section_2_text():
    st.write("""
    **Understanding the Market Force**
    
    Summit County is a **"High Beta"** housing market. Unlike a standard suburb where prices are pegged to local wages, this market acts more like a financial asset.
    
    **1. Capital Flow (Buyer Origins):**
    Check the **"Buyer Origins"** tab. You'll see that "Out-of-State" (Red) and "In-State Non-Local" (Blue - likely Denver) buyers make up a massive portion of the market volume. This external capital flow makes prices sensitive to national wealth effects (like the S&P 500) rather than local job growth.
    
    **2. Macro Economics:** 
    Notice the inverse correlation with interest rates (Red Dotted Line in the first chart). When money is cheap (2020-2021), prices explode.
    
    **3. Inelastic Supply:** 
    The **"Supply Growth"** tab shows distinct eras of construction. Note how condo construction (Orange) boomed in the 70s/80s and has slowed, while Single Family homes have seen steady but constrained growth. You can't build more mountain.
    """)

# --- Section 3: Feature Importance ---
def section_3_chart():
    tab_features, tab_shap, tab_corr, tab_pdp = st.tabs(["Variable Selection", "SHAP Importance", "Correlation Matrix", "Partial Dependence"])
    
    # 1. Initialize Placeholders for Heavy Tabs
    # This prevents users from seeing blank screens if they switch tabs while others are determining
    with tab_shap:
        shap_placeholder = st.empty()
        shap_placeholder.markdown("""
        <div style="padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background-color: #fafafa; color: #666;">
            ⏳ <b>Calculating SHAP Values...</b><br>
            <small>Deconstructing model predictions to find global feature importance.</small>
        </div>
        """, unsafe_allow_html=True)
        
    with tab_corr:
        corr_placeholder = st.empty()
        corr_placeholder.markdown("""
        <div style="padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background-color: #fafafa; color: #666;">
            ⏳ <b>Computing Correlation Matrix...</b><br>
            <small>Analyzing multi-collinearity between feature pairs.</small>
        </div>
        """, unsafe_allow_html=True)

    with tab_pdp:
        pdp_placeholder = st.empty()
        pdp_placeholder.markdown("""
        <div style="padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background-color: #fafafa; color: #666;">
            ⏳ <b>Generating Partial Dependence...</b><br>
            <small>Calculating marginal effects of features on price.</small>
        </div>
        """, unsafe_allow_html=True)

    
    with tab_features:
        st.markdown("#### 🛠️ Model Feature Selection")
        st.write("We do not simply throw every column into the machine. We curate features to prevent data leakage and noise.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("**✅ INCLUDED Variables**")
            st.markdown("""
            *   **Structure:** SqFt, Beds, Baths, Year Built, Garage Size, Lot Size
            *   **Quality:** Construction Grade (1-6), Condition (1-6), Scenic View Score
            *   **Location:** City, Distance to Ski Lifts, Distance to Lake Dillon
            *   **Macro:** 30y Mortgage Rate, S&P 500 Index, CPI (Inflation), Local Population
            """)
            
        with c2:
            st.error("**❌ EXCLUDED Variables**")
            st.markdown("""
            *   **Tax Assessor Value:** This is a lagging indicator and often updated *after* a sale. Using it would be "cheating" (Data Leakage).
            *   **Transaction Date:** We want the model to learn *market conditions* (Rates/S&P), not just memorize that "2022 was expensive."
            *   **Subdivision Name:** Too many unique values (High Cardinality) causing overfitting. Use Geospatial usage instead.
            *   **Street Address:** Replaced by GPS Coordinates/Distances.
            """)
    
    with tab_shap:
        # SHAP Summary Plot Image
        # Note: We keep the spinner as well for the active "doing work" indicator if the user is on this tab
        try:
            pipeline, X_test, y_test, input_cols, shap_cols = get_trained_model_v2()
            explainer, shap_values = get_shap_values(pipeline, X_test)
            preprocessor = pipeline.named_steps['preprocessor']
            X_test_transformed = preprocessor.transform(X_test)
            
            # Clear placeholder before showing result
            shap_placeholder.empty()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            # Remove plot_type="bar" to show directional impact (beeswarm plot)
            shap.summary_plot(shap_values, X_test_transformed, feature_names=shap_cols, show=False)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            shap_placeholder.empty()
            st.error(f"Could not load ML Model: {e}")
    
    with tab_corr:
        # Correlation Matrix using analyze_features
        try:
            imp_df, corr_df = analyze_features()
            
            # Clear placeholder
            corr_placeholder.empty()
            
            fig_corr = px.imshow(
                corr_df, 
                text_auto=".2f",
                aspect="auto", 
                color_continuous_scale="RdBu_r", 
                zmin=-1, zmax=1,
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            corr_placeholder.empty()
            st.error(f"Could not load Correlation Matrix: {e}")
            
    with tab_pdp:
        # Clear placeholder immediately as we show UI controls first
        pdp_placeholder.empty()
        
        # Partial Dependence Plot
        st.caption("Marginal effect of specific features on price (holding others constant).")
        feature = st.selectbox("Select Feature for PDP", ["sfla", "year_blt", "dist_to_lift"])
        
        # We need a NEW spinner or placeholder for the interactive part
        with st.spinner(f"Computing PDP for {feature}..."):
            df_pdp = get_pdp_data(feature)
            if not df_pdp.empty:
                fig_pdp = px.line(
                    df_pdp, 
                    x='value', 
                    y='average_prediction',
                    title=f"Partial Dependence: {feature}",
                    labels={'value': feature, 'average_prediction': 'Predicted Price ($)'}
                )
                st.plotly_chart(fig_pdp, use_container_width=True)
            else:
                st.warning("Could not calculate PDP.")

def section_3_text():
    st.write("""
    **What Actually Drives the Price?**
    
    We trained a **Gradient Boosting Regressor** to predict sales prices. Using multiple interpretation techniques, we can verify our "Story":
    
    **1. SHAP (Global Importance):**
    As shown in the first tab, **Square Footage (SFLA)** and **Year Built** are the dominant factors. Location (Distance to Lift) is a strong secondary factor.
    
    **2. Correlation Matrix:**
    This allows us to see multi-collinearity. For example, `year_blt` might correlate with `baths` (newer homes have more bathrooms).
    
    **3. Partial Dependence (PDP):**
    This is the most powerful view. Select **`dist_to_lift`** in the PDP tab. 
    *   You will likely see a steep drop-off in price as you move away from the lifts (0-1 miles).
    *   Select **`year_blt`**: You often see a "U-shape" or a flat period for 80s homes, then a sharp rise for post-2000 construction.
    """)
    
    st.info("""
    **Scientific Rigor (Model Architecture):**
    To ensure this model is robust and not just "memorizing" the past, we implemented two key scientific safeguards:
    
    1.  **Time-Based Split (No Forward-Looking Bias):**  
        Instead of a random shuffle, we trained the model on the *First 80%* of historical sales and tested it on the *Last 20%* (Future). This simulates real-world conditions where we cannot peer into the future.
    
    2.  **Log-Space Target Transformation:**  
        Tree-based models (like GBMs) typically cannot predict values higher than what they saw in training. To fix this, we trained the model to predict the **Logarithm of the Price** (`log1p`), not the price itself. This allows the model to extrapolate trends exponentially, ensuring that future price projections aren't artificially capped by historical highs.
    """)

# --- Section 4: Model Benchmarks ---
def section_benchmarks():
    st.subheader("Model Status & Benchmarks")
    st.caption("Automated comparison of model architectures across standard property types.")
    
    # 0. Load Model
    gbm_pipeline = None
    nn_model = None
    nn_preprocessor = None
    nn_y_scaler = None
    
    with st.spinner(f"Loading Models for Benchmarking..."):
        try:
            gbm_pipeline, _, _, _, _ = get_trained_model_v2()
            nn_model, nn_preprocessor, nn_y_scaler, _ = get_trained_nn_model_v4()
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            return

    # --- FEATURE: Model Benchmarks ---
    with st.expander("📊 Benchmark Scenarios", expanded=True):
        st.write("Comparing the **Gradient Boosting Machine (GBM)** (Tree-based) vs. **Deep Learning (NN)** (Neural Network) across typical property profiles.")
        
        # 2. Define Base Scenarios (Must match training data casing - UPPERCASE)
        locations = ['BRECKENRIDGE', 'FRISCO', 'SILVERTHORNE', 'DILLON']
        
        # Create synthetic data
        scenarios = []
        
        # Scenario A: Entry Level Condos
        for loc in locations:
            scenarios.append({
                'desc': f"Entry Condo ({loc.title()})", 'city': loc, 'prop_type': 'Condo',
                'sfla': 600, 'beds': 1, 'baths': 1, 'year_blt': 1980, 'garage_size': 0, 'acres': 0,
                'grade_numeric': 4, 'cond_numeric': 4, 'scenic_view': 0
            })
            
        # Scenario B: Family Homes
        for loc in locations:
            scenarios.append({
                'desc': f"Family Home ({loc.title()})", 'city': loc, 'prop_type': 'Single Family',
                'sfla': 2200, 'beds': 3, 'baths': 2.5, 'year_blt': 1995, 'garage_size': 400, 'acres': 0.25,
                'grade_numeric': 5, 'cond_numeric': 5, 'scenic_view': 2
            })
            
        # Scenario C: Luxury Builds
        for loc in locations:
            scenarios.append({
                'desc': f"Luxury Build ({loc.title()})", 'city': loc, 'prop_type': 'Single Family',
                'sfla': 4500, 'beds': 5, 'baths': 5, 'year_blt': 2020, 'garage_size': 800, 'acres': 1.0,
                'grade_numeric': 6, 'cond_numeric': 6, 'scenic_view': 4
            })
            
        # Scenario D: Old Cabins
        scenarios.append({
            'desc': "Old Cabin (Breckenridge)", 'city': 'BRECKENRIDGE', 'prop_type': 'Single Family',
            'sfla': 1200, 'beds': 2, 'baths': 1, 'year_blt': 1960, 'garage_size': 0, 'acres': 0.5,
            'grade_numeric': 3, 'cond_numeric': 3, 'scenic_view': 3
        })
        
        df_bench = pd.DataFrame(scenarios)
        
        # Add Macro constants (Use current defaults)
        df_bench['mortgage_rate'] = 6.5
        df_bench['sp500'] = 5000
        df_bench['cpi'] = 310.0
        df_bench['summit_pop'] = 31.0
        
        # Hardcoded Distance Map (Approximate driving miles from town center)
        # Sourced from Google Maps / Local Knowledge
        dist_map = {
            "BRECKENRIDGE":    {'dist_breck': 0.5, 'dist_keystone': 12, 'dist_copper': 16, 'dist_abasin': 18, 'dist_dillon': 10},
            "FRISCO":          {'dist_breck': 9,   'dist_keystone': 9,  'dist_copper': 6,  'dist_abasin': 13, 'dist_dillon': 4},
            "SILVERTHORNE":    {'dist_breck': 12,  'dist_keystone': 7,  'dist_copper': 11, 'dist_abasin': 10, 'dist_dillon': 1},
            "DILLON":          {'dist_breck': 11,  'dist_keystone': 5,  'dist_copper': 12, 'dist_abasin': 9,  'dist_dillon': 0.5},
            "KEYSTONE":        {'dist_breck': 14,  'dist_keystone': 0.5,'dist_copper': 15, 'dist_abasin': 6,  'dist_dillon': 6},
            "COPPER MOUNTAIN": {'dist_breck': 17,  'dist_keystone': 18, 'dist_copper': 0.5,'dist_abasin': 21, 'dist_dillon': 12}
        }
        
        # Apply Distances based on City
        for col in ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin', 'dist_dillon']:
            df_bench[col] = df_bench['city'].apply(lambda c: dist_map.get(c, {'dist_breck':10}).get(col, 10))
            
        # Calculate 'dist_to_lift' (Min of ski resorts)
        lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
        df_bench['dist_to_lift'] = df_bench[lift_cols].min(axis=1)

            
        # Predict GBM
        try:
            df_bench['GBM Estimate'] = gbm_pipeline.predict(df_bench)
        except:
             df_bench['GBM Estimate'] = gbm_pipeline.predict(df_bench)

        # Predict NN
        try:
            # Preprocess
            X_bench_prepped = nn_preprocessor.transform(df_bench)
            X_bench_t = torch.FloatTensor(X_bench_prepped)
            
            nn_model.eval()
            with torch.no_grad():
                y_bench_scaled = nn_model(X_bench_t).numpy()
                y_bench_log = nn_y_scaler.inverse_transform(y_bench_scaled)
                df_bench['NN Estimate'] = np.expm1(y_bench_log).flatten()
        except Exception as e:
            # st.write(f"NN Error: {e}")
            df_bench['NN Estimate'] = 0

        # Formatting
        df_display = df_bench[['desc', 'sfla', 'year_blt', 'GBM Estimate', 'NN Estimate']].copy()
        df_display['Diff %'] = ((df_display['NN Estimate'] - df_display['GBM Estimate']) / df_display['GBM Estimate']) * 100
        
        # Show Table
        st.dataframe(
            df_display.style.format({
                'GBM Estimate': '${:,.0f}',
                'NN Estimate': '${:,.0f}',
                'Diff %': '{:+.1f}%'
            })
        )
        
        # Show Plot
        fig = px.scatter(
            df_display, 
            x='GBM Estimate', 
            y='NN Estimate', 
            hover_data=['desc'],
            title="GBM vs Neural Net (Linear=Match)"
        )
        max_val = max(df_display['GBM Estimate'].max(), df_display['NN Estimate'].max())
        fig.add_shape(
            type='line', line=dict(dash='dash', color='gray'),
            x0=0, x1=max_val, y0=0, y1=max_val
        )
        st.plotly_chart(fig, use_container_width=True)


# --- Main App Skeleton ---
def main():
    st.title("Summit County Housing Analysis")
    st.caption("A Data Science Portfolio Project by Brian")
    st.markdown("---")
    
    # --- Layout Skeleton ---
    # We create containers for all sections upfront so the user sees the full page structure
    # immediately, rather than waiting for Section 3 to finish before seeing Section 4's existence.
    container_1 = st.container()
    container_2 = st.container()
    container_3 = st.container()
    container_4 = st.container()
    container_cta = st.container()
    
    # --- Render Placeholders ---
    # Show "Pending" state for the bottom section which is often blocked by upstream heavy compute
    with container_4:
        pending_4 = st.empty()
        pending_4.markdown("""
        ### 4. Model Status & Benchmarks
        <div style="padding: 20px; border: 1px dashed #ccc; border-radius: 10px; background-color: #fafafa; color: #666;">
            ⏳ <b>Waiting for upstream analysis...</b><br>
            <small>This section will analyze model performance once the feature drivers are computed.</small>
        </div>
        """, unsafe_allow_html=True)
        
    # --- Execute Logic Sequentially ---
    
    with container_1:
        story_section(
            "1. The Data", 
            "What are we looking at?",
            section_1_chart,
            section_1_text
        )
    
    with container_2:
        story_section(
            "2. The Context",
            "Market Forces & Trends",
            section_2_chart,
            section_2_text
        )
    
    with container_3:
        story_section(
            "3. The Drivers",
            "Feature Importance Analysis",
            section_3_chart,
            section_3_text
        )
    
    # Clear the "Pending" message and render real content
    pending_4.empty()
    with container_4:
        section_benchmarks()
    
    with container_cta:
        st.markdown("---")
        # CTA for Inference
        col_cta1, col_cta2 = st.columns([2, 1])
        with col_cta1:
            st.markdown("### 🔮 Ready to predict future prices?")
            st.write("Use the interactive inference engine to simulate property values based on the features discovered above.")
        with col_cta2:
            st.info("👈 **Select 'Predictor' in the sidebar** to start.")

        st.caption("Built with Streamlit, Plotly, and PyDeck.")

if __name__ == "__main__":
    main()


