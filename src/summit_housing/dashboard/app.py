import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from summit_housing.queries import MarketAnalytics, MARKET_TRENDS_SQL, ANALYTICS_SQL, SALES_EVENTS_SQL

st.set_page_config(page_title="Summit Housing SQL Analytics", page_icon="📈", layout="wide")

st.title("🏔️ Summit County Housing: SQL Analytics Portfolio")
st.markdown("""
This dashboard demonstrates **Advanced SQL Analytics** on property sales data.
Instead of pre-calculating metrics in Python, we use **SQL Window Functions**, **CTEs**, and **Moving Averages** 
directly in the database query.
""")

analytics = MarketAnalytics()

# Tabs
# --- Data Scientist Controls ---
with st.sidebar:
    st.header("👩‍🔬 Data Scientist Controls")
    st.caption("Apply constraints to remove outliers and non-market transactions.")
    
    # 1. Arms-Length Filter
    min_price = st.slider(
        "Min Sales Price (Arms-Length)", 
        min_value=0, 
        max_value=100000, 
        value=10000, 
        step=5000,
        help="Filters out transactions likely to be Quit Claim Deeds, Timeshares, or intra-family transfers < $X."
    )
    
    # 2. Statistical Outlier Filter
    filter_iqr = st.checkbox(
        "Filter Statistical Outliers (IQR)", 
        value=True,
        help="Removes points where Appreciation % is > 1.5x the Interquartile Range (Extreme outliers)."
    )

    
    # 3. Metadata
    try:
        import json
        with open("data/metadata.json", "r") as f:
            meta = json.load(f)
            last_updated = meta.get("last_updated", "Unknown")
        st.divider()
        st.caption(f"📅 Data Last Pulled:\n{last_updated}")
    except Exception:
        pass

# --- Common Filters (Sidebar) ---
st.sidebar.divider()
exclude_multiunit = st.sidebar.checkbox(
    "Exclude Multi-Unit Sales (> 1 Unit)",
    value=True,
    help="Removes bulk sales (e.g., apartment complexes) that skew the average price."
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Market Trends", "💰 Investment Analysis", "🏗️ Supply & Owners", "📊 Examine Data", "💰 Value Estimator", "🧬 ML Insights", "🔍 SQL Showcase"])

with tab1:
    st.header("Price Trends & Moving Averages")
    st.markdown("Uses SQL `AVG() OVER (ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)` to smooth volatility.")
    
    # Load Data
    with st.spinner("Running SQL Query..."):
        df_trends = analytics.get_market_trends(exclude_multiunit=exclude_multiunit)
    
    # Filter
    all_cities = df_trends['city'].dropna().unique()
    
    col_filter, col_metric = st.columns([2, 1])
    with col_filter:
        selected_cities = st.multiselect("Filter Cities", all_cities, default=["BRECKENRIDGE", "FRISCO"])
    with col_metric:
        metric_mode = st.radio("Metric", ["Sales Price ($)", "Price Per SqFt ($/sf)"], horizontal=True)

    if selected_cities:
        df_filtered = df_trends[df_trends['city'].isin(selected_cities)]
        
        # Calculate sizing reference for bubbles (target max size ~50px)
        max_sales = df_filtered['sales_count'].max() if not df_filtered.empty else 100
        sizeref = 2.0 * max_sales / (50. ** 2)
        
        fig = go.Figure()
        
        # Determine columns based on selection
        if metric_mode == "Sales Price ($)":
            y_col = 'avg_price'
            ma_col = 'avg_price_3yr_ma'
            title = "Average Sales Price & Volume"
            y_fmt = ",.0f"
        else:
            y_col = 'avg_ppsf'
            ma_col = 'avg_ppsf_3yr_ma'
            title = "Price Per Square Foot (PPSF)"
            y_fmt = ".2f"

        # Load Mortgage Data overlay
        try:
            df_mortgage = pd.read_csv("data/mortgage_rate.csv")
            # Convert YYYY-MM-DD to Year
            df_mortgage['year'] = pd.to_datetime(df_mortgage['date']).dt.year
            # Aggregate to Annual Avg
            df_rates = df_mortgage.groupby('year')['value'].mean().reset_index()
            
            # Add Trace
            fig.add_trace(go.Scatter(
                x=df_rates['year'],
                y=df_rates['value'],
                name="30-Year Mfg Rate",
                line=dict(color='red', width=2, dash='dot'),
                yaxis='y2'
            ))
        except Exception as e:
            st.warning(f"Could not load mortgage data: {e}")

        for city in selected_cities:
            city_data = df_filtered[df_filtered['city'] == city]
            
            # Raw Markers (Bubble)
            fig.add_trace(go.Scatter(
                x=city_data['tx_year'], y=city_data[y_col],
                mode='markers', name=f"{city} (Raw)",
                opacity=0.6,
                customdata=city_data[['sales_count']],
                hovertemplate=f"<b>%{{x}}</b><br>{metric_mode}: $%{{y:{y_fmt}}}<br>Sales Count: %{{customdata[0]}}<extra></extra>",
                marker=dict(
                    size=city_data['sales_count'],
                    sizemode='area',
                    sizeref=sizeref,
                    sizemin=4
                )
            ))
            # Moving Average Line
            fig.add_trace(go.Scatter(
                x=city_data['tx_year'], y=city_data[ma_col],
                mode='lines', name=f"{city} (3yr MA)",
                line=dict(width=3)
            ))
            
        fig.update_layout(
            title=f"{title} (Bubble Size = Sales Volume)", 
            xaxis_title="Year", 
            yaxis_title=metric_mode,
            yaxis2=dict(
                title="Interest Rate (%)",
                overlaying='y',
                side='right',
                range=[0, 20],
                showgrid=False
            ),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Trends by Property Type")
    st.markdown("Segmenting the market using Abstract Codes (`1212`=SFR, `1230`=Condo) via SQL `CASE`.")
    
    with st.spinner("Analyzing Property Types..."):
        df_types = analytics.get_price_by_type()
        
    fig_types = px.line(
        df_types,
        x="tx_year",
        y="avg_price",
        color="prop_type",
        title="Average Price by Property Type",
        markers=True,
        hover_data=["sales_count"]
    )
    st.plotly_chart(fig_types, use_container_width=True)

    # --- Price Band Evolution ---
    st.subheader("Market Composition by Price Tier")
    st.markdown("Is the market bifurcating? This chart groups sales by their price **relative to that year's median**.")
    
    with st.spinner("Analyzing Market Strata..."):
        df_bands = analytics.get_price_band_evolution()
        
    fig_bands = px.area(
        df_bands, 
        x='tx_year', 
        y='sales_count', 
        color='price_band',
        groupnorm='percent', # 100% Stacked Area Chart
        title="Price Band Evolution (Normalized %)",
        labels={'sales_count': 'Share of Sales', 'tx_year': 'Year'},
        category_orders={
            "price_band": [
                '1. Budget (<50% Med)', 
                '2. Below Med (50-100%)', 
                '3. Above Med (100-150%)', 
                '4. Premium (150-200%)', 
                '5. Luxury (>200% Med)'
            ]
        },
        color_discrete_map={
            '1. Budget (<50% Med)': '#2ca02c', # Green
            '2. Below Med (50-100%)': '#7f7f7f', # Gray
            '3. Above Med (100-150%)': '#17becf', # Light Blue
            '4. Premium (150-200%)': '#1f77b4', # Dark Blue
            '5. Luxury (>200% Med)': '#d62728' # Red
        }
    )
    st.plotly_chart(fig_bands, use_container_width=True)
    
    with st.expander("Show Logic (Pandas)"):
        st.code("""
# Calculated in Pandas (SQL does not support Median easily)
yearly_medians = df.groupby('tx_year')['estimated_price'].median()
df['pct_of_median'] = (df['estimated_price'] / df['annual_median']) * 100

def classify_band(pct):
    if pct < 50: return '1. Budget (<50% Med)'
    elif pct <= 100: return '2. Below Med (50-100%)'
    ...
        """, language="python")

    # --- Seasonality Analysis ---
    st.subheader("Seasonal Patterns (Heatmap)")
    
    # Load Data
    df_season = analytics.get_seasonality_stats()
    
    # Toggle
    season_metric = st.radio("Seasonality Metric", ["Sales Volume (Activity)", "Avg PPSF (Valuation)"], horizontal=True)
    
    if season_metric == "Sales Volume (Activity)":
        # Normalize by City (Row-wise)
        # Calculate Total Sales per City
        city_totals = df_season.groupby('city')['sales_count'].transform('sum')
        df_season['pct_of_annual'] = (df_season['sales_count'] / city_totals) * 100.0
        
        z_col = 'pct_of_annual'
        colors = 'Viridis'
        fmt = ".1f"
        check_col = "sales_count" # For custom data
        title = "Market Activity Heatmap: % of Annual Sales by Month"
        hovertemplate = "<b>%{y} - %{x}</b><br>Share of Year: %{z:.1f}%<br>Raw Sales: %{customdata[0]:.0f}<extra></extra>"
    else:
        # For Price, we can normalize too, but raw is often useful. 
        # Let's Normalize to % of Annual Avg for consistency if requested, but user specifically asked for volume help.
        # Let's stick to PPSF raw for now but maybe just make it clear.
        z_col = 'avg_ppsf'
        colors = 'Magma'
        fmt = "$.0f"
        check_col = "sales_count" # Unused really
        title = "Valuation Heatmap: When are prices highest?"
        hovertemplate = "<b>%{y} - %{x}</b><br>Avg PPSF: $%{z:.0f}<extra></extra>"
        
    # Pivot for Heatmap (Cities as Rows, Months as Cols)
    # Using Plotly Express directly for simplicity with pivot logic
    import plotly.express as px
    
    # We want Months sorted correctly, so we need to ensure categorical order
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig_season = px.density_heatmap(
        df_season, 
        x='month_name', 
        y='city', 
        z=z_col,
        histfunc='sum' if z_col == 'pct_of_annual' else 'avg',
        title=title,
        category_orders={'month_name': month_order},
        text_auto=fmt,
        color_continuous_scale=colors
    )
    # Update hover trace
    fig_season.update_traces(
        customdata=df_season[[check_col]] if season_metric == "Sales Volume (Activity)" else None,
        hovertemplate=hovertemplate
    )
    
    fig_season.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig_season, use_container_width=True)
    
    with st.expander("👀 View Underlying SQL"):
        st.code(MARKET_TRENDS_SQL, language="sql")


# ... (Previous code)

with tab2:
    st.header("Flipper Analytics: Appreciation vs. Holding Period")
    st.markdown("Uses SQL `LAG()` to calculate the time between sales and price change percentage.")
    
    with st.spinner("Analyzing Repeat Sales..."):
        df_sales = analytics.get_sales_history()
        # Filter for only repeat sales
        df_repeats = df_sales[df_sales['days_held'].notnull()].copy()
        
        # Renovation Filter
        show_renovated = st.checkbox("Include Renovated Properties?", value=True, help="Excludes homes where Effective Year Built > Actual Year Built")
        
        if not show_renovated:
            df_repeats = df_repeats[df_repeats['is_renovated'] == 0]
            st.info(f"Filtered out {len(df_sales) - len(df_repeats)} renovated properties.")
            
        # Convert days to years
        df_repeats['years_held'] = df_repeats['days_held'] / 365.0
        df_repeats['tx_year'] = df_repeats['tx_year'].astype(int)
        
        # Create 5-Year Buckets
        def get_era(yr):
            start = (yr // 5) * 5
            return f"{start}-{start+4}"
            
        df_repeats['era'] = df_repeats['tx_year'].apply(get_era)
        
        df_repeats['era'] = df_repeats['tx_year'].apply(get_era)
        
    # Apply Filters

    # Apply Filters
    original_count = len(df_repeats)
    
    # Price Filter
    df_clean = df_repeats[ 
        (df_repeats['estimated_price'] >= min_price) & 
        (df_repeats['prev_price'] >= min_price)
    ].copy()
    
    dropped_price = original_count - len(df_clean)
    
    # IQR Filter
    if filter_iqr:
        Q1 = df_clean['growth_pct'].quantile(0.25)
        Q3 = df_clean['growth_pct'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        
        df_clean = df_clean[ (df_clean['growth_pct'] >= lower_bound) & (df_clean['growth_pct'] <= upper_bound) ]
        
    dropped_iqr = original_count - dropped_price - len(df_clean)
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sales Pairs", original_count)
    m2.metric("Filtered (Low Price)", f"-{dropped_price}")
    m3.metric("Filtered (Outliers)", f"-{dropped_iqr}")
    
    # Use cleaned data for plotting
    df_chart = df_clean
        
    # Controls
    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        view_mode = st.radio("Chart visualization:", ["Scatter Points Only", "Trendlines Only", "Combined"], horizontal=True)

    # Base Figure
    fig2 = go.Figure()
    
    # 1. Scatter Points Trace
    if view_mode in ["Scatter Points Only", "Combined"]:
        # We use a trick here: standard scatter doesn't support 'color' arg like px, 
        # so we revert to px and extract traces or just build go.Scatter
        # To maintain the nice 'Viridis' or 'Turbo' scale easily, px is better.
        # Let's use the px figure as base if showing points, or build from scratch.
        
        # Simpler approach: Create px scatter then add/remove traces? 
        # Or just use px for points and add lines.
        
        # We'll stick to PX for points to keep the complex hovering/sizing logic easy
        fig_px = px.scatter(
            df_chart, 
            x="years_held", 
            y="growth_pct",
            color="era", 
            size="estimated_price",
            hover_data=["address", "city", "tx_year"],
            category_orders={"era": sorted(df_chart['era'].unique())},
            opacity=0.5 if view_mode == "Combined" else 0.7,
            labels={
                "years_held": "Holding Period (Years)",
                "growth_pct": "Appreciation (%)",
                "estimated_price": "Sale Price",
                "era": "Sale Era",
                "tx_year": "Sale Year",
                "address": "Property Address",
                "city": "Town"
            }
        )
        # Copy traces to our main figure
        for trace in fig_px.data:
            fig2.add_trace(trace)
            
    # 2. Trendlines Trace
    if view_mode in ["Trendlines Only", "Combined"]:
        eras = sorted(df_chart['era'].unique())
        colors = px.colors.qualitative.Plotly # Cycle colors
        
        for i, era in enumerate(eras):
            df_era = df_chart[df_chart['era'] == era]
            if len(df_era) > 1:
                try:
                    m, b = np.polyfit(df_era['years_held'], df_era['growth_pct'], 1)
                    x_range = np.array([df_era['years_held'].min(), df_era['years_held'].max()])
                    y_range = m * x_range + b
                    
                    fig2.add_trace(go.Scatter(
                        x=x_range, 
                        y=y_range, 
                        mode='lines',
                        name=f"{era} Trend",
                        line=dict(width=4), # Thicker lines for visibility
                        hovertemplate=f"<b>{era} Trend</b><br>Avg Annual Return: {m:.1f}%<br><extra></extra>"
                    ))
                except Exception:
                    pass

    # Layout Updates
    fig2.update_layout(
        title=f"Appreciation by Holding Period ({view_mode})",
        xaxis_title="Years Held",
        yaxis_title="Appreciation %",
        legend_title="Sale Era"
    )
    # fig2.update_yaxes(range=[-50, 500]) # Let Plotly auto-scale to handle outliers
    # fig2.update_xaxes(range=[0, 30])
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("High Velocity Flippers (>20% Profit in <2 Years)")
    df_flip = analytics.get_top_flippers()
    st.dataframe(df_flip[['schno', 'address', 'city', 'days_held', 'growth_pct', 'estimated_price']])
    
    # --- Renovation Premium ---
    st.subheader("Renovation Premium Analysis")
    st.caption("Do renovated homes actually appreciate faster?")
    
    df_reno = analytics.get_renovation_impact()
    
    # Create line chart for Appreciation
    fig_reno = px.line(
        df_reno, 
        x='tx_year', 
        y='avg_appreciation', 
        color='status',
        title="Avg Annual Appreciation: Renovated vs Original Condition",
        labels={'avg_appreciation': 'Avg Appreciation (%)'},
        color_discrete_map={'Renovated': 'green', 'Original Condition': 'gray'}
    )
    st.plotly_chart(fig_reno, use_container_width=True)

    with st.expander("👀 View Underlying SQL"):
        st.code(ANALYTICS_SQL, language="sql")

with tab3:
    st.header("Housing Supply & Owner Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historical Housing Supply")
        st.markdown("Cumulative count of residential units built over time, segmented by type.")
        
        with st.spinner("Calculating Supply..."):
            # Old Method: df_supply = analytics.get_cumulative_supply()
            df_supply = analytics.get_cumulative_supply_by_type()
            
        fig_supply = px.area(
            df_supply, 
            x='year_blt', 
            y='cumulative_units', 
            color='prop_type',
            title="Cumulative Housing Supply by Type",
            labels={'cumulative_units': 'Total Units Built', 'year_blt': 'Year Built'},
            color_discrete_map={
                'Single Family': '#1F77B4',
                'Condo': '#FF7F0E',
                'Townhouse': '#2CA02C',
                'Other': '#7F7F7F'
            }
        )
        # fig_supply.update_layout(title="Cumulative Housing Supply", xaxis_title="Year Built", yaxis_title="Total Units", hovermode="x unified")
        st.plotly_chart(fig_supply, use_container_width=True)
        
        with st.expander("Show SQL Logic"):
            st.code("""
SELECT 
    year_blt, prop_type,
    SUM(units) OVER (PARTITION BY prop_type ORDER BY year_blt ROWS UNBOUNDED PRECEDING) 
FROM raw_records
            """, language="sql")

    with col2:
        st.subheader("Square Footage per Resident")
        st.markdown("Ratio of **Cumulative Residential SFLA** to **Summit County Population** (Census/FRED).")
        
        try:
            # Load Population Data
            df_pop = pd.read_csv("data/summit_pop.csv")
            # Ensure filtering to likely valid years
            df_pop = df_pop[df_pop['year'] >= 1990].sort_values('year')
            
            # Fetch Trend Data specifically for this chart (Need cumulative_sqft)
            df_total_supply = analytics.get_cumulative_supply()
            
            # Merge
            df_total_supply['year'] = df_total_supply['year_blt']
            df_density = pd.merge_asof(
                df_total_supply.sort_values('year'), 
                df_pop.sort_values('year'), 
                on='year', 
                direction='backward'
            )
            
            # FRED Population is in Thousands (e.g., 30.000 = 30k people)
            df_density['sfla_per_capita'] = df_density['cumulative_sqft'] / (df_density['value'] * 1000)
            
            # Filter rows with no population data
            df_density = df_density.dropna(subset=['sfla_per_capita'])
            
            # Plot with Dual Axis
            fig_dens = go.Figure()
            
            # Trace 1: SqFt per Capita (Left Axis)
            fig_dens.add_trace(go.Scatter(
                x=df_density['year'],
                y=df_density['sfla_per_capita'],
                name='SqFt / Resident',
                mode='lines+markers',
                line=dict(color='#FF4B4B', width=3)
            ))
            
            # Trace 2: Population (Right Axis)
            fig_dens.add_trace(go.Scatter(
                x=df_density['year'],
                y=df_density['value'],
                name='Population',
                mode='lines',
                line=dict(color='#1C83E1', dash='dot'),
                yaxis='y2'
            ))
            
            fig_dens.update_layout(
                title="Residential Density vs Population Growth",
                xaxis_title="Year",
                yaxis=dict(title="SqFt per Resident"),
                yaxis2=dict(
                    title="Population",
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                hovermode="x unified",
                legend=dict(x=0, y=1.1, orientation='h')
            )
            st.plotly_chart(fig_dens, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not load population data: {e}")

        st.subheader("Who Owns Summit County?")
        st.caption("We define 'Local' as owners with mailing addresses in Summit County towns.")
        
        col_pie, col_profile = st.columns(2)
        
        with col_pie:
            st.markdown("**Owner Distribution**")
            with st.spinner("Analyzing Owners..."):
                df_owners = analytics.get_owner_location_stats()
                
            fig_owners = px.pie(
                df_owners, 
                values='count', 
                names='location_type', 
                hole=0.4,
                color='location_type',
                color_discrete_map={
                    'Local (In-County)': '#2CA02C',  # Green
                    'In-State (Non-Local)': '#1F77B4', # Blue
                    'Out-of-State': '#FF7F0E' # Orange
                }
            )
            fig_owners.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_owners, use_container_width=True)

        with col_profile:
            st.markdown("**Buyer Profile: Local vs Tourist**")
            df_profile = analytics.get_inventory_profile()
            
            # Melt simply for grouped bar chart or toggle
            # Simplest: Toggle
            metric_view = st.radio("Metric:", ["Avg Property Value ($)", "Avg Property Size (SqFt)"], horizontal=True, label_visibility="collapsed")
            
            if "Value" in metric_view: # Price
                 fig_prof = px.bar(
                     df_profile, x='location_type', y='avg_value', color='location_type',
                     text_auto='$.2s', title="Average Property Value"
                 )
            else: # Size
                 fig_prof = px.bar(
                     df_profile, x='location_type', y='avg_sqft', color='location_type',
                     text_auto='.0f', title="Average Home Size (SqFt)"
                 )
                 
            fig_prof.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_prof, use_container_width=True)
            
            # Insight Text
            try:
                # Safe access
                local_mask = df_profile['location_type'].str.contains("Local")
                oos_mask = df_profile['location_type'].str.contains("Out")
                
                if local_mask.any() and oos_mask.any():
                    local_val = df_profile[local_mask]['avg_sqft'].values[0]
                    oos_val = df_profile[oos_mask]['avg_sqft'].values[0]
                    diff = oos_val - local_val
                    st.info(f"💡 Out-of-State owners buy homes **{abs(diff):.0f} sqft** {'larger' if diff >0 else 'smaller'} than Locals.")
            except Exception:
                pass
        
        with st.expander("Show SQL Logic"):
            st.code("""
CASE 
    WHEN city IN ('BRECKENRIDGE', 'FRISCO', 'DILLON', ...) THEN 'Local (In-County)'
    WHEN state = 'CO' THEN 'In-State (Non-Local)'
    ELSE 'Out-of-State'
END as location_type
            """, language="sql")
            
        st.divider()
        st.divider()
        st.subheader("Retention & Buyer Trends (Cohorts)")
        st.caption("Bar = **Current Owners** (Remaining from that year). Line = **Total Sales** in that year. The gap represents inventory turnover (Sold properties).")
        
        df_trends = analytics.get_owner_purchase_trends()
        
        # Overlay Total Sales Volume to show Retention/Churn
        with st.spinner("Fetching Market Context..."):
            df_market_total = analytics.get_market_trends()
            # Aggregate to just Year + Total Sales
            df_market_agg = df_market_total.groupby('tx_year')['sales_count'].sum().reset_index()
            df_market_agg.columns = ['purchase_year', 'total_sales_volume']
        
        tab_comp, tab_price = st.tabs(["📊 Retention & Composition", "💵 Purchase Price Trends"])
        
        with tab_comp:
            # Combo Chart: Bars (Owners) + Line (Total Sales)
            fig_comp = go.Figure()
            
            # 1. Stacked Bars (Current Owners)
            for loc, color in {'Local (In-County)': '#2CA02C', 'In-State (Non-Local)': '#1F77B4', 'Out-of-State': '#FF7F0E'}.items():
                subset = df_trends[df_trends['location_type'] == loc]
                fig_comp.add_trace(go.Bar(
                    x=subset['purchase_year'],
                    y=subset['buyer_count'],
                    name=loc,
                    marker_color=color
                ))

            # 2. Line (Total Sales)
            fig_comp.add_trace(go.Scatter(
                x=df_market_agg['purchase_year'],
                y=df_market_agg['total_sales_volume'],
                name='Total Market Sales (Original Volume)',
                line=dict(color='gray', width=2, dash='dot'),
                mode='lines'
            ))
            
            fig_comp.update_layout(
                title="Retention Analysis: Who bought vs. Who stayed?",
                xaxis_title="Purchase Year",
                yaxis_title="Count",
                barmode='stack',
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
        with tab_price:
            # Line Chart of Avg Purchase Price
            fig_price_trend = px.line(
                df_trends, 
                x='purchase_year', 
                y='avg_purchase_price', 
                color='location_type', 
                title="Average Purchase Price for Current Owners",
                labels={'avg_purchase_price': 'Avg Price', 'purchase_year': 'Purchase Year'},
                 color_discrete_map={
                    'Local (In-County)': '#2CA02C', 
                    'In-State (Non-Local)': '#1F77B4', 
                    'Out-of-State': '#FF7F0E'
                }
            )
            st.plotly_chart(fig_price_trend, use_container_width=True)

with tab5:
    st.header("💰 Value Estimator (Comparable Sales)")
    st.markdown("Estimates value based on **recent sales** of similar properties (last 18 months). This approach is transparent and grounded in actual market data.")
    
    col_input, col_comps = st.columns([1, 2])
    
    with col_input:
        st.subheader("Subject Property")
        in_city = st.selectbox("Location", ["BRECKENRIDGE", "FRISCO", "DILLON", "SILVERTHORNE", "KEYSTONE", "COPPERMOUNTAIN", "BLUERIVER", "OTHER"], index=0)
        in_type = st.selectbox("Type", ["Single Family", "Condo", "Townhouse"], index=0)
        in_sfla = st.number_input("Square Feet", value=2000, step=100)
        
        st.info("Finds sales within +/- 25% SqFt in the same town.")
        
        if st.button("Estimate Value", type="primary"):
            st.session_state['run_comps'] = True
    
    with col_comps:
        if st.session_state.get('run_comps'):
            with st.spinner("Finding Comps..."):
                try:
                    df_comps = analytics.get_comparable_sales(in_city, in_type, in_sfla)
                    
                    if not df_comps.empty:
                        avg_ppsf = df_comps['ppsf'].mean()
                        est_value = avg_ppsf * in_sfla
                        
                        st.metric("Estimated Market Value", f"${est_value:,.0f}", f"Avg PPSF: ${avg_ppsf:.0f}")
                        
                        st.subheader(f"Found {len(df_comps)} Comparable Sales")
                        
                        # Display Table
                        st.dataframe(
                            df_comps[['tx_date', 'price', 'sfla', 'beds', 'baths', 'year_blt', 'address', 'ppsf']].sort_values('tx_date', ascending=False),
                            column_config={
                                "price": st.column_config.NumberColumn(format="$%d"),
                                "ppsf": st.column_config.NumberColumn(format="$%.2f"),
                            },
                            use_container_width=True,
                            height=300
                        )
                        
                        # Simple Scatter Plot
                        fig_comps = px.scatter(
                            df_comps, x='sfla', y='price', 
                            title="Comps: Price vs Size",
                            hover_data=['address', 'tx_date', 'ppsf'],
                            labels={'sfla': 'Square Feet', 'price': 'Sold Price'},
                            trendline="ols"  # Add simple trendline
                        )
                        # Add Subject Property as a red star
                        fig_comps.add_scatter(
                            x=[in_sfla], y=[est_value], 
                            mode='markers', 
                            marker=dict(size=20, symbol='star', color='red'),
                            name='Subject Property Estimate'
                        )
                        st.plotly_chart(fig_comps, use_container_width=True)
                        
                    else:
                        st.warning("No comparable sales found in the last 18 months matching these criteria. Try adjusting the Square Footage or Property Type.")
                
                except Exception as e:
                    st.error(f"Error fetching comps: {e}")



with tab4:
    st.header("Examine Underlying Data")
    st.markdown("Here is a sample of the processed dataset, joining raw property records with the calculated SQL metrics.")
    
    with st.spinner("Fetching Data Sample..."):
        df_sample = analytics.get_dataset_sample(limit=2000)
        
    st.dataframe(df_sample, use_container_width=True)
    st.caption(f"Showing {len(df_sample)} rows. Data is joined from `raw_records` and the calculated `sales_events` CTE.")

with tab6:
    st.header("🧬 Scientific Feature Selection")
    
    # Shared Imports & Model Logic
    from summit_housing.ml import load_and_prep_data
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # Cache the linear model fit so it doesn't re-run on every interaction
    @st.cache_data
    def get_base_model():
        df_lin = load_and_prep_data()
        # Modified features: Removed 'year', Added 'mortgage_rate'
        lin_features = ['sfla', 'beds', 'baths', 'garage_size', 'year_blt', 'acres', 'mortgage_rate']
        X = df_lin[lin_features].fillna(0)
        y = df_lin['price']
        lr = LinearRegression()
        lr.fit(X, y)
        return dict(zip(lin_features, lr.coef_)), lr.intercept_

    st.subheader("Step 1: Analyze Market Drivers")
    st.markdown("""
    Before building a model, we must understand the "Physics" of the market. 
    Use **Permutation Importance**, **Correlation**, and **PDP** to scientifically validate which variables matter.
    """)
    
    col_run_analysis, col_spacer = st.columns([1, 4])
    with col_run_analysis:
        run_analysis = st.button("Run Feature Analysis", type="primary")

    if run_analysis:
        with st.spinner("Training Random Forest & Calculating Permutations..."):
            from summit_housing.ml import analyze_features, get_pdp_data
            imp_df, corr_df = analyze_features()
            # Also fetch linear coefs for display
            base_coefs_ana, _ = get_base_model()
            
        # Display Headline Insight
        st.success(f"**Insight:** Statistical Analysis suggests a 1% increase in Interest Rates reduces value by **${abs(int(base_coefs_ana['mortgage_rate'])):,.0f}**. This is used as the default in Step 2.")

        t1, t2, t3 = st.tabs(["Feature Importance", "Correlation Matrix", "Partial Dependence (PDP)"])
        
        with t1:
            st.caption("How much does the model error increase if we scramble this variable? (Permutation Importance)")
            fig_imp = px.bar(imp_df, x='importance', y='feature', orientation='h', title="Relative Importance", color='importance', color_continuous_scale='Viridis')
            fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with t2:
            st.caption("Linear relationship between variables (-1 to 1).")
            fig_corr = px.imshow(corr_df, text_auto=".2f", aspect="auto", title="Correlation Heatmap", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with t3:
            st.caption("Shape of the relationship (Marginal Effect).")
            pdp_col1, pdp_col2 = st.columns([1, 3])
            with pdp_col1:
                 pdp_feature = st.selectbox("Select Feature", ['sfla', 'mortgage_rate', 'year_blt', 'summit_pop', 'cpi'])
            with pdp_col2:
                 df_pdp = get_pdp_data(pdp_feature)
                 if not df_pdp.empty:
                     fig_pdp = px.line(df_pdp, x='value', y='average_prediction', markers=True, title=f"Effect of {pdp_feature}")
                     st.plotly_chart(fig_pdp, use_container_width=True)

    st.divider()
    
    st.subheader("Step 2: Build Prediction Model")
    st.markdown("""
    **"White Box" Modeling:**
    Below, we've pre-filled a Linear Model with the **statistically derived weights** from the data.
    You can use the **Presets** to switch property types, or **Override Weights** based on your domain knowledge.
    """)
    
    with st.spinner("Calibrating Statistical Baseline..."):
        base_coefs, base_intercept = get_base_model()
        
        # Calculate Historical CAGR (10-Year)
        try:
            df_hist = analytics.get_market_trends(exclude_multiunit=True)
            df_hist['year'] = pd.to_numeric(df_hist['tx_year'])
            # County-wide weighted average price per year
            county_trend = df_hist.groupby('year').apply(lambda x: np.average(x['avg_price'], weights=x['sales_count']) if x['sales_count'].sum() > 0 else x['avg_price'].mean()).sort_index()
            
            # Use last 10 years or max available
            if len(county_trend) > 1:
                idx_start = max(county_trend.index.min(), county_trend.index.max() - 10)
                idx_end = county_trend.index.max()
                
                # Ensure we have data for start/end
                if idx_start in county_trend.index and idx_end in county_trend.index:
                    start_p = county_trend.loc[idx_start]
                    end_p = county_trend.loc[idx_end]
                    yrs = idx_end - idx_start
                    
                    cagr = (end_p / start_p) ** (1/yrs) - 1 if yrs > 0 else 0.05
                    base_coefs['appreciation_pct'] = cagr * 100.0
                else:
                    base_coefs['appreciation_pct'] = 5.3
            else:
                base_coefs['appreciation_pct'] = 5.2
        except Exception as e:
            # Fallback
            base_coefs['appreciation_pct'] = 5.0

    # PRESETS SECTION
    preset_map = {
        "Custom": {},
        "Breckenridge Luxury SF": {"sfla": 3500, "beds": 4, "baths": 3.5, "garage": 600, "year": 2005, "acres": 0.5},
        "Frisco Condo": {"sfla": 900, "beds": 2, "baths": 2, "garage": 0, "year": 1990, "acres": 0},
        "Silverthorne Townhome": {"sfla": 1600, "beds": 3, "baths": 2.5, "garage": 400, "year": 2000, "acres": 0.05}
    }
    
    selected_preset = st.selectbox("Load Property Preset", list(preset_map.keys()))
    preset = preset_map[selected_preset]

    st.markdown("##### A. Model Coefficients (The 'Brain')")
    with st.expander("⚙️ Fine-Tune Valuation Weights (Advanced)", expanded=False):
        col_w1, col_w2, col_w3 = st.columns(3)
        user_coefs = {}
        with col_w1:
            st.markdown("**Physical**")
            user_coefs['sfla'] = st.number_input("$/SqFt", value=int(base_coefs['sfla']), step=10)
            user_coefs['garage_size'] = st.number_input("$/Garage SqFt", value=int(base_coefs['garage_size']), step=10)
            user_coefs['acres'] = st.number_input("$/Acre", value=int(base_coefs['acres']), step=1000)
        with col_w2:
            st.markdown("**Config**")
            user_coefs['beds'] = st.number_input("$/Bed", value=int(base_coefs['beds']), step=1000)
            user_coefs['baths'] = st.number_input("$/Bath", value=int(base_coefs['baths']), step=1000)
            user_coefs['year_blt'] = st.number_input("$/Year Newer", value=int(base_coefs['year_blt']), step=100)
        with col_w3:
            st.markdown("**Macro**")
            st.caption("Interest rates & Time.")
            user_coefs['mortgage_rate'] = st.number_input("Impact of 1% Rate Increase ($)", value=int(base_coefs['mortgage_rate']), step=5000, help="Typically negative (rates up = prices down).")
            st.markdown("---")
            user_coefs['appreciation_pct'] = st.number_input("Annual Appreciation (%)", value=float(base_coefs.get('appreciation_pct', 5.0)), step=0.5, format="%.2f", help="Compound Annual Growth. Default derived from 10y history.")

    st.markdown("##### B. Scenario Planning")
    
    c_p1, c_p2 = st.columns([1, 2])
    with c_p1:
        st.info("👇 **Adjust these to forecast value**")
        
        # Use Preset values if available, else defaults
        def get_val(key, default):
            return preset.get(key, default) if selected_preset != "Custom" else default
            
        t_sfla = st.number_input("SqFt", value=get_val("sfla", 2000), step=100)
        t_year_blt = st.number_input("Year Built", value=get_val("year", 2000), step=1)
        t_garage = st.number_input("Garage SqFt", value=get_val("garage", 400), step=100)
        
        st.divider()
        t_rate = st.slider("Predicted 30y Interest Rate (%)", 2.5, 12.0, 6.5, 0.1, help="The model relies on this rate to set the price context.")
        
        # Hidden/Less important inputs
        t_beds = preset.get("beds", 3)
        t_baths = preset.get("baths", 2.5)
        t_acres = preset.get("acres", 0.1)
    
    with c_p2:
        def predict_manual(coefs, intercept, **kwargs):
            price = intercept
            for k, v in kwargs.items():
                price += coefs.get(k, 0) * v
            return price
            
        custom_price = predict_manual(
            user_coefs, base_intercept, 
            sfla=t_sfla, garage_size=t_garage, beds=t_beds, baths=t_baths, year_blt=t_year_blt, acres=t_acres, mortgage_rate=t_rate
        )
        
        stats_price = predict_manual(
            base_coefs, base_intercept, 
            sfla=t_sfla, garage_size=t_garage, beds=t_beds, baths=t_baths, year_blt=t_year_blt, acres=t_acres, mortgage_rate=t_rate
        )
        
        # Visualizing the Result
        delta = custom_price - stats_price
        
        st.metric("Predicted Market Value", f"${custom_price:,.0f}", delta=f"${delta:,.0f} vs Baseline" if abs(delta) > 1 else None)
        
        # Sensitivity Chart
        st.caption("Interest Rate Sensitivity Curve")
        
        sim_rates = np.linspace(2.5, 10.5, 40)
        sim_prices = [predict_manual(user_coefs, base_intercept, sfla=t_sfla, garage_size=t_garage, beds=t_beds, baths=t_baths, year_blt=t_year_blt, acres=t_acres, mortgage_rate=r) for r in sim_rates]
        
        fig_sens = px.line(x=sim_rates, y=sim_prices, title=None, labels={'x': 'Mortgage Rate (%)', 'y': 'Estimated Price'}, height=300)
        # Add Current Point
        fig_sens.add_scatter(x=[t_rate], y=[custom_price], mode='markers', marker=dict(size=15, color='red'), name='Your Forecast')
        
        st.plotly_chart(fig_sens, use_container_width=True)
        
        # Forecast Over Time
        st.divider()
        st.caption("Future Value Forecast (10-Year)")
        
        years_hold = 10
        future_years = range(2026, 2026 + years_hold + 1)
        future_vals = [custom_price * ((1 + user_coefs['appreciation_pct']/100) ** (y - 2026)) for y in future_years]
        
        fig_time = px.line(x=future_years, y=future_vals, markers=True, labels={'x': 'Year', 'y': 'Projected Value'})
        st.plotly_chart(fig_time, use_container_width=True, height=250)

    st.caption(f"Base Intercept: ${base_intercept:,.0f}")

with tab7:
    st.header("Raw SQL Playground")
    st.markdown("The full queries used in this analysis.")
    
    st.subheader("1. Data Normalization (The Unpivot CTE)")
    st.markdown("""
    **The Problem:** The raw Assessor Recorders data is "wide" (denormalized). It has 4 sets of columns for sales: `recdate1`/`docfee1` through `recdate4`/`docfee4`. You cannot easily group by "Year" when dates are spread across 4 columns.
    
    **The Solution (CTE):** We use a Common Table Expression with `UNION ALL` to stack these columns into a single "long" stream of events.
    """)
    st.code(SALES_EVENTS_SQL, language="sql")
    
    st.subheader("2. Advanced Metrics (Window Functions)")
    st.markdown("""
    **The Problem:** To calculate "Appreciation", we need to know the *previous* sale price of a property. Standard SQL `GROUP BY` collapses rows, making it impossible to compare a row to its predecessor.
    
    **The Solution (Window Functions):** We use `LAG() OVER (PARTITION BY schno ORDER BY tx_date)`.
    *   `PARTITION BY schno`: ISOLATES the calculation to a single property.
    *   `ORDER BY`: Ensures we look at the chronologically previous sale.
    *   `LAG()`: Fetches the value from that previous row without a complex self-join.
    """)
    st.code(ANALYTICS_SQL, language="sql")
    
    st.subheader("3. Smoothing Volatility (Moving Averages)")
    st.markdown("""
    **The Problem:** Real estate data is noisy. In small towns, one luxury mansion sale can skew the entire year's average (as seen in 1992).
    
    **The Solution:** Instead of a simple average, we calculate a **3-Year Moving Average** directly in SQL.
    *   `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`: Defines a sliding window including this year and the previous 2 years.
    """)
    st.code(MARKET_TRENDS_SQL, language="sql")
