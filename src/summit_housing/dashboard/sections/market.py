import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from scipy.stats import pearsonr
from summit_housing.queries import MarketAnalytics
from summit_housing.geo import RESORT_LIFTS
from summit_housing.dashboard.utils import get_map_dataset, get_analytics_data

def section_2_chart():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price vs Rates", "Buyer Origins", "Seasonality", "Supply Growth", "Lift Proximity"])
    
    # Use cached data fetching
    sec2_trends = get_analytics_data("get_market_trends", exclude_multiunit=True)
    sec2_owners = get_analytics_data("get_owner_purchase_trends")
    sec2_props = get_map_dataset()
    sec2_seasonal = get_analytics_data("get_seasonality_stats")

    with tab5:
        st.markdown("#### 🎿 Ski Resort Proximity & Town Centers")
        st.caption("A simplified 2D view. **Dots** = Properties. **Color** = Distance to Lift (Red=Close, Blue=Far).")
        df_props = sec2_props
        
        if not df_props.empty:
            layer_props = pdk.Layer(
                "ScatterplotLayer",
                df_props,
                get_position='[LONGITUDE, LATITUDE]',
                get_fill_color='[r, g, b, 140]',
                get_radius=80,
                pickable=True,
            )
            
            resort_data = []
            for resort, coords in RESORT_LIFTS.items():
                if resort == 'dist_dillon': continue
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
            
            layer_text = pdk.Layer(
                "TextLayer",
                pd.DataFrame(resort_data),
                get_position='[lon, lat]',
                get_text='name',
                get_color=[0, 0, 0, 255],
                get_size=14,
                get_alignment_baseline="'top'",
                get_text_anchor="'middle'",
                get_background_color=[255, 255, 255, 200],
                get_background_padding=[4, 4]
            )

            view_state = pdk.ViewState(
                latitude=39.55,
                longitude=-106.05,
                zoom=10,
                pitch=0
            )

            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=view_state,
                layers=[layer_props, layer_resorts, layer_text],
                tooltip={"text": "{address}\nDist: {dist_to_lift:.1f} miles"}
            ))
        else:
            st.warning("Could not load property data for map.")
    
    with tab1:
        trends = sec2_trends
        if not trends.empty:
            trends['tx_year'] = pd.to_numeric(trends['tx_year'], errors='coerce')
            all_cities = ["BRECKENRIDGE", "FRISCO", "SILVERTHORNE", "DILLON", "KEYSTONE"]
            df_filtered = trends[trends['city'].isin(all_cities)]
            fig = go.Figure()
            try:
                df_mortgage = pd.read_csv("data/mortgage_rate.csv")
                df_mortgage['year'] = pd.to_datetime(df_mortgage['date']).dt.year
                df_rates = df_mortgage.groupby('year')['value'].mean().reset_index()
                # Filter to match sales data start
                df_rates = df_rates[df_rates['year'] >= 1980]
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
        try:
            df_owners = sec2_owners
            c_view, _ = st.columns([2,1])
            with c_view:
                view_mode = st.radio("Display Mode", ["Volume (Count)", "Market Share (%)"], horizontal=True, label_visibility="collapsed")
            if view_mode == "Market Share (%)":
                groupnorm = 'percent'
                y_title = "Market Share (%)"
                title = "Buyer Origin: Market Share Evolution"
            else:
                groupnorm = None
                y_title = "Number of Buyers"
                title = "Buyer Origin: Volume Trends"

            fig_owners = px.area(
                df_owners, x='purchase_year', y='buyer_count', color='location_type',
                title=title, labels={'buyer_count': y_title, 'purchase_year': 'Year Bought'},
                groupnorm=groupnorm,
                color_discrete_map={'Local (In-County)': '#10b981', 'In-State (Non-Local)': '#3b82f6', 'Out-of-State': '#ef4444'},
                category_orders={'location_type': ['Local (In-County)', 'In-State (Non-Local)', 'Out-of-State']}
            )
            if view_mode == "Market Share (%)":
                fig_owners.update_layout(yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig_owners, use_container_width=True)
            st.markdown("#### 💡 Trend Analysis")
            def get_oos_share(year):
                year_data = df_owners[pd.to_numeric(df_owners['purchase_year']) == year]
                if year_data.empty: return 0
                total = year_data['buyer_count'].sum()
                oos = year_data[year_data['location_type'] == 'Out-of-State']['buyer_count'].sum()
                return (oos / total) * 100 if total > 0 else 0
            share_2000 = get_oos_share(2000)
            share_2021 = get_oos_share(2021)
            st.write(f"**Market Shift:**\n* **2000:** Out-of-State buyers were **{share_2000:.1f}%** of the market.\n* **2021 (Peak):** This reached **{share_2021:.1f}%**.")
            st.info("**Buyer Personas:**\n* 🏔️ **Local Buyers:** Rate Agnostic.\n* 🚙 **In-State Buyers:** Rate Sensitive.\n* ✈️ **Out-of-State Buyers:** The Safety Net.")
            st.subheader("🧪 Hypothesis Tester")
            df_pivot = df_owners.pivot_table(index='purchase_year', columns='location_type', values='buyer_count', aggfunc='sum').fillna(0)
            df_pivot.reset_index(inplace=True)
            df_pivot['year'] = pd.to_numeric(df_pivot['purchase_year'])
            df_pivot['Total Volume'] = df_pivot.sum(axis=1, numeric_only=True) - df_pivot['year']
            df_pivot['Out-of-State Share (%)'] = (df_pivot['Out-of-State'] / df_pivot['Total Volume']) * 100
            if 'In-State (Non-Local)' in df_pivot.columns: df_pivot['In-State Share (%)'] = (df_pivot['In-State (Non-Local)'] / df_pivot['Total Volume']) * 100
            if 'Local (In-County)' in df_pivot.columns: df_pivot['Local Share (%)'] = (df_pivot['Local (In-County)'] / df_pivot['Total Volume']) * 100
            try:
                sp500 = pd.read_csv("data/sp500_annual.csv")
                rates = pd.read_csv("data/mortgage_rate.csv")
                rates['year'] = pd.to_datetime(rates['date']).dt.year
                rates_annual = rates.groupby('year')['value'].mean().reset_index(name='Mortgage Rate (%)')
                rates_annual['Mortgage Rate Change (YoY)'] = rates_annual['Mortgage Rate (%)'].diff()
                df_macro = pd.merge(df_pivot, sp500, on='year', how='inner')
                df_macro = pd.merge(df_macro, rates_annual, on='year', how='inner')
                df_macro.rename(columns={'return_pct': 'S&P 500 Return (%)', 'avg_value': 'S&P 500 Price ($)', 'Total Volume': 'Total Sales Volume'}, inplace=True)
                matrix_cols = ['Out-of-State Share (%)', 'In-State Share (%)', 'Local Share (%)', 'Total Sales Volume', 'S&P 500 Return (%)', 'Mortgage Rate (%)', 'Mortgage Rate Change (YoY)']
                valid_cols = [c for c in matrix_cols if c in df_macro.columns]
                corr_matrix = df_macro[valid_cols].corr()
                fig_matrix = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Buyer Sensitivity Matrix")
                st.plotly_chart(fig_matrix, use_container_width=True)
                c_x, c_y = st.columns(2)
                with c_x: x_axis = st.selectbox("Market Driver (X-Axis)", ['S&P 500 Return (%)', 'S&P 500 Price ($)', 'Mortgage Rate (%)', 'Mortgage Rate Change (YoY)'])
                with c_y: y_axis = st.selectbox("Market Response (Y-Axis)", ['Out-of-State Share (%)', 'In-State Share (%)', 'Local Share (%)', 'Total Sales Volume'])
                fig_scatter = px.scatter(df_macro, x=x_axis, y=y_axis, hover_data=['year'], trendline="ols", title=f"Correlation Plot: {x_axis} vs {y_axis}", color='year')
                st.plotly_chart(fig_scatter, use_container_width=True)
                valid_data = df_macro[[x_axis, y_axis]].dropna()
                if len(valid_data) > 2:
                    corr, p_value = pearsonr(valid_data[x_axis], valid_data[y_axis])
                    strength = "Weak"
                    if abs(corr) > 0.3: strength = "Moderate"
                    if abs(corr) > 0.6: strength = "Strong"
                    sig_label = "✅ Statistically Significant" if p_value < 0.05 else "❌ Not Significant"
                    st.success(f"**Statistical Verdict: {sig_label}**\n* Correlation (r): `{corr:.2f}` ({strength} relationship)\n* P-Value: `{p_value:.4f}`")
            except Exception as e:
                st.warning(f"Could not calc correlations: {e}")
        except Exception as e:
            st.write(f"Could not load owner trends: {e}")

    with tab3:
        st.markdown("#### 🌡️ Seasonal Heatmap: Volume & Price")
        df_seasonal = sec2_seasonal
        if not df_seasonal.empty:
            df_pivot_seasonal = df_seasonal.pivot(index='month_name', columns='city', values='sales_count')
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            df_pivot_seasonal = df_pivot_seasonal.reindex(month_order)
            fig_seasonal = px.imshow(df_pivot_seasonal, labels=dict(x="City", y="Month", color="Sales"), x=df_pivot_seasonal.columns, y=df_pivot_seasonal.index, color_continuous_scale="YlOrRd", title="Sales Volume by Month (Historical Average)")
            st.plotly_chart(fig_seasonal, use_container_width=True)
            st.markdown("**Market Timing:**\n* **The Spring Surge:** Notice the \"Heat\" in **June and July**.\n* **The Winter Lull:** Volume drops in **January**.")
        else:
            st.warning("No seasonality data available.")

    with tab4:
        try:
            df_supply = get_analytics_data("get_cumulative_supply")
            df_pop = pd.read_csv("data/summit_pop.csv")
            df_pop['population'] = df_pop['value'] * 1000
            df_pop['year'] = df_pop['year'].astype(int)
            df_merged = pd.merge(df_supply, df_pop, left_on='year_blt', right_on='year', how='inner')
            df_merged['sqft_per_resident'] = df_merged['cumulative_sqft'] / df_merged['population']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_merged['year'], y=df_merged['cumulative_sqft'], name="Total Residential SqFt", fill='tozeroy', line=dict(color='#cbd5e1', width=0)))
            fig.add_trace(go.Scatter(x=df_merged['year'], y=df_merged['sqft_per_resident'], name="SqFt per Resident", line=dict(color='#8b5cf6', width=3), yaxis='y2'))
            fig.update_layout(title="Supply Growth vs. Population Density", xaxis_title="Year", yaxis=dict(title="Total Built SqFt (Supply)", showgrid=False), yaxis2=dict(title="SqFt per Resident", overlaying='y', side='right', showgrid=False), legend=dict(x=0, y=1.1, orientation='h'), hovermode="x unified", height=450)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**The \"Empty Home\" Paradox:** Density is rising faster than population.")
        except Exception as e:
            st.write(f"Could not load supply/pop comparison: {e}")

def section_2_text():
    st.write("""
    **Understanding the Market Force**
    Summit County is a **"High Beta"** housing market. Unlike a standard suburb where prices are pegged to local wages, this market acts more like a financial asset.
    
    Check the **"Buyer Origins"** tab to see how much external capital from Denver and Out-of-State drives the market.
    """)
