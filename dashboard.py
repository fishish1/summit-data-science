import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Summit County Housing Analysis",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Loading ---
@st.cache_data
def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

# Define the data directory relative to the script's location
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Load unified dataset for cross-filtering
unified_df = load_data(os.path.join(DATA_DIR, "unified_dataset.csv"))

# --- Initialize Session State for Cross-Filtering ---
if 'active_filters' not in st.session_state:
    st.session_state.active_filters = {}

# Use a different key for storing selections from plotly
if 'selection' not in st.session_state:
    st.session_state.selection = {}

if 'filter_update_time' not in st.session_state:
    st.session_state.filter_update_time = datetime.now()

def update_filter(filter_name, values):
    """Update a filter and trigger refresh"""
    st.session_state.active_filters[filter_name] = values
    st.session_state.filter_update_time = datetime.now()

def clear_filters():
    """Clear all active filters"""
    st.session_state.active_filters = {}
    st.session_state.selection = {} # Also clear plotly selections
    st.session_state.filter_update_time = datetime.now()
    # Clear individual chart selections
    for key in st.session_state:
        if key.endswith("_chart"):
            if 'selection' in st.session_state[key]:
                st.session_state[key]['selection'] = None

def apply_filters(df):
    """Apply all active filters to the dataframe"""
    if not st.session_state.active_filters:
        return df

    filtered_df = df.copy()
    for filter_name, values in st.session_state.active_filters.items():
        if not values:
            continue
        # Special case: Decade is derived from year_blt
        if filter_name == 'Decade' and 'year_blt' in filtered_df.columns:
            decade_years = [int(str(d).replace('s', '')) for d in values]
            filtered_df = filtered_df[
                filtered_df['year_blt'].apply(lambda x: ((x // 10) * 10) in decade_years)
            ]
            continue
        # Special case: Owner Portfolio Size uses integer values
        if filter_name == 'Owner Portfolio Size' and 'Owner Portfolio Size' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Owner Portfolio Size'].isin(values)]
            continue
        # Regular column-based filters
        if filter_name in filtered_df.columns:
            if isinstance(values, list):
                filtered_df = filtered_df[filtered_df[filter_name].isin(values)]
            else:
                filtered_df = filtered_df[filtered_df[filter_name] == values]
    return filtered_df

# --- Chart Configuration for Universal Selection Processing ---

def get_portfolio_sizes_from_category(category, max_size):
    """Converts a portfolio category string to a list of integer sizes."""
    if category == "1 Property": return [1]
    if category == "2 Properties": return [2]
    if category == "3 Properties": return [3]
    if category == "4-5 Properties": return [4, 5]
    if category == "6-10 Properties": return list(range(6, 11))
    if category == "10+ Properties": return list(range(11, max_size + 1))
    return []

CHART_CONFIG = {
    "property_types_chart": {
        "filter_name": "BuildingType",
        "value_key": "x",
    },
    "location_chart": {
        "filter_name": "LocationType",
        "value_key": "label",
        "is_pie": True,
    },
    "owner_types_chart": {
        "filter_name": "OwnerType",
        "value_key": "x",
    },
    "size_cat_chart": {
        "filter_name": "SizeCategory",
        "value_key": "x",
    },
    "portfolio_chart": {
        "filter_name": "Owner Portfolio Size",
        "value_key": "label",
        "is_pie": True,
        "value_processor": get_portfolio_sizes_from_category,
    },
    "timeline_chart": {
        "filter_name": "Decade",
        "value_key": "x",
    },
}

# --- Process Chart Selections (early in the script) ---
def process_selections():
    """
    Checks session state for any new chart selections and updates filters accordingly.
    Returns True if a filter was updated, indicating a rerun is needed.
    """
    if unified_df is None:
        return False

    for chart_key, config in CHART_CONFIG.items():
        selection = st.session_state.get(chart_key, {}).get("selection")
        click = st.session_state.get(chart_key, {}).get("click")
        filter_name = config["filter_name"]
        value_key = config.get("value_key", "x")
        is_pie = config.get("is_pie", False)
        value_processor = config.get("value_processor")
        filter_updated = False

        # Handle multi-select (box/lasso select) for non-pie charts
        if not is_pie and selection and selection.get("points"):
            current_filter = st.session_state.active_filters.get(filter_name, [])
            
            if value_processor:
                # Complex case like portfolio sizes
                max_size = int(unified_df['Owner Portfolio Size'].max())
                selected_values = []
                for point in selection["points"]:
                    category = point[value_key]
                    selected_values.extend(value_processor(category, max_size))
                selected_values = sorted(list(set(selected_values)))
            else:
                # Standard case for most charts
                selected_values = sorted([point[value_key] for point in selection["points"]])

            if selected_values and sorted(current_filter) != selected_values:
                update_filter(filter_name, selected_values)
                filter_updated = True

        # Handle single-click toggle (for pie charts)
        if is_pie and click and click.get("points"):
            label = click["points"][0].get(value_key)
            if label:
                current_filter_set = set(st.session_state.active_filters.get(filter_name, []))
                
                if value_processor:
                    max_size = int(unified_df['Owner Portfolio Size'].max())
                    toggle_values = set(value_processor(label, max_size))
                else:
                    toggle_values = {label}

                if toggle_values:
                    if toggle_values.issubset(current_filter_set):
                        new_vals = sorted(list(current_filter_set - toggle_values))
                    else:
                        new_vals = sorted(list(current_filter_set | toggle_values))
                    
                    update_filter(filter_name, new_vals)
                    filter_updated = True
        
        if filter_updated:
            # Clear the event data to prevent re-triggering on the next run
            if click: st.session_state[chart_key]['click'] = None
            if selection: st.session_state[chart_key]['selection'] = None
            return True # Signal that a filter was changed

    return False

# Process selections and rerun if filters were updated
if process_selections():
    st.rerun()

# --- Cross-Filter Controls in Sidebar ---
st.sidebar.header("🔍 Active Filters")

if unified_df is not None:
    # Show active filters
    if st.session_state.active_filters:
        for filter_name, values in st.session_state.active_filters.items():
            if isinstance(values, list):
                values_str = ", ".join(map(str, values))
            else:
                values_str = str(values)
            st.sidebar.write(f"**{filter_name}:** {values_str}")
        
        if st.sidebar.button("🗑️ Clear All Filters"):
            clear_filters()
            st.rerun()
    else:
        st.sidebar.write("No active filters")
    
    # Apply filters to get current dataset
    filtered_df = apply_filters(unified_df)
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Properties Shown", f"{len(filtered_df):,}")
    st.sidebar.metric("Total Properties", f"{len(unified_df):,}")
    
    if len(filtered_df) < len(unified_df):
        pct_shown = (len(filtered_df) / len(unified_df)) * 100
        st.sidebar.metric("Percentage of Total", f"{pct_shown:.1f}%")

else:
    st.sidebar.warning("Unified dataset not found. Cross-filtering unavailable.")
    filtered_df = None

# --- Dashboard UI ---
st.title("Interactive Summit County Housing Analysis")
st.markdown("**Click on any chart element to filter all other charts!** Use the sidebar to see active filters.")

# Add filtering indicator
if filtered_df is not None and len(filtered_df) < len(unified_df):
    st.info(f"📊 Showing {len(filtered_df):,} of {len(unified_df):,} properties based on your selections.")

if unified_df is not None and filtered_df is not None:
    # Create cross-filtering charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Property Type Distribution
        if 'BuildingType' in filtered_df.columns:
            type_counts = filtered_df['BuildingType'].value_counts().reset_index()
            type_counts.columns = ['BuildingType', 'Count']
            
            fig_types = px.bar(
                type_counts.head(10),
                x='BuildingType',
                y='Count',
                title="Property Types (Click to Filter)",
                labels={'BuildingType': 'Property Type', 'Count': 'Number of Properties'}
            )
            fig_types.update_layout(xaxis_tickangle=-45)
            # Bidirectional selection highlight
            selected_vals = st.session_state.active_filters.get('BuildingType', [])
            selected_idx = [i for i, row in type_counts.head(10).iterrows() if row['BuildingType'] in selected_vals] if selected_vals else None
            fig_types.update_traces(selectedpoints=selected_idx)
            fig_types.update_traces(unselected={'marker': {'opacity': 0.3}})
            fig_types.update_layout(clickmode="event+select")
            
            # Display chart and capture selection into session_state
            st.plotly_chart(
                fig_types, 
                use_container_width=True, 
                on_select="rerun",
                key="property_types_chart"
            )
            
    with col2:
        # Owner Location Distribution
        if 'LocationType' in filtered_df.columns:
            location_counts = filtered_df['LocationType'].value_counts().reset_index()
            location_counts.columns = ['LocationType', 'Count']
            
            fig_locations = px.pie(
                location_counts,
                names='LocationType',
                values='Count',
                title="Owner Locations (Click to Filter)",
                hole=.4
            )
            # Highlight selected slices
            selected_vals = st.session_state.active_filters.get('LocationType', [])
            if selected_vals:
                pull_values = [0.1 if loc in selected_vals else 0 for loc in location_counts['LocationType']]
                fig_locations.update_traces(pull=pull_values, marker={'colors': px.colors.qualitative.Plotly})

            fig_locations.update_layout(legend_title_text='Click to Select')

            st.plotly_chart(
                fig_locations, 
                use_container_width=True,
                on_click="rerun",
                key="location_chart"
            )
            
    with col3:
        # Owner Type Distribution
        if 'OwnerType' in filtered_df.columns:
            owner_counts = filtered_df['OwnerType'].value_counts().reset_index()
            owner_counts.columns = ['OwnerType', 'Count']
            
            fig_owners = px.bar(
                owner_counts,
                x='OwnerType',
                y='Count',
                title="Owner Types (Click to Filter)",
                labels={'OwnerType': 'Owner Type', 'Count': 'Number of Properties'}
            )
            fig_owners.update_layout(xaxis_tickangle=-45)
            # Bidirectional selection highlight
            selected_vals = st.session_state.active_filters.get('OwnerType', [])
            selected_idx = [i for i, row in owner_counts.iterrows() if row['OwnerType'] in selected_vals] if selected_vals else None
            fig_owners.update_traces(selectedpoints=selected_idx)
            fig_owners.update_traces(unselected={'marker': {'opacity': 0.3}})
            fig_owners.update_layout(clickmode="event+select")
            
            st.plotly_chart(
                fig_owners, 
                use_container_width=True,
                on_select="rerun",
                key="owner_types_chart"
            )
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Property Size Categories
        if 'SizeCategory' in filtered_df.columns:
            size_counts = filtered_df['SizeCategory'].value_counts().reset_index()
            size_counts.columns = ['SizeCategory', 'Count']
            
            # Define order for size categories
            size_order = ["Small (< 1,000 sqft)", "Medium (1,000-2,000 sqft)", 
                         "Large (2,000-3,000 sqft)", "Very Large (3,000+ sqft)", "Unknown"]
            size_counts['SizeCategory'] = pd.Categorical(
                size_counts['SizeCategory'], categories=size_order, ordered=True
            )
            size_counts = size_counts.sort_values('SizeCategory')
            
            fig_sizes = px.bar(
                size_counts,
                x='SizeCategory',
                y='Count',
                title="Property Sizes by SFLA (Click to Filter)",
                labels={'SizeCategory': 'Size Category', 'Count': 'Number of Properties'}
            )
            fig_sizes.update_layout(xaxis_tickangle=-45)
            # Bidirectional selection highlight
            selected_vals = st.session_state.active_filters.get('SizeCategory', [])
            selected_idx = [i for i, row in size_counts.reset_index(drop=True).iterrows() if row['SizeCategory'] in selected_vals] if selected_vals else None
            fig_sizes.update_traces(selectedpoints=selected_idx)
            fig_sizes.update_traces(unselected={'marker': {'opacity': 0.3}})
            fig_sizes.update_layout(clickmode="event+select")
            
            st.plotly_chart(
                fig_sizes, 
                use_container_width=True,
                on_select="rerun",
                key="size_cat_chart"
            )
            
    with col2:
        # Owner Portfolio Sizes
        if 'Owner Portfolio Size' in filtered_df.columns:
            # Create bins for the integer portfolio sizes
            portfolio_bins = [0, 1, 2, 3, 5, 10, float('inf')]
            portfolio_labels = ["1 Property", "2 Properties", "3 Properties", 
                              "4-5 Properties", "6-10 Properties", "10+ Properties"]
            
            # Create categorical bins from the integer values
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy['Portfolio Category'] = pd.cut(
                filtered_df_copy['Owner Portfolio Size'], 
                bins=portfolio_bins, 
                labels=portfolio_labels, 
                include_lowest=True
            )
            
            ownership_counts = filtered_df_copy['Portfolio Category'].value_counts().reset_index()
            ownership_counts.columns = ['Portfolio Category', 'Count']
            
            # Order the categories properly
            ownership_counts['Portfolio Category'] = pd.Categorical(
                ownership_counts['Portfolio Category'], categories=portfolio_labels, ordered=True
            )
            ownership_counts = ownership_counts.sort_values('Portfolio Category')
            
            fig_ownership = px.pie(
                ownership_counts,
                names='Portfolio Category',
                values='Count',
                title="Owner Portfolio Sizes (Click to Filter)",
                hole=.4
            )
            # Highlight selected slices
            selected_sizes = st.session_state.active_filters.get('Owner Portfolio Size', [])
            if selected_sizes:
                def to_cat(n):
                    if n == 1: return "1 Property"
                    if n == 2: return "2 Properties"
                    if n == 3: return "3 Properties"
                    if n in (4, 5): return "4-5 Properties"
                    if 6 <= n <= 10: return "6-10 Properties"
                    return "10+ Properties"
                selected_cats = set(to_cat(int(n)) for n in selected_sizes)
                pull_values = [0.1 if cat in selected_cats else 0 for cat in ownership_counts['Portfolio Category']]
                fig_ownership.update_traces(pull=pull_values, marker={'colors': px.colors.qualitative.Plotly})

            fig_ownership.update_layout(legend_title_text='Click to Select')
            
            st.plotly_chart(
                fig_ownership, 
                use_container_width=True,
                on_click="rerun",
                key="portfolio_chart"
            )

    # Timeline chart
    if 'year_blt' in filtered_df.columns:
        # Filter out properties with no valid construction year (year_blt = 0)
        timeline_df = filtered_df[filtered_df['year_blt'] > 0].copy()
        
        if len(timeline_df) > 0:
            # Group by decade for better visualization
            timeline_df['Decade'] = (timeline_df['year_blt'] // 10) * 10
            decade_summary = timeline_df.groupby('Decade').agg({
                'sfla': ['count', 'sum']
            }).reset_index()
            
            # Flatten column names
            decade_summary.columns = ['Decade', 'Count', 'Total_SFLA']
            decade_summary['Decade'] = decade_summary['Decade'].astype(str) + 's'
            
            # Create dual-axis chart
            fig_timeline = go.Figure()
            
            # Add bar chart for property count
            fig_timeline.add_trace(go.Bar(
                x=decade_summary['Decade'],
                y=decade_summary['Count'],
                name='Properties Built',
                yaxis='y',
                marker_color='lightblue'
            ))
            
            # Add line chart for total SFLA
            fig_timeline.add_trace(go.Scatter(
                x=decade_summary['Decade'],
                y=decade_summary['Total_SFLA'],
                mode='lines+markers',
                name='Total Sq Ft (SFLA)',
                yaxis='y2',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            # Update layout for dual y-axis
            fig_timeline.update_layout(
                title="Construction by Decade: Properties Built & Square Footage (Click to Filter)",
                xaxis=dict(title='Decade Built'),
                yaxis=dict(
                    title='Number of Properties',
                    side='left'
                ),
                yaxis2=dict(
                    title='Total Square Feet (SFLA)',
                    side='right',
                    overlaying='y'
                ),
                legend=dict(x=0.01, y=0.99),
                hovermode='x unified'
            )
            # Bidirectional selection highlight for bar trace
            selected_decades = st.session_state.active_filters.get('Decade', [])
            selected_idx = [i for i, x in enumerate(decade_summary['Decade']) if x in selected_decades] if selected_decades else None
            fig_timeline.data[0].update(selectedpoints=selected_idx)
            fig_timeline.data[0].update(unselected={'marker': {'opacity': 0.3}})
            fig_timeline.update_layout(clickmode="event+select")
            
            st.plotly_chart(
                fig_timeline, 
                use_container_width=True,
                on_select="rerun",
                key="timeline_chart"
            )
            
            # Show how many properties were excluded from timeline
            excluded_count = len(filtered_df) - len(timeline_df)
            if excluded_count > 0:
                st.caption(f"ℹ️ Timeline excludes {excluded_count:,} properties without valid construction dates")
        else:
            st.info("No properties with valid construction dates found in current selection.")
    
    # Summary metrics row
    st.markdown("---")
    st.subheader("📊 Current Selection Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'sfla' in filtered_df.columns and len(filtered_df) > 0:
            avg_sqft = filtered_df['sfla'].mean()
            st.metric("Average Square Feet", f"{avg_sqft:,.0f}")
    
    with col2:
        if 'PropertyAge' in filtered_df.columns and len(filtered_df) > 0:
            avg_age = filtered_df['PropertyAge'].mean()
            st.metric("Average Property Age", f"{avg_age:.0f} years")
    
    with col3:
        if 'HasGarage' in filtered_df.columns and len(filtered_df) > 0:
            pct_garage = filtered_df['HasGarage'].mean() * 100
            st.metric("Properties with Garage", f"{pct_garage:.1f}%")
    
    with col4:
        if 'OwnerPropertyCount' in filtered_df.columns and len(filtered_df) > 0:
            avg_portfolio = filtered_df['OwnerPropertyCount'].mean()
            st.metric("Avg Properties per Owner", f"{avg_portfolio:.1f}")
    
    # Scatter plot for relationships
    if all(col in filtered_df.columns for col in ['sfla', 'PropertyAge', 'BuildingType']) and len(filtered_df) > 0:
        st.subheader("🔍 Property Relationships")
        
        # Filter out properties with no structure (year_blt = 0) for meaningful age analysis
        scatter_df = filtered_df[filtered_df['year_blt'] > 0].copy()
        
        if len(scatter_df) > 0:
            # Sample data if too large for performance
            plot_df = scatter_df.sample(min(2000, len(scatter_df))) if len(scatter_df) > 2000 else scatter_df
            
            fig_scatter = px.scatter(
                plot_df,
                x='PropertyAge',
                y='sfla',
                color='BuildingType',
                size='OwnerPropertyCount' if 'OwnerPropertyCount' in plot_df.columns else None,
                title="Property Size vs Age (Structures Only - Excludes Vacant Land)",
                labels={"PropertyAge": "Property Age (years)", "sfla": "Square Footage"},
                hover_data=['LocationType', 'OwnerType'] if all(col in plot_df.columns for col in ['LocationType', 'OwnerType']) else None
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Show how many properties were excluded
            excluded_count = len(filtered_df) - len(scatter_df)
            if excluded_count > 0:
                st.caption(f"ℹ️ {excluded_count:,} properties excluded (vacant land without structures)")
        else:
            st.info("No properties with structures found in current selection.")
    
    # Data export section
    st.markdown("---")
    st.subheader("📥 Export Filtered Data")
    
    if len(filtered_df) > 0:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label=f"Download {len(filtered_df):,} properties as CSV",
            data=csv,
            file_name=f"summit_county_filtered_{len(filtered_df)}_properties.csv",
            mime="text/csv"
        )
        
        # Show sample of current data
        with st.expander("Preview Current Data"):
            st.dataframe(filtered_df.head(100), use_container_width=True)
    else:
        st.warning("No properties match current filters.")

else:
    st.error("Unable to load unified dataset. Please run analysis.py first to generate the required data files.")

# Add instructions
st.markdown("---")
st.markdown("""
### 🎯 How to Use Interactive Filtering
1. **Click on any chart element** (bars, pie slices, etc.) to filter all other charts
2. **Multiple selections**: Hold `Shift` while clicking to select multiple items in a chart.
3. **Box/Lasso select**: Use the toolbar on charts to draw selection areas
4. **Clear filters**: Use the button in the sidebar to reset all filters
5. **Export data**: Download the currently filtered dataset as CSV

### 📊 What Charts are Cross-Filtered
- **Property Types** (bar chart)
- **Owner Locations** (donut chart) 
- **Owner Types** (bar chart)
- **Property Sizes** (bar chart)
- **Portfolio Sizes** (donut chart)
- **Construction Decades** (bar chart)
- **Summary Metrics** (update automatically)
- **Relationship Scatter Plot** (updates automatically)
""", unsafe_allow_html=True)
