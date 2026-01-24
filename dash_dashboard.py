import dash
from dash import dcc, html, Input, Output, State
import dash_table
import plotly.express as px
import pandas as pd
import os

# Data loading
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "unified_dataset.csv")
if os.path.exists(DATA_PATH):
    unified_df = pd.read_csv(DATA_PATH)
else:
    unified_df = pd.DataFrame()

# Load Summit County population data
SUMMIT_POP_PATH = os.path.join(os.path.dirname(__file__), "summit_pop.csv")
if os.path.exists(SUMMIT_POP_PATH):
    summit_pop_df = pd.read_csv(SUMMIT_POP_PATH)
    summit_pop_df["year"] = pd.to_numeric(summit_pop_df["year"], errors="coerce")
    summit_pop_df["value"] = pd.to_numeric(summit_pop_df["value"], errors="coerce")
else:
    summit_pop_df = pd.DataFrame(columns=["year", "value"])

# Load contextual growth metrics with Fed interest rate
CONTEXTUAL_PATH = os.path.join(os.path.dirname(__file__), "contextual_growth_metrics.csv")
if os.path.exists(CONTEXTUAL_PATH):
    contextual_df = pd.read_csv(CONTEXTUAL_PATH)
else:
    contextual_df = pd.DataFrame()

# Load Fed rate data
FED_RATE_PATH = os.path.join(os.path.dirname(__file__), "data/fed_rate.csv")
if os.path.exists(FED_RATE_PATH):
    fed_rate_df = pd.read_csv(FED_RATE_PATH)
    fed_rate_df['year'] = pd.to_numeric(fed_rate_df['year'], errors='coerce')
    fed_rate_df['value'] = pd.to_numeric(fed_rate_df['value'], errors='coerce')
    fed_rate_annual = fed_rate_df.groupby('year')['value'].mean().reset_index()
    fed_rate_annual.columns = ['year_blt', 'fed_interest_rate']
else:
    fed_rate_annual = pd.DataFrame(columns=['year_blt', 'fed_interest_rate'])

# Helper functions for filtering
def get_portfolio_sizes_from_category(category, max_size):
    if category == "1 Property": return [1]
    if category == "2 Properties": return [2]
    if category == "3 Properties": return [3]
    if category == "4-5 Properties": return [4, 5]
    if category == "6-10 Properties": return list(range(6, 11))
    if category == "10+ Properties": return list(range(11, max_size + 1))
    return []

# Dash app setup
app = dash.Dash(__name__)
app.title = "Summit County Housing Analysis"

# Initial filter state
initial_filters = {
    "BuildingType": [],
    "LocationType": [],
    "OwnerType": [],
    "SizeCategory": [],
    "Owner Portfolio Size": [],
    "Decade": [],
    "state": []
}

# Layout
app.layout = html.Div([
    dcc.Store(id="filter-store", data=initial_filters.copy()),
    html.Div([
        html.H2("Summit County Housing Analysis"),
        html.Div(id="sidebar-metrics"),
        html.Button("Clear All Filters", id="clear-filters", n_clicks=0),
        html.Button("Clear Location Filter", id="clear-location-filter", n_clicks=0),
        html.Button("Clear Owner Type Filter", id="clear-owner-type-filter", n_clicks=0),
        html.Button("Clear Size Category Filter", id="clear-size-category-filter", n_clicks=0),
        html.Button("Clear Portfolio Size Filter", id="clear-portfolio-size-filter", n_clicks=0),
        html.Button("Clear Decade Filter", id="clear-decade-filter", n_clicks=0),
        html.Button("Clear State Filter", id="clear-state-filter", n_clicks=0),
    ], style={"width": "20%", "display": "inline-block", "verticalAlign": "top"}),
    html.Div([
        html.Div([
            dcc.Graph(id="location-chart"),
            dcc.Graph(id="owner-types-chart"),
        ], style={"display": "flex"}),
        html.Div([
            dcc.Graph(id="portfolio-chart"),
        ], style={"display": "flex"}),
        dcc.Graph(id="owner-location-by-state-chart"),
        dcc.Graph(id="building-type-chart"),
        dcc.Graph(id="timeline-chart"),
        dcc.Graph(id="construction-by-year-chart"),
        dcc.Graph(id="construction-by-year-sizecat-chart"),
        dcc.Graph(id="sfla-per-resident-chart"),
        dcc.Graph(id="commercial-lodging-chart"),
        dcc.Graph(id="size-cat-chart"),
        html.H4("Export Filtered Data"),
        html.Div(id="data-table"),
        html.Button("Download CSV", id="download-csv-btn"),
        dcc.Download(id="download-csv")
    ], style={"width": "78%", "display": "inline-block", "verticalAlign": "top"})
], style={"width": "100%"})

# Callback for all charts and metrics
@app.callback(
    [Output("location-chart", "figure"),
     Output("owner-types-chart", "figure"),
     Output("size-cat-chart", "figure"),
     Output("portfolio-chart", "figure"),
     Output("timeline-chart", "figure"),
     Output("owner-location-by-state-chart", "figure"),
     Output("building-type-chart", "figure"),
     Output("sidebar-metrics", "children"),
     Output("data-table", "children"),
     Output("filter-store", "data"),
     Output("construction-by-year-chart", "figure"),
     Output("sfla-per-resident-chart", "figure"),
     Output("construction-by-year-sizecat-chart", "figure"),
     Output("commercial-lodging-chart", "figure")],
    [Input("location-chart", "clickData"),
     Input("owner-types-chart", "selectedData"),
     Input("owner-types-chart", "clickData"),
     Input("size-cat-chart", "selectedData"),
     Input("size-cat-chart", "clickData"),
     Input("portfolio-chart", "clickData"),
     Input("timeline-chart", "selectedData"),
     Input("timeline-chart", "clickData"),
     Input("owner-location-by-state-chart", "clickData"),
     Input("building-type-chart", "clickData"),
     Input("clear-filters", "n_clicks"),
     Input("clear-location-filter", "n_clicks"),
     Input("clear-owner-type-filter", "n_clicks"),
     Input("clear-size-category-filter", "n_clicks"),
     Input("clear-portfolio-size-filter", "n_clicks"),
     Input("clear-decade-filter", "n_clicks"),
     Input("clear-state-filter", "n_clicks")],
    [State("location-chart", "figure"),
     State("owner-types-chart", "figure"),
     State("size-cat-chart", "figure"),
     State("portfolio-chart", "figure"),
     State("timeline-chart", "figure"),
     State("filter-store", "data")]
)
def update_dashboard(loc_click, ot_sel, ot_click, sc_sel, sc_click, pf_click, tl_sel, tl_click, state_click, bt_click,
                    clear_clicks, clear_loc, clear_owner, clear_size, clear_portfolio, clear_decade, clear_state,
                    loc_fig, ot_fig, sc_fig, pf_fig, tl_fig, filter_store):
    global contextual_df
    # Set default values for clear filter buttons if None
    clear_loc = clear_loc or 0
    clear_owner = clear_owner or 0
    clear_size = clear_size or 0
    clear_portfolio = clear_portfolio or 0
    clear_decade = clear_decade or 0
    clear_state = clear_state or 0
    # Use filter_store for current filters
    filters = filter_store.copy() if filter_store else initial_filters.copy()
    ctx = dash.callback_context
    if ctx.triggered:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "clear-filters":
            filters = initial_filters.copy()
        elif triggered_id == "clear-state-filter":
            filters["state"] = []
        else:
            if triggered_id == "clear-location-filter":
                filters["LocationType"] = []
            elif triggered_id == "clear-owner-type-filter":
                filters["OwnerType"] = []
            elif triggered_id == "clear-size-category-filter":
                filters["SizeCategory"] = []
            elif triggered_id == "clear-portfolio-size-filter":
                filters["Owner Portfolio Size"] = []
            elif triggered_id == "clear-decade-filter":
                filters["Decade"] = []
            else:
                if loc_click and loc_click.get("points"):
                    filters["LocationType"] = [loc_click["points"][0]["label"]]
                if ot_sel and ot_sel.get("points"):
                    filters["OwnerType"] = [p["x"] for p in ot_sel["points"]]
                elif ot_click and ot_click.get("points"):
                    filters["OwnerType"] = [ot_click["points"][0]["x"]]
                if sc_sel and sc_sel.get("points"):
                    filters["SizeCategory"] = [p["x"] for p in sc_sel["points"]]
                elif sc_click and sc_click.get("points"):
                    filters["SizeCategory"] = [sc_click["points"][0]["x"]]
                if pf_click and pf_click.get("points"):
                    label = pf_click["points"][0]["label"]
                    max_size = int(unified_df['Owner Portfolio Size'].max())
                    filters["Owner Portfolio Size"] = get_portfolio_sizes_from_category(label, max_size)
                if tl_sel and tl_sel.get("points"):
                    filters["Decade"] = [p["x"] for p in tl_sel["points"]]
                elif tl_click and tl_click.get("points"):
                    filters["Decade"] = [tl_click["points"][0]["x"]]
                if state_click and state_click.get("points"):
                    filters["state"] = [state_click["points"][0]["x"]]
                if bt_click and bt_click.get("points"):
                    filters["BuildingType"] = [bt_click["points"][0]["x"]]
    # Apply filters
    df = unified_df.copy()
    for k, v in filters.items():
        if v:
            if k == "Owner Portfolio Size":
                df = df[df[k].isin(v)]
            elif k == "Decade" and "year_blt" in df.columns:
                decade_years = [int(str(d).replace('s', '')) for d in v]
                df = df[df['year_blt'].apply(lambda x: ((x // 10) * 10) in decade_years)]
            elif k in df.columns:
                df = df[df[k].isin(v)]
    # Chart figures
    # Property Types Bar Chart
    if 'BuildingType' in df.columns:
        type_counts = df['BuildingType'].value_counts().reset_index()
        type_counts.columns = ['BuildingType', 'Count']
        pt_fig = px.bar(type_counts, x='BuildingType', y='Count', title='Property Types', color='BuildingType', color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        pt_fig = {}
    # Owner Location Pie Chart
    if 'LocationType' in df.columns:
        location_counts = df['LocationType'].value_counts().reset_index()
        location_counts.columns = ['LocationType', 'Count']
        loc_fig = px.pie(location_counts, names='LocationType', values='Count', title='Owner Locations', hole=0.4, color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        loc_fig = {}
    # Owner Types Bar Chart
    if 'OwnerType' in df.columns:
        owner_counts = df['OwnerType'].value_counts().reset_index()
        owner_counts.columns = ['OwnerType', 'Count']
        ot_fig = px.bar(owner_counts, x='OwnerType', y='Count', title='Owner Types', color='OwnerType', color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        ot_fig = {}
    # Size Category Stacked Bar Chart by Building Type
    if 'SizeCategory' in df.columns and 'BuildingType' in df.columns:
        size_order = ["small", "medium", "large", "extra-large"]
        size_cat_bt_counts = df.groupby(['SizeCategory', 'BuildingType']).size().reset_index(name='Count')
        sc_fig = px.bar(
            size_cat_bt_counts,
            x='SizeCategory', y='Count', color='BuildingType', barmode='stack',
            title='Size Categories by Building Type',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        sc_fig.update_xaxes(categoryorder='array', categoryarray=size_order)
    else:
        sc_fig = {}
    # Portfolio Pie Chart
    if 'Owner Portfolio Size' in df.columns:
        portfolio_bins = [0, 1, 2, 3, 5, 10, float('inf')]
        portfolio_labels = ["1 Property", "2 Properties", "3 Properties", "4-5 Properties", "6-10 Properties", "10+ Properties"]
        df_copy = df.copy()
        df_copy['Portfolio Category'] = pd.cut(df_copy['Owner Portfolio Size'], bins=portfolio_bins, labels=portfolio_labels, include_lowest=True)
        portfolio_counts = df_copy['Portfolio Category'].value_counts().reset_index()
        portfolio_counts.columns = ['Portfolio Category', 'Count']
        pf_fig = px.pie(portfolio_counts, names='Portfolio Category', values='Count', title='Owner Portfolio Sizes', hole=0.4, color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        pf_fig = {}
    # Timeline Bar Chart with dual y-axes (Units, Total sfla, Avg sfla/unit)
    import plotly.graph_objects as go
    if 'year_blt' in df.columns and 'sfla' in df.columns:
        timeline_df = df[(df['year_blt'] > 0) & (df['sfla'].notnull()) & (df['sfla'] > 0)].copy()
        if len(timeline_df) > 0:
            timeline_df['Decade'] = (timeline_df['year_blt'] // 10) * 10
            summary = timeline_df.groupby('Decade').agg(
                Count=('year_blt', 'count'),
                Total_sfla=('sfla', 'sum')
            ).reset_index()
            summary['Avg_sfla_per_unit'] = summary['Total_sfla'] / summary['Count']
            summary['Decade'] = summary['Decade'].astype(str) + 's'
            summary = summary.sort_values('Decade')
            # Scale Avg_sfla_per_unit to match the range of Count (units added)
            min_count, max_count = summary['Count'].min(), summary['Count'].max()
            min_avg, max_avg = summary['Avg_sfla_per_unit'].min(), summary['Avg_sfla_per_unit'].max()
            if max_avg > min_avg:
                summary['Scaled_Avg_sfla_per_unit'] = (summary['Avg_sfla_per_unit'] - min_avg) / (max_avg - min_avg) * (max_count - min_count) + min_count
            else:
                summary['Scaled_Avg_sfla_per_unit'] = min_count
            tl_fig = go.Figure()
            tl_fig.add_bar(x=summary['Decade'], y=summary['Total_sfla'], name='Total sfla Added', marker_color=px.colors.qualitative.Plotly[1], yaxis='y2')
            tl_fig.add_trace(go.Scatter(x=summary['Decade'], y=summary['Count'], name='Units Added', mode='lines+markers', marker_color=px.colors.qualitative.Plotly[0], yaxis='y', line=dict(width=3)))
            tl_fig.add_trace(go.Scatter(x=summary['Decade'], y=summary['Avg_sfla_per_unit'], name='Avg sfla per Unit', mode='lines+markers', marker_color=px.colors.qualitative.Plotly[2], yaxis='y', line=dict(width=3)))
            tl_fig.update_layout(
                title='Construction by Decade (Units, Total sfla, Avg sfla/unit)',
                xaxis_title='Decade',
                yaxis=dict(title='Units Added / Avg sfla per Unit'),
                yaxis2=dict(title='Total sfla Added', overlaying='y', side='right', showgrid=False),
                legend=dict(x=0.01, y=0.99),
                bargap=0.2
            )
        else:
            tl_fig = {}
    else:
        tl_fig = {}
    # Owner Location By State Bar Chart
    if 'state' in df.columns:
        state_counts = df['state'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        owner_location_by_state_fig = px.bar(state_counts, x='State', y='Count', title='Owner Location By State', color='State', color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        owner_location_by_state_fig = {}
    # Building Type Bar Chart
    if 'BuildingType' in df.columns and 'sfla' in df.columns:
        building_type_counts = df.groupby('BuildingType').agg(
            Count=('BuildingType', 'count'),
            Avg_sfla=('sfla', 'mean')
        ).reset_index()
        building_type_fig = px.bar(
            building_type_counts.melt(id_vars='BuildingType', value_vars=['Count', 'Avg_sfla']),
            x='BuildingType', y='value', color='variable', barmode='group',
            title='Building Type (Count & Avg sfla per Unit)',
            labels={'value': 'Value', 'variable': 'Metric'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
    else:
        building_type_fig = {}
    # Construction By Year Chart (stacked bar: sfla by year and building type, with Fed interest rate)
    import plotly.graph_objects as go
    if 'year_blt' in df.columns and 'sfla' in df.columns and 'BuildingType' in df.columns:
        construction_df = df[(df['year_blt'] >= 1970) & (df['sfla'] > 0)].copy()
        if not construction_df.empty:
            construction_summary = construction_df.groupby(['year_blt', 'BuildingType'])['sfla'].sum().reset_index()
            pivot_df = construction_summary.pivot(index='year_blt', columns='BuildingType', values='sfla').fillna(0)
            pivot_df = pivot_df.reset_index()  # Ensure 'year_blt' is a column
            # Merge with fed_rate_annual for interest rate
            plot_df = pd.merge(pivot_df, fed_rate_annual, on='year_blt', how='left')
            fig = go.Figure()
            color_palette = px.colors.qualitative.Plotly
            building_types = [bt for bt in construction_summary['BuildingType'].unique() if bt in plot_df.columns]
            for i, bt in enumerate(building_types):
                fig.add_bar(x=plot_df['year_blt'], y=plot_df[bt], name=bt, marker_color=color_palette[i % len(color_palette)])
            # Add Fed interest rate as line on secondary y-axis if available and numeric
            if 'fed_interest_rate' in plot_df.columns:
                fed_rate = pd.to_numeric(plot_df['fed_interest_rate'], errors='coerce')
                fig.add_trace(go.Scatter(
                    x=plot_df['year_blt'], y=fed_rate,
                    name='Fed Interest Rate', mode='lines+markers', marker_color='black', yaxis='y2'))
            fig.update_layout(
                title='Construction By Year (sfla added, segmented by Building Type, with Fed Interest Rate)',
                xaxis_title='Year',
                yaxis=dict(title='sfla Added'),
                yaxis2=dict(title='Fed Interest Rate (%)', overlaying='y', side='right', showgrid=False),
                barmode='stack',
                legend=dict(x=0.01, y=0.99)
            )
            construction_by_year_fig = fig
        else:
            construction_by_year_fig = {}
    else:
        construction_by_year_fig = {}
    # Construction By Year by Size Category Chart (stacked bar: sfla by year and size category)
    if 'year_blt' in df.columns and 'sfla' in df.columns and 'SizeCategory' in df.columns:
        construction_sizecat_df = df[(df['year_blt'] >= 1970) & (df['sfla'] > 0)].copy()
        if not construction_sizecat_df.empty:
            construction_sizecat_summary = construction_sizecat_df.groupby(['year_blt', 'SizeCategory'])['sfla'].sum().reset_index()
            construction_by_year_sizecat_fig = px.bar(
                construction_sizecat_summary,
                x='year_blt', y='sfla', color='SizeCategory', barmode='stack',
                title='Construction By Year (sfla added, segmented by Size Category)',
                labels={'year_blt': 'Year', 'sfla': 'sfla Added'},
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
        else:
            construction_by_year_sizecat_fig = {}
    else:
        construction_by_year_sizecat_fig = {}
    # SFLA per Summit County Resident Over the Years (cumulative sfla / population)
    if 'year_blt' in df.columns and 'sfla' in df.columns:
        sfla_year_df = df[(df['year_blt'] >= 1970) & (df['sfla'] > 0)].copy()
        if not sfla_year_df.empty:
            sfla_by_year = sfla_year_df.groupby('year_blt')['sfla'].sum().reset_index()
            sfla_by_year['cumulative_sfla'] = sfla_by_year['sfla'].cumsum()
            # Merge with summit_pop_df for actual population
            sfla_by_year = pd.merge(sfla_by_year, summit_pop_df[['year', 'value']], left_on='year_blt', right_on='year', how='left')
            sfla_by_year['cumulative_sfla_per_resident'] = sfla_by_year['cumulative_sfla'] / (sfla_by_year['value'] * 1000)
            sfla_per_resident_fig = go.Figure()
            sfla_per_resident_fig.add_trace(go.Scatter(
                x=sfla_by_year['year_blt'], y=sfla_by_year['cumulative_sfla_per_resident'],
                name='Cumulative SFLA per Resident', mode='lines+markers', marker_color=px.colors.qualitative.Plotly[0], yaxis='y'))
            sfla_per_resident_fig.add_trace(go.Scatter(
                x=sfla_by_year['year_blt'], y=sfla_by_year['value'] * 1000,
                name='Summit County Population', mode='lines+markers', marker_color=px.colors.qualitative.Plotly[1], yaxis='y2'))
            sfla_per_resident_fig.update_layout(
                title='Cumulative SFLA per Summit County Resident & Population Over the Years',
                xaxis_title='Year',
                yaxis=dict(title='Cumulative SFLA per Resident'),
                yaxis2=dict(title='Summit County Population', overlaying='y', side='right', showgrid=False),
                legend=dict(x=0.01, y=0.99)
            )
        else:
            sfla_per_resident_fig = {}
    else:
        sfla_per_resident_fig = {}

    # Commercial Lodging SqFt by Year
    if 'year_blt' in df.columns and 'commercial_lodging_sqft' in df.columns:
        commercial_df = df[df['commercial_lodging_sqft'] > 0].copy()
        if not commercial_df.empty:
            commercial_summary = commercial_df.groupby('year_blt')['commercial_lodging_sqft'].sum().reset_index()
            commercial_lodging_fig = px.bar(
                commercial_summary,
                x='year_blt',
                y='commercial_lodging_sqft',
                title='Commercial Lodging SqFt Built Per Year'
            )
        else:
            commercial_lodging_fig = {}
    else:
        commercial_lodging_fig = {}

    # Sidebar metrics and dynamic clear buttons
    metrics = [html.H4("Active Filters")]
    filter_buttons = []
    for k, v in filters.items():
        if v:
            metrics.append(html.Li(f"{k}: {', '.join(map(str, v))}"))
            btn_id = f"clear-{k.replace(' ', '-').lower()}-filter"
            filter_buttons.append(html.Button(f"Clear {k} Filter", id=btn_id, n_clicks=0))
    metrics = [html.H4("Active Filters"), html.Ul([html.Li(f"{k}: {', '.join(map(str, v))}") for k, v in filters.items() if v])] + filter_buttons + [
        html.H4("Properties Shown"), html.P(f"{len(df):,}"),
        html.H4("Total Properties"), html.P(f"{len(unified_df):,}"),
        html.H4("Percentage of Total"), html.P(f"{(len(df)/len(unified_df)*100 if len(unified_df) else 0):.1f}%")
    ]
    # Data table
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.head(100).to_dict("records"),
        page_size=10
    )
    # Return figures and sidebar
    return loc_fig, ot_fig, sc_fig, pf_fig, tl_fig, owner_location_by_state_fig, building_type_fig, metrics, table, filters, construction_by_year_fig, sfla_per_resident_fig, construction_by_year_sizecat_fig, commercial_lodging_fig

# Download callback
@app.callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    [State("location-chart", "clickData"),
     State("owner-types-chart", "selectedData"),
     State("size-cat-chart", "selectedData"),
     State("portfolio-chart", "clickData"),
     State("timeline-chart", "selectedData")]
)
def download_csv(n_clicks, loc_click, ot_sel, sc_sel, pf_click, tl_sel):
    # Filtering logic
    filters = initial_filters.copy()
    if loc_click and loc_click.get("points"):
        filters["LocationType"] = [loc_click["points"][0]["label"]]
    if ot_sel and ot_sel.get("points"):
        filters["OwnerType"] = [p["x"] for p in ot_sel["points"]]
    if sc_sel and sc_sel.get("points"):
        filters["SizeCategory"] = [p["x"] for p in sc_sel["points"]]
    if pf_click and pf_click.get("points"):
        label = pf_click["points"][0]["label"]
        max_size = int(unified_df['Owner Portfolio Size'].max())
        filters["Owner Portfolio Size"] = get_portfolio_sizes_from_category(label, max_size)
    if tl_sel and tl_sel.get("points"):
        filters["Decade"] = [p["x"] for p in tl_sel["points"]]
    # Apply filters
    df = unified_df.copy()
    for k, v in filters.items():
        if v:
            if k == "Owner Portfolio Size":
                df = df[df[k].isin(v)]
            elif k == "Decade" and "year_blt" in df.columns:
                decade_years = [int(str(d).replace('s', '')) for d in v]
                df = df[df['year_blt'].apply(lambda x: ((x // 10) * 10) in decade_years)]
            elif k in df.columns:
                df = df[df[k].isin(v)]
    # Convert to CSV
    csv_string = df.to_csv(index=False, encoding='utf-8')
    return dict(content=csv_string, filename="filtered_properties.csv") if n_clicks else dash.no_update

if __name__ == "__main__":
    app.run(debug=False)