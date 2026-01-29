/**
 * Summit Housing Static Dashboard Core Logic
 * Handles ONNX inference, UI orchestration, and data viz.
 */

const CONFIG = {
    models: {
        gbm: { path: 'models/gbm.onnx', metadata: 'models/gbm_metadata.json' },
        nn: { path: 'models/nn.onnx', metadata: 'models/nn_metadata.json' }
    },
    colors: {
        primary: '#0f1b2c',
        accent: '#3b82f6',
        teal: '#12b3b6',
        oos: '#ef4444',
        local: '#10b981',
        instate: '#3b82f6'
    }
};

// Approximate Summit County Population History (Source: Census/FRED)
const POP_DATA = {
    1960: 2000, 1970: 2700, 1980: 8800, 1990: 12800,
    2000: 23500, 2010: 28000, 2020: 31000, 2024: 31500
};

function getPop(year) {
    const years = Object.keys(POP_DATA).map(Number).sort((a, b) => a - b);
    if (year <= years[0]) return POP_DATA[years[0]];
    if (year >= years[years.length - 1]) return POP_DATA[years[years.length - 1]];
    // Linear interpolation
    for (let i = 0; i < years.length - 1; i++) {
        if (year >= years[i] && year <= years[i + 1]) {
            const range = years[i + 1] - years[i];
            const diff = year - years[i];
            const popRange = POP_DATA[years[i + 1]] - POP_DATA[years[i]];
            return POP_DATA[years[i]] + (diff / range) * popRange;
        }
    }
    return 31000;
}

let state = {
    activePage: 'intro',
    activeModel: 'gbm',
    models: {},
    metadata: {},
    inputs: {
        sfla: 1500, beds: 2, baths: 2, year_blt: 1995, garage_size: 500,
        acres: 0.1, mortgage_rate: 6.5, sp500: 5000, cpi: 310,
        summit_pop: 31, grade_numeric: 4, cond_numeric: 4, scenic_view: 0,
        city: 'BRECKENRIDGE', prop_type: 'Single Family'
    }
};

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    initUI();
    await loadModels();
    updateDashboard();
});

function initUI() {
    // Main Nav logic
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.activePage = btn.dataset.page;
            togglePage(state.activePage);
            window.scrollTo({ top: 0, behavior: 'smooth' });
            updateDashboard();
        });
    });

    // Dynamic Binding for all Predictor inputs
    const bindings = [
        { id: 'sfla', key: 'sfla', type: 'num' },
        { id: 'beds', key: 'beds', type: 'num' },
        { id: 'baths', key: 'baths', type: 'num' },
        { id: 'year', key: 'year_blt', type: 'num' },
        { id: 'city', key: 'city', type: 'val' },
        { id: 'type', key: 'prop_type', type: 'val' },
        { id: 'grade', key: 'grade_numeric', type: 'num' },
        { id: 'view', key: 'scenic_view', type: 'num' },
        { id: 'rate', key: 'mortgage_rate', type: 'num' },
        { id: 'cpi', key: 'cpi', type: 'num' }
    ];

    let debounceTimer;
    bindings.forEach(b => {
        const el = document.getElementById(`input-${b.id}`);
        const valEl = document.getElementById(`val-${b.id}`);

        // If neither exists, skip
        if (!el && !valEl) return;

        const update = (v) => {
            state.inputs[b.key] = b.type === 'num' ? parseFloat(v) : v;
            if (valEl && valEl !== document.activeElement) valEl.value = v;
            if (el && el !== document.activeElement) el.value = v;

            if (state.activePage === 'predictor') {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => updateDashboard(), 300);
            }
        };

        if (el) {
            el.addEventListener('input', (e) => update(e.target.value));
            el.addEventListener('change', (e) => update(e.target.value));
        }

        if (valEl) {
            valEl.addEventListener('input', (e) => update(e.target.value));
            valEl.addEventListener('change', (e) => update(e.target.value));
        }
    });

    document.querySelectorAll('input[name="model-choice"]').forEach(radio => {
        radio.addEventListener('change', (e) => handleModelToggle(e.target.value));
    });

    loadLeaderboard();
}

/**
 * Populates the version selector dropdown based on selected architecture.
 */
async function populateVersionSelector(architecture, selectedRunId = null) {
    const selector = document.getElementById('input-model-version');
    if (!selector) return;

    try {
        // Cache bust the history file to ensure we get new runs
        const resp = await fetch(`data/experiment_history.json?t=${new Date().getTime()}`);
        const history = await resp.json();
        const modelName = architecture === 'nn' ? 'price_net_macro' : 'gbm';
        const versions = history.filter(h => h.model_name === modelName).sort((a, b) => b.run_id - a.run_id);

        selector.innerHTML = versions.map(v =>
            `<option value="${v.run_id}" ${v.run_id == selectedRunId ? 'selected' : ''}>
                Run #${v.run_id} (MAE: $${Math.round(v.metrics.mae).toLocaleString()})
            </option>`
        ).join('');

        selector.onchange = async (e) => {
            const runId = parseInt(e.target.value);
            const run = history.find(h => h.run_id === runId);
            if (run) selectModelVersion(run);
        };
    } catch (e) {
        console.error('Failed to populate version selector:', e);
    }
}

/**
 * Handles architecture toggle from the radio buttons.
 * Selects the best performing version of that architecture.
 */
async function handleModelToggle(architecture) {
    const resp = await fetch('data/experiment_history.json');
    const history = await resp.json();

    // Filter history for the selected architecture
    const modelName = architecture === 'nn' ? 'price_net_macro' : 'gbm';
    const versions = history.filter(h => h.model_name === modelName);

    if (versions.length > 0) {
        // Find best version
        const bestVersion = versions.reduce((prev, curr) => (prev.metrics.mae < curr.metrics.mae) ? prev : curr);
        await selectModelVersion(bestVersion);
    } else {
        // Fallback for default if no history (unlikely given our data)
        state.activeModel = architecture;
        document.getElementById('active-model-name').textContent = architecture === 'gbm' ? 'Gradient Boosting' : 'Neural Network';
        if (state.activePage === 'predictor') updateDashboard();
    }
}

/**
 * Sets the active model based on a run from the history
 * @param {Object} run The run object from experiment_history.json
 */
async function selectModelVersion(run) {
    state.activeModel = run.model_name === 'price_net_macro' ? 'nn' : 'gbm';
    state.activeRun = run;

    // Update UI
    const modelRadios = document.querySelectorAll('input[name="model-choice"]');
    modelRadios.forEach(r => {
        r.checked = r.value === state.activeModel;
    });

    document.getElementById('active-model-name').textContent = run.model_name === 'price_net_macro' ? `Neural Net (#${run.run_id})` : `Gradient Boosting (#${run.run_id})`;
    document.getElementById('model-mae').textContent = `$${Math.round(run.metrics.mae).toLocaleString()}`;

    // Update version selector
    const selector = document.getElementById('input-model-version');
    if (selector) {
        // If the selector is currently for a different architecture, refresh it
        const currentModelName = run.model_name === 'price_net_macro' ? 'nn' : 'gbm';
        // We checking if we need to swap the options
        if (selector.options.length === 0 || !Array.from(selector.options).some(o => o.value == run.run_id)) {
            await populateVersionSelector(currentModelName, run.run_id);
        } else {
            selector.value = run.run_id;
        }
    }

    // Highlight the row in the leaderboard
    document.querySelectorAll('#leaderboard tr').forEach(tr => tr.classList.remove('selected-run'));
    const row = document.querySelector(`#leaderboard tr[data-run-id="${run.run_id}"]`);
    if (row) row.classList.add('selected-run');

    console.log(`Model switched to: ${run.model_name} (Run #${run.run_id})`);
    if (state.activePage === 'predictor') updateDashboard();
}

function togglePage(page) {
    document.querySelectorAll('.page-section').forEach(section => section.classList.add('hidden'));
    document.getElementById(`${page}-page`).classList.remove('hidden');
}

async function updateDashboard() {
    if (state.activePage === 'predictor') {
        renderPredictor();
    } else if (state.activePage === 'story') {
        initStoryMap();
        renderAllStoryCharts();
    } else if (state.activePage === 'experiments') {
        // If a model is already selected, show its detail view
        if (state.activeRun) {
            await renderModelDetail(state.activeRun);
        }
    }
}

// --- Story Rendering ---

let map = null;

function initStoryMap() {
    if (map) return;
    map = L.map('map-hotspots').setView([39.8, -98.5], 4); // Start at US view
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png').addTo(map);

    // Load County Boundary
    fetch('data/summit_boundary.geojson')
        .then(response => response.json())
        .then(data => {
            L.geoJSON(data, {
                style: {
                    color: "#4a5568",
                    weight: 2,
                    opacity: 0.8,
                    fillOpacity: 0.05
                }
            }).addTo(map);
        })
        .catch(err => console.error("Failed to load boundary:", err));

    // Legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function (map) {
        const div = L.DomUtil.create('div', 'info legend');
        div.style.backgroundColor = 'white';
        div.style.padding = '10px';
        div.style.borderRadius = '5px';
        div.style.boxShadow = '0 0 15px rgba(0,0,0,0.2)';

        div.innerHTML += '<i style="background:red; width: 10px; height: 10px; display: inline-block; margin-right: 5px; border-radius: 50%;"></i> Ski Resort<br>';
        div.innerHTML += '<i style="background:blue; width: 10px; height: 10px; display: inline-block; margin-right: 5px; border-radius: 50%;"></i> Town Center<br>';
        div.innerHTML += '<i style="background:cyan; width: 10px; height: 10px; display: inline-block; margin-right: 5px; border-radius: 50%;"></i> Lake/Feature<br>';
        div.innerHTML += '<i style="background:gold; width: 10px; height: 10px; display: inline-block; margin-right: 5px; border-radius: 50%;"></i> School';
        return div;
    };
    legend.addTo(map);

    // Animate to Summit County
    setTimeout(() => {
        map.flyTo([39.55, -106.05], 10, {
            duration: 3 // seconds
        });

        // Add markers only after arriving to avoid "giant dots" during zoom
        map.once('moveend', () => {
            const features = [
                // Ski Resorts (Red)
                { name: "Breckenridge Ski Resort", lat: 39.4805, lon: -106.0666, color: "red", type: "Resort" },
                { name: "Keystone Resort", lat: 39.605, lon: -105.9439, color: "red", type: "Resort" },
                { name: "Copper Mountain", lat: 39.5022, lon: -106.1506, color: "red", type: "Resort" },
                { name: "A-Basin", lat: 39.6425, lon: -105.8719, "type": "Resort", color: "red" },

                // Town Centers (Blue)
                { name: "Main St, Breck", lat: 39.4817, lon: -106.0384, color: "blue", type: "Town" },
                { name: "Main St, Frisco", lat: 39.5744, lon: -106.0975, color: "blue", type: "Town" },
                { name: "Silverthorne Outlets", lat: 39.6296, lon: -106.0713, color: "blue", type: "Town" },

                // Key Locations (Gold/Cyan)
                { name: "Dillon Reservoir", lat: 39.615, lon: -106.05, color: "cyan", type: "Feature" },
                { name: "Summit High School", lat: 39.553, lon: -106.062, color: "gold", type: "School" }
            ];

            features.forEach(f => {
                L.circleMarker([f.lat, f.lon], {
                    radius: f.type === 'Resort' ? 12 : 8,
                    color: f.color,
                    fillOpacity: 0.8
                }).addTo(map).bindPopup(`<b>${f.name}</b><br>${f.type}`);
            });
        });

    }, 1000);
}

async function renderAllStoryCharts() {
    // Render all charts at once since there are no tabs
    renderOverviewData();
    renderMarketHistory();
    renderCorrelationMatrix();
}

async function renderOverviewData() {
    try {
        const resp = await fetch('data/records_sample_curated.json');
        const data = await resp.json();

        // Metrics Banner
        const latest = data[0]; // Simplified proxy
        const banner = document.getElementById('market-overview-metrics');
        banner.innerHTML = `
            <div class="metric-item"><div class="metric-value">50,000+</div><div class="metric-label">Parcel Records</div></div>
            <div class="metric-item"><div class="metric-value">20+ Years</div><div class="metric-label">Sales History</div></div>
            <div class="metric-item"><div class="metric-value">100+</div><div class="metric-label">Engineered Features</div></div>
        `;


        // Distance Map
        renderDistanceAnalysis();

        // Raw Sample Table
        const container = document.getElementById('raw-sample-container');
        const toggle = document.getElementById('toggle-all-cols');

        const renderTable = (showAll) => {
            // Basic columns vs All
            const displayCols = showAll ? Object.keys(data[0]) : ['address', 'city', 'year_blt', 'sfla', 'totactval'];

            // Wrap in scroll container for sticky headers (handled by outer container now)
            let html = '<table class="data-table"><thead><tr>';
            displayCols.forEach(c => html += `<th>${c}</th>`);
            html += '</tr></thead><tbody>';

            data.slice(0, 50).forEach(r => { // Show more rows now that it scrolls
                html += '<tr>';
                displayCols.forEach(c => {
                    let val = r[c];
                    if (typeof val === 'number' && (c.includes('val') || c.includes('price'))) val = `$${val.toLocaleString()}`;
                    html += `<td>${val}</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody></table>';
            container.innerHTML = html;
        };

        renderTable(false);
        if (toggle) {
            toggle.onchange = (e) => renderTable(e.target.checked);
        }

    } catch (e) { console.error('Overview Error:', e); }
}

async function renderDistanceAnalysis() {
    try {
        const resp = await fetch('data/geo_distances.json');
        const data = await resp.json();

        // Load Ski Lift Lines for Plotly
        const liftsResp = await fetch('data/ski_lifts.geojson');
        const liftGeoJSON = await liftsResp.json();

        // We'll use a Scatter Mapbox or just Scatter on Plotly
        const trace = {
            type: 'scattermapbox',
            lat: data.map(d => d.lat),
            lon: data.map(d => d.lon),
            mode: 'markers',
            marker: {
                size: 6,
                color: data.map(d => d.dist_to_lift),
                colorscale: 'RdBu', // Red (close) to Blue (far)
                reversescale: true,
                cmin: 0,
                cmax: 10, // Cap visual scale at 10 miles
                opacity: 0.6,
                colorbar: { title: 'Miles to Lift' }
            },
            text: data.map(d => `${d.address}<br>$${d.price?.toLocaleString()}<br>${d.dist_to_lift?.toFixed(1)} mi`),
            hoverinfo: 'text'
        };

        const layout = {
            mapbox: {
                style: "carto-positron",
                center: { lat: 39.55, lon: -106.05 },
                zoom: 9.5,
                layers: [
                    {
                        source: liftGeoJSON,
                        type: "line",
                        color: "red",
                        line: { width: 2 }
                    }
                ]
            },
            margin: { t: 0, b: 0, l: 0, r: 0 },
            showlegend: false
        };

        Plotly.newPlot('distance-map', [trace], layout);
    } catch (e) { console.error("Distance Map Error", e); }
}

async function renderMarketHistory() {
    try {
        // 1. Price vs Rates
        const trends = await (await fetch('data/market_trends.json')).json();
        const cities = ["BRECKENRIDGE", "FRISCO", "SILVERTHORNE", "DILLON", "KEYSTONE"];
        const traces = cities.map(city => {
            const cityData = trends.filter(d => d.city === city);
            return { x: cityData.map(d => d.tx_year), y: cityData.map(d => d.avg_price_3yr_ma), name: city, type: 'scatter' };
        });
        Plotly.newPlot('chart-market-trends', traces, { height: 400, margin: { t: 20 } });

        // 2. Buyer Origins (Market share) with percentage toggle
        const owners = await (await fetch('data/owner_trends.json')).json();
        const types = ['Local (In-County)', 'In-State (Non-Local)', 'Out-of-State'];

        const renderBuyerOrigins = (showPercentage) => {
            if (showPercentage) {
                // Calculate percentages per year
                const yearTotals = {};
                owners.forEach(d => {
                    if (!yearTotals[d.purchase_year]) yearTotals[d.purchase_year] = 0;
                    yearTotals[d.purchase_year] += d.buyer_count;
                });

                const ownerTraces = types.map(t => {
                    const tData = owners.filter(d => d.location_type === t);
                    return {
                        x: tData.map(d => d.purchase_year),
                        y: tData.map(d => (d.buyer_count / yearTotals[d.purchase_year] * 100).toFixed(1)),
                        name: t,
                        stackgroup: 'one',
                        type: 'scatter',
                        hovertemplate: '%{y:.1f}%<extra></extra>'
                    };
                });

                Plotly.newPlot('chart-buyer-origins', ownerTraces, {
                    height: 400,
                    margin: { t: 20 },
                    yaxis: { title: 'Percentage (%)', ticksuffix: '%' }
                });
            } else {
                // Show absolute counts
                const ownerTraces = types.map(t => {
                    const tData = owners.filter(d => d.location_type === t);
                    return {
                        x: tData.map(d => d.purchase_year),
                        y: tData.map(d => d.buyer_count),
                        name: t,
                        stackgroup: 'one',
                        type: 'scatter'
                    };
                });

                Plotly.newPlot('chart-buyer-origins', ownerTraces, {
                    height: 400,
                    margin: { t: 20 },
                    yaxis: { title: 'Number of Buyers' }
                });
            }
        };

        // Initial render
        renderBuyerOrigins(false);

        // Set up toggle
        const buyerToggle = document.getElementById('toggle-buyer-percentage');
        if (buyerToggle) {
            buyerToggle.onchange = (e) => renderBuyerOrigins(e.target.checked);
        }

        // 3. Seasonality Heatmap
        const seasonal = await (await fetch('data/seasonality.json')).json();
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const uniqueCities = [...new Set(seasonal.map(d => d.city))];
        const zData = months.map(m => uniqueCities.map(c => {
            const d = seasonal.find(x => x.month_name === m && x.city === c);
            return d ? d.sales_count : 0;
        }));
        Plotly.newPlot('chart-seasonality', [{ z: zData, x: uniqueCities, y: months, type: 'heatmap', colorscale: 'YlOrRd' }], { height: 400, margin: { t: 20 } });

        // 4. Supply Growth
        // 4. Supply Growth vs Density
        const supply = await (await fetch('data/supply_growth.json')).json();

        // Calculate SqFt per Person
        const densityData = supply.map(d => {
            const pop = getPop(d.year_blt);
            return {
                year: d.year_blt,
                sqft: d.cumulative_sqft,
                pop: pop,
                per_capita: d.cumulative_sqft / pop
            };
        }).filter(d => d.year >= 1960); // Focus on relevant history

        Plotly.newPlot('chart-supply', [
            {
                x: densityData.map(d => d.year),
                y: densityData.map(d => d.sqft),
                name: 'Total SqFt Supply',
                type: 'scatter',
                fill: 'tozeroy',
                line: { color: CONFIG.colors.primary }
            },
            {
                x: densityData.map(d => d.year),
                y: densityData.map(d => d.per_capita),
                name: 'SqFt per Person',
                type: 'scatter',
                mode: 'lines',
                line: { color: CONFIG.colors.teal, width: 3 },
                yaxis: 'y2'
            },
            {
                x: densityData.map(d => d.year),
                y: densityData.map(d => d.pop),
                name: 'Population',
                type: 'scatter',
                mode: 'lines',
                line: { color: CONFIG.colors.oos, width: 2, dash: 'dot' },
                yaxis: 'y3'
            }
        ], {
            height: 400,
            margin: { t: 40, r: 50, l: 60 },
            yaxis: { title: 'Total Residential SqFt' },
            yaxis2: {
                title: 'SqFt / Capita',
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                titlefont: { color: CONFIG.colors.teal },
                tickfont: { color: CONFIG.colors.teal }
            },
            yaxis3: {
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                visible: false
            },
            legend: { x: 0, y: 1.1, orientation: 'h' },
            font: { family: 'Space Grotesk' }
        }, { responsive: true });

    } catch (e) { console.error(e); }
}



async function renderCorrelationMatrix() {
    try {
        const corrResp = await fetch('data/correlation_matrix.json');
        const data = await corrResp.json();

        const labels = data.map(d => d.index);
        const zData = labels.map(rowLabel => labels.map(colLabel => data.find(d => d.index === rowLabel)[colLabel] || 0));

        Plotly.newPlot('chart-correlations', [{
            z: zData, x: labels, y: labels,
            type: 'heatmap', colorscale: 'RdBu', zmin: -1, zmax: 1
        }], { height: 500, margin: { t: 20 }, font: { family: 'Space Grotesk' } });
    } catch (e) {
        console.error('Failed to load correlation matrix:', e);
    }
}

/**
 * Renders model-specific SHAP and PDP visualizations in the experiments page
 * @param {Object} run The selected model run from experiment history
 */
async function renderModelDetail(run) {
    const detailView = document.getElementById('model-detail-view');
    if (!detailView) return;

    // Show the detail view
    detailView.classList.remove('hidden');

    // Update title
    const title = document.getElementById('model-detail-title');
    if (title) {
        title.textContent = `Model Interpretability: ${run.model_name} (Run #${run.run_id})`;
    }

    try {
        // 1. SHAP Bar Chart
        let shapData = [];

        // Use run-specific SHAP summary if available
        if (run.shap_summary && Array.isArray(run.shap_summary) && run.shap_summary.length > 0) {
            shapData = run.shap_summary;
        } else {
            // Fallback to global summary (latest model)
            const shapResp = await fetch('data/shap_summary.json');
            shapData = await shapResp.json();
        }

        const shapTrace = {
            y: shapData.map(d => d.feature),
            x: shapData.map(d => d.importance),
            type: 'bar',
            orientation: 'h',
            marker: { color: CONFIG.colors.primary }
        };

        Plotly.newPlot('chart-shap-experiments', [shapTrace], {
            title: `Feature Importance (SHAP): ${run.model_name} (Run #${run.run_id})`,
            xaxis: { title: 'Mean |SHAP value|' },
            yaxis: { automargin: true },
            margin: { l: 150 },
            font: { family: 'Space Grotesk' }
        });

        // 2. PDP Lines
        const pdpResp = await fetch('data/pdp_data.json');
        const pdpData = await pdpResp.json();

        // Helper function to format feature names for display
        const formatFeatureName = (feat) => {
            const nameMap = {
                'sfla': 'Square Footage',
                'year_blt': 'Year Built',
                'beds': 'Bedrooms',
                'baths': 'Bathrooms',
                'garage_size': 'Garage Size',
                'acres': 'Lot Size (Acres)',
                'mortgage_rate': 'Mortgage Rate',
                'cpi': 'CPI',
                'sp500': 'S&P 500',
                'summit_pop': 'Summit Population',
                'grade_numeric': 'Property Grade',
                'cond_numeric': 'Property Condition',
                'scenic_view': 'Scenic View',
                'dist_to_lift': 'Distance to Nearest Lift',
                'dist_breck': 'Distance to Breckenridge',
                'dist_keystone': 'Distance to Keystone',
                'dist_copper': 'Distance to Copper Mountain',
                'dist_abasin': 'Distance to A-Basin',
                'dist_dillon': 'Distance to Dillon'
            };
            return nameMap[feat] || feat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        };

        const renderPDP = (feature) => {
            const subset = pdpData.filter(d => d.feature === feature);
            if (!subset.length) return;

            const pdpTrace = {
                x: subset.map(d => d.value),
                y: subset.map(d => d.average_prediction),
                type: 'scatter',
                mode: 'lines',
                line: { color: CONFIG.colors.teal, width: 3 },
                fill: 'tozeroy'
            };

            Plotly.newPlot('chart-pdp-experiments', [pdpTrace], {
                title: `Marginal Effect: ${formatFeatureName(feature)}`,
                yaxis: { title: 'Predicted Log Price' },
                xaxis: { title: formatFeatureName(feature) },
                font: { family: 'Space Grotesk' }
            });
        };

        const pdpSelect = document.getElementById('pdp-feature-select-experiments');
        if (pdpSelect) {
            // Get unique features from the data and populate dropdown
            const availableFeatures = [...new Set(pdpData.map(d => d.feature))].sort();

            // Clear and populate dropdown
            pdpSelect.innerHTML = availableFeatures.map(feat =>
                `<option value="${feat}">${formatFeatureName(feat)}</option>`
            ).join('');

            // Set up event handler and initial render
            pdpSelect.onchange = (e) => renderPDP(e.target.value);
            renderPDP(pdpSelect.value); // Initial render with first feature
        }
    } catch (e) {
        console.error('Failed to load SHAP/PDP data:', e);
    }
}

// --- Inference Engine ---

async function renderPredictor() {
    const priceEl = document.getElementById('estimated-price');
    const lastUpdatedEl = document.getElementById('last-updated');

    try {
        const price = await runInference();

        if (priceEl) {
            priceEl.textContent = new Intl.NumberFormat('en-US', {
                style: 'currency', currency: 'USD', maximumFractionDigits: 0
            }).format(price);

            // Trigger pulse animation
            priceEl.classList.remove('pulse-update');
            void priceEl.offsetWidth; // Force reflow
            priceEl.classList.add('pulse-update');
        }

        if (lastUpdatedEl) {
            lastUpdatedEl.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
        }

        updatePredictorCharts(price);
    } catch (e) {
        console.error('Inference failed:', e);
    }
}

async function updatePredictorCharts(currentPrice) {
    // 1. Sensitivity: Square Footage
    const sqftRange = Array.from({ length: 12 }, (_, i) => state.inputs.sfla * (0.4 + i * 0.1));
    const sqftPrices = [];
    for (const s of sqftRange) {
        sqftPrices.push(await runInference({ ...state.inputs, sfla: s }));
    }

    Plotly.newPlot('chart-sqft', [{
        x: sqftRange, y: sqftPrices, type: 'scatter', mode: 'lines+markers', line: { color: CONFIG.colors.accent, width: 3 }
    }, {
        x: [state.inputs.sfla], y: [currentPrice], mode: 'markers', marker: { color: 'red', size: 12 }
    }], { margin: { t: 10, r: 10, l: 50, b: 40 }, xaxis: { title: 'SqFt' }, yaxis: { title: 'Price ($)' }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Space Grotesk' }, showlegend: false }, { responsive: true });

    // 2. Sensitivity: Mortgage Rate
    // Covering 0% to 20% as requested for "wild times"
    const rateRange = [0, 2, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20];
    const ratePrices = [];
    for (const r of rateRange) {
        ratePrices.push(await runInference({ ...state.inputs, mortgage_rate: r }));
    }

    Plotly.newPlot('chart-rates', [{
        x: rateRange, y: ratePrices, type: 'scatter', mode: 'lines+markers', line: { color: CONFIG.colors.teal, width: 3 }
    }, {
        x: [state.inputs.mortgage_rate], y: [currentPrice], mode: 'markers', marker: { color: 'red', size: 12 }
    }], { margin: { t: 10, r: 10, l: 50, b: 40 }, xaxis: { title: 'Interest Rate (%)' }, yaxis: { title: 'Price ($)' }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: 'Space Grotesk' }, showlegend: false }, { responsive: true });

    // 3. Main Chart: Valuation Confidence Distribution (Bell Curve)
    // Calculate proportional error based on model MAE relative to a baseline market price
    // Average price ~ $1.5M implies ~30% error
    const baselinePrice = 1500000;
    const baseMae = state.activeRun ? state.activeRun.metrics.mae : baselinePrice * 0.25;
    const errorPct = baseMae / baselinePrice;

    // Sigma scales with price
    const sigma = currentPrice * errorPct * 1.0;
    const displayMae = Math.round(sigma * 0.8);

    const mu = currentPrice;

    // Generate distribution curve
    const xMin = mu - 4 * sigma;
    const xMax = mu + 4 * sigma;
    const xValues = [];
    const yValues = [];
    const steps = 100;

    for (let i = 0; i <= steps; i++) {
        const x = xMin + (i / steps) * (xMax - xMin);
        const y = (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
        xValues.push(x);
        yValues.push(y);
    }

    // Likely range (1 sigma)
    const RangeX = xValues.filter(x => x >= mu - sigma && x <= mu + sigma);
    const RangeY = yValues.filter((_, i) => xValues[i] >= mu - sigma && xValues[i] <= mu + sigma);

    Plotly.newPlot('main-chart', [
        {
            x: xValues,
            y: yValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Probability',
            line: { color: CONFIG.colors.ink_light, width: 2, dash: 'solid' },
            fill: 'tozeroy',
            fillcolor: 'rgba(241, 245, 249, 0.5)', // lightly filled
            hoverinfo: 'none'
        },
        {
            x: RangeX,
            y: RangeY,
            type: 'scatter',
            mode: 'lines', // Hidden line, just for fill
            line: { width: 0 },
            fill: 'tozeroy',
            fillcolor: 'rgba(37, 99, 235, 0.1)', // Accent fill for likely range
            hoverinfo: 'none',
            showlegend: false
        },
        {
            x: [mu, mu],
            y: [0, Math.max(...yValues) * 1.1],
            type: 'scatter',
            mode: 'lines',
            name: 'Predicted Value',
            line: { color: CONFIG.colors.accent, width: 4 },
            hoverinfo: 'x'
        }
    ], {
        height: 450,
        margin: { t: 40, b: 40, l: 40, r: 40 },
        showlegend: false,
        xaxis: {
            title: 'Estimated Market Value',
            showgrid: false,
            zeroline: false,
            tickformat: '$,.0f'
        },
        yaxis: {
            showgrid: false,
            zeroline: false,
            showticklabels: false
        },
        font: { family: 'Inter' }, // Matches new premium font
        annotations: [
            {
                x: mu,
                y: Math.max(...yValues) * 1.05,
                xref: 'x',
                yref: 'y',
                text: '▼ Predicted',
                showarrow: false,
                font: { color: CONFIG.colors.accent, size: 14, weight: 600 }
            },
            {
                x: mu - sigma,
                y: Math.max(...yValues) * 0.4,
                xref: 'x',
                yref: 'y',
                text: `-$${Math.round(displayMae / 1000)}k`,
                showarrow: false,
                font: { color: CONFIG.colors.ink_light, size: 12 }
            },
            {
                x: mu + sigma,
                y: Math.max(...yValues) * 0.4,
                xref: 'x',
                yref: 'y',
                text: `+$${Math.round(displayMae / 1000)}k confidence`,
                showarrow: false,
                font: { color: CONFIG.colors.ink_light, size: 12 }
            }
        ]
    }, { responsive: true, displayModeBar: false });
}

async function runInference(customInputs = null) {
    const activeModel = state.activeModel;
    const session = state.models[activeModel];
    const meta = state.metadata[activeModel];
    if (!session || !meta) return 0;

    const inputs = customInputs || state.inputs;
    const processed = calculateDistances(inputs);
    const vector = transformInputs(processed, meta);
    const tensor = new ort.Tensor('float32', Float32Array.from(vector), [1, vector.length]);
    const results = await session.run({ input: tensor });
    const rawOutput = results[Object.keys(results)[0]].data[0];

    if (activeModel === 'nn') {
        const logPrice = rawOutput * meta.y_scale[0] + meta.y_mean[0];
        return Math.exp(logPrice) - 1; // Corresponds to Math.expm1 in python or JS
    } else {
        return Math.exp(rawOutput) - 1;
    }
}

function calculateDistances(inputs) {
    const distMap = {
        "BRECKENRIDGE": { dist_breck: 0.5, dist_keystone: 12, dist_copper: 16, dist_abasin: 18, dist_dillon: 10 },
        "FRISCO": { dist_breck: 9, dist_keystone: 9, dist_copper: 6, dist_abasin: 13, dist_dillon: 4 },
        "SILVERTHORNE": { dist_breck: 12, dist_keystone: 7, dist_copper: 11, dist_abasin: 10, dist_dillon: 1 },
        "DILLON": { dist_breck: 11, dist_keystone: 5, dist_copper: 12, dist_abasin: 9, dist_dillon: 0.5 },
        "KEYSTONE": { dist_breck: 14, dist_keystone: 0.5, dist_copper: 15, dist_abasin: 6, dist_dillon: 6 },
        "COPPER MOUNTAIN": { dist_breck: 17, dist_keystone: 18, dist_copper: 0.5, dist_abasin: 21, dist_dillon: 12 }
    };
    const dists = distMap[inputs.city] || distMap["BRECKENRIDGE"];
    const dist_to_lift = Math.min(dists.dist_breck, dists.dist_keystone, dists.dist_copper, dists.dist_abasin);

    // Normalize city names for model compatibility (e.g., matching metadata categories)
    let modelCity = inputs.city;
    if (modelCity === "COPPER MOUNTAIN") modelCity = "COPPERMOUNTAIN";

    return { ...inputs, ...dists, dist_to_lift, city: modelCity };
}

function transformInputs(inputs, meta) {
    const vector = [];
    meta.input_features.numeric.forEach((feat, i) => {
        vector.push(((inputs[feat] || 0) - meta.num_means[i]) / meta.num_scales[i]);
    });
    meta.input_features.categorical.forEach((feat, i) => {
        const categories = meta.cat_categories[i];
        const val = inputs[feat];
        categories.forEach(cat => vector.push(cat === val ? 1.0 : 0.0));
    });
    return vector;
}

async function loadLeaderboard() {
    try {
        // Cache bust the history file to ensure we get new runs
        const resp = await fetch(`data/experiment_history.json?t=${new Date().getTime()}`);
        const history = await resp.json();
        const tbody = document.querySelector('#leaderboard tbody');
        if (!tbody) return;
        tbody.innerHTML = '';

        const bestRun = history.reduce((prev, curr) => (prev.metrics.mae < curr.metrics.mae) ? prev : curr);
        const bestRunId = bestRun.run_id;

        // Set initial active run if not set
        if (!state.activeRun) {
            state.activeRun = bestRun;
            document.getElementById('model-mae').textContent = `$${Math.round(bestRun.metrics.mae).toLocaleString()}`;
            populateVersionSelector(state.activeModel, bestRun.run_id);
        }

        history.sort((a, b) => b.run_id - a.run_id).forEach(run => {
            const tr = document.createElement('tr');
            tr.dataset.runId = run.run_id;
            tr.style.cursor = 'pointer';
            if (run.run_id === bestRunId) tr.classList.add('best-run');
            if (state.activeRun && run.run_id === state.activeRun.run_id) tr.classList.add('selected-run');

            tr.innerHTML = `<td>#${run.run_id}</td><td>${run.model_name}</td><td>$${Math.round(run.metrics.mae).toLocaleString()}</td><td>${run.metrics.r2?.toFixed(3) || '-'}</td><td>${run.run_id === bestRunId ? '🏆 Champion' : '✅ Verified'}</td>`;

            tr.addEventListener('click', async () => {
                await selectModelVersion(run);
                // Render model detail view with SHAP/PDP
                if (state.activePage === 'experiments') {
                    await renderModelDetail(run);
                }
            });
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error('Failed to load leaderboard:', e);
    }
}

async function loadModels() {
    try {
        for (const [key, cfg] of Object.entries(CONFIG.models)) {
            state.models[key] = await ort.InferenceSession.create(cfg.path);
            state.metadata[key] = await (await fetch(cfg.metadata)).json();
        }
    } catch (e) { }
}
