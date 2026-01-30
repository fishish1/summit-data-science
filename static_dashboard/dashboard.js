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

function getThemeLayout() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const isMobile = window.innerWidth < 1024;

    return {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Space Grotesk, sans-serif',
            color: isDark ? '#f8fafc' : '#1e293b',
            size: isMobile ? 10 : 12
        },
        legend: {
            orientation: 'h',
            yanchor: 'top',
            y: isMobile ? -0.4 : -0.2, // Move legend further down on mobile
            xanchor: 'center',
            x: 0.5,
            itemwidth: 30, // More compact legend items
            font: { size: isMobile ? 9 : 11 }
        },
        xaxis: {
            gridcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
            zerolinecolor: isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)',
            tickfont: { size: isMobile ? 9 : 11 }
        },
        yaxis: {
            gridcolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
            zerolinecolor: isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)',
            tickfont: { size: isMobile ? 9 : 11 },
            automargin: true
        },
        margin: {
            t: isMobile ? 60 : 80,
            b: isMobile ? 120 : 100, // Extra bottom margin for legend
            l: isMobile ? 40 : 80,
            r: isMobile ? 20 : 80
        },
        autosize: true
    };
}

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

function calculateRegression(x, y) {
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const xMin = Math.min(...x);
    const xMax = Math.max(...x);
    return {
        x: [xMin, xMax],
        y: [slope * xMin + intercept, slope * xMax + intercept],
        slope: slope,
        intercept: intercept
    };
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
    initTheme();
    initScrolly(); // Initialize scrollytelling observer
    showSkeletons();
    await loadModels();
    hideSkeletons();
    updateDashboard();
});

function showSkeletons() {
    const charts = ['main-chart', 'chart-sqft', 'chart-rates', 'chart-market-trends', 'chart-buyer-origins', 'chart-seasonality', 'chart-supply', 'chart-correlations'];
    charts.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.add('skeleton-loading');
            el.innerHTML = '<div class="skeleton" style="height: 100%; width: 100%;"></div>';
        }
    });
}

function hideSkeletons() {
    const charts = ['main-chart', 'chart-sqft', 'chart-rates', 'chart-market-trends', 'chart-buyer-origins', 'chart-seasonality', 'chart-supply', 'chart-correlations'];
    charts.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove('skeleton-loading');
            el.innerHTML = ''; // Clear skeleton before Plotly takes over
        }
    });
}

function initTheme() {
    const toggles = document.querySelectorAll('.theme-toggle');
    if (toggles.length === 0) return;

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    toggles.forEach(toggle => {
        toggle.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            updateThemeIcon(next);

            // Refresh charts for new theme
            updateDashboard();
        });
    });
}

function updateThemeIcon(theme) {
    const icons = document.querySelectorAll('.theme-icon');
    icons.forEach(icon => {
        icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    });
}

// --- Scrollytelling ---
function initScrolly() {
    const steps = document.querySelectorAll('.scrolly-step');
    const overlay = document.getElementById('scrolly-viz-overlay');
    const progressBar = document.getElementById('scrolly-progress-bar');

    const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -40% 0px',
        threshold: 0.1
    };

    const handleIntersect = (entries) => {
        if (window.innerWidth < 1024) return;
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const step = entry.target.dataset.step;
                activateStep(step);
                updateProgress();
            }
        });
    };

    const observer = new IntersectionObserver(handleIntersect, observerOptions);
    steps.forEach(step => observer.observe(step));

    const updateProgress = () => {
        if (progressBar) {
            const container = document.querySelector('.scrolly-narrative');
            if (container) {
                const scrollPos = window.scrollY - container.offsetTop;
                const totalScroll = container.scrollHeight - window.innerHeight;
                const progress = Math.max(0, Math.min(100, (scrollPos / totalScroll) * 100));
                progressBar.style.width = `${progress}%`;
            }
        }
    };

    window.addEventListener('scroll', updateProgress);

    // Integrated Toggles
    const storyBuyerToggle = document.getElementById('story-toggle-buyer-percentage');
    if (storyBuyerToggle) {
        storyBuyerToggle.onchange = (e) => {
            if (typeof window.renderBuyerOrigins === 'function') {
                window.renderBuyerOrigins(e.target.checked);
            }
        };
    }

    const storyMarketToggle = document.getElementById('story-toggle-market-aggregate');
    if (storyMarketToggle) {
        storyMarketToggle.onchange = (e) => {
            console.log("Market toggle clicked:", e.target.checked);
            if (typeof window.renderMarketTrends === 'function') {
                window.renderMarketTrends(e.target.checked);
            }
        };
    }

    // Responsive Layout Management
    const syncLayout = () => {
        const isMobile = window.innerWidth < 1024;
        const vizTargets = ['landscape', 'ski-lift', 'market-cycles', 'buyer-origins', 'seasonality', 'supply', 'deep-dive'];
        const sharedTarget = document.getElementById('scrolly-viz-target');

        vizTargets.forEach(stepId => {
            const chartId = getTargetIdForStep(stepId);
            const chartEl = document.getElementById(chartId);
            const mobilePlaceholder = document.getElementById(`mobile-viz-${stepId}`);

            if (chartEl && mobilePlaceholder && sharedTarget) {
                if (isMobile) {
                    if (chartEl.parentElement !== mobilePlaceholder) {
                        mobilePlaceholder.appendChild(chartEl);
                        chartEl.classList.remove('hidden');
                        if (chartId.startsWith('chart-') || chartId === 'distance-map') Plotly.Plots.resize(chartEl);
                        if (chartId === 'map-hotspots' && map) map.invalidateSize();
                    }
                } else {
                    if (chartEl.parentElement !== sharedTarget) {
                        sharedTarget.appendChild(chartEl);
                    }
                }
            }
        });

        if (!isMobile) activateStep(steps[0].dataset.step);
    };

    window.addEventListener('resize', syncLayout);
    syncLayout();

    function activateStep(stepId) {
        if (window.innerWidth < 1024) return;

        steps.forEach(s => s.classList.remove('active'));
        const activeStepEl = document.querySelector(`.scrolly-step[data-step="${stepId}"]`);
        if (activeStepEl) activeStepEl.classList.add('active');

        const vizIds = ['map-hotspots', 'distance-map', 'chart-market-trends', 'chart-buyer-origins', 'chart-seasonality', 'chart-supply', 'scrolly-data-explorer'];
        vizIds.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.add('hidden');
        });

        const targetId = getTargetIdForStep(stepId);
        const targetEl = document.getElementById(targetId);
        if (targetEl) {
            targetEl.classList.remove('hidden');
            if (targetId.startsWith('chart-') || targetId === 'distance-map') Plotly.Plots.resize(targetEl);
            if (targetId === 'map-hotspots' && map) map.invalidateSize();
            if (targetId === 'scrolly-data-explorer') renderScrollyExplorer();
        }

        if (overlay) overlay.textContent = getOverlayTextForStep(stepId);
        if (stepId === 'landscape' && map) map.flyTo([39.55, -106.05], 11);
    }

    async function renderScrollyExplorer() {
        const tableBody = document.querySelector('#scrolly-property-table tbody');
        const tableHead = document.querySelector('#scrolly-property-table thead');
        if (!tableBody || tableBody.children.length > 0) return; // Only render once

        const resp = await fetch('data/records_sample_curated.json');
        const data = await resp.json();
        const cols = ['address', 'city', 'year_blt', 'sfla', 'totactval'];

        tableHead.innerHTML = `<tr>${cols.map(c => `<th>${c.toUpperCase()}</th>`).join('')}</tr>`;
        data.slice(0, 50).forEach(r => {
            const tr = document.createElement('tr');
            tr.innerHTML = cols.map(c => {
                let val = r[c];
                if (c === 'totactval') val = `$${val.toLocaleString()}`;
                return `<td>${val}</td>`;
            }).join('');
            tableBody.appendChild(tr);
        });
    }

    function getTargetIdForStep(stepId) {
        const stepMap = {
            'landscape': 'map-hotspots',
            'ski-lift': 'distance-map',
            'market-cycles': 'chart-market-trends',
            'buyer-origins': 'chart-buyer-origins',
            'seasonality': 'chart-seasonality',
            'supply': 'chart-supply',
            'deep-dive': 'scrolly-data-explorer'
        };
        return stepMap[stepId] || 'map-hotspots';
    }

    function getOverlayTextForStep(stepId) {
        const labelMap = {
            'landscape': '📍 Points of Interest',
            'ski-lift': '❄️ Ski Lift Proximity',
            'market-cycles': '📈 Market Cycles',
            'buyer-origins': '🏠 Buyer Demographics',
            'seasonality': '🌡️ Sales Seasonality',
            'supply': '🏗️ Housing Growth',
            'deep-dive': '🔍 Raw Data Explorer'
        };
        return labelMap[stepId] || '';
    }
}

function initUI() {
    // Mobile Menu Toggle
    const menuToggle = document.getElementById('menu-toggle');
    const appNav = document.getElementById('app-nav');
    const sidebar = document.getElementById('sidebar');

    if (menuToggle && appNav) {
        menuToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            appNav.classList.toggle('show');
            sidebar.classList.toggle('nav-open');
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (appNav.classList.contains('show') && !sidebar.contains(e.target)) {
                appNav.classList.remove('show');
                sidebar.classList.remove('nav-open');
            }
        });
    }

    // Main Nav logic
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => {
            // Close mobile menu if open
            if (appNav) {
                appNav.classList.remove('show');
                sidebar.classList.remove('nav-open');
            }

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

    // Theme Toggle is handled in initTheme()
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
                { name: "Arapahoe Basin", lat: 39.6425, lon: -105.8719, "type": "Resort", color: "red" },

                // Town Centers (Blue)
                { name: "Breckenridge", lat: 39.4817, lon: -106.0384, color: "blue", type: "Town" },
                { name: "Frisco", lat: 39.5744, lon: -106.0975, color: "blue", type: "Town" },
                { name: "Silverthorne", lat: 39.6296, lon: -106.0713, color: "blue", type: "Town" },
                { name: "Dillon", lat: 39.6294, lon: -106.0438, color: "blue", type: "Town" },
                { name: "Keystone", lat: 39.6080, lon: -105.9525, color: "blue", type: "Town" },

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
            <div class="metric-item"><div class="metric-value">100+</div><div class="metric-label">Data Points Per Property</div></div>
        `;


        // Distance Map
        renderDistanceAnalysis();

        // Raw Sample Table
        const container = document.getElementById('raw-sample-container');
        const toggle = document.getElementById('toggle-all-cols');

        if (container) {
            const renderTable = (showAll) => {
                const displayCols = showAll ? Object.keys(data[0]) : ['address', 'city', 'year_blt', 'sfla', 'totactval'];
                let html = '<table class="data-table"><thead><tr>';
                displayCols.forEach(c => html += `<th>${c}</th>`);
                html += '</tr></thead><tbody>';

                data.slice(0, 50).forEach(r => {
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

        const isMobile = window.innerWidth < 1024;
        const designLayout = {
            mapbox: {
                style: "carto-positron",
                center: { lat: 39.55, lon: -106.05 },
                zoom: isMobile ? 10 : 11
            },
            margin: { t: 0, b: 0, l: 0, r: 0 },
            showlegend: false,
            height: isMobile ? 400 : 700
        };
        Plotly.newPlot('distance-map', [trace], designLayout, { responsive: true });

    } catch (e) { console.error("Distance Map Error", e); }
}

let trendsCache = null;
let ratesCache = null;

window.renderMarketTrends = async (showAggregate = false) => {
    try {
        if (!trendsCache) trendsCache = await (await fetch('data/market_trends.json')).json();
        if (!ratesCache) ratesCache = await (await fetch('data/mortgage_history.json')).json();

        const trends = trendsCache;
        const rates = ratesCache;
        const cities = ["BRECKENRIDGE", "FRISCO", "SILVERTHORNE", "DILLON", "KEYSTONE"];

        let traces = [];

        if (showAggregate) {
            // Group by year and average the 3yr MA across cities
            const years = [...new Set(trends.map(d => d.tx_year))].sort();
            const aggData = years.map(y => {
                const yearCityData = trends.filter(d => d.tx_year === y && cities.includes(d.city));
                const avg = yearCityData.reduce((sum, d) => sum + d.avg_price_3yr_ma, 0) / (yearCityData.length || 1);
                return { tx_year: y, avg_price_3yr_ma: avg };
            }).filter(d => d.avg_price_3yr_ma > 0);

            traces.push({
                x: aggData.map(d => d.tx_year),
                y: aggData.map(d => d.avg_price_3yr_ma),
                name: 'County Average (3yr MA)',
                type: 'scatter',
                line: { color: CONFIG.colors.accent, width: 4 } // Use accent color for visibility
            });
        } else {
            traces = cities.map((city, idx) => {
                const cityData = trends.filter(d => d.city === city);
                return {
                    x: cityData.map(d => d.tx_year),
                    y: cityData.map(d => d.avg_price_3yr_ma),
                    name: city,
                    type: 'scatter',
                    line: { width: 2 }
                };
            });
        }

        // Add Mortgage Rate Trace (Secondary Axis)
        traces.push({
            x: rates.map(d => d.year.toString()),
            y: rates.map(d => d.value),
            name: '30Y Mortgage Rate',
            type: 'scatter',
            yaxis: 'y2',
            line: { color: '#ef4444', width: 3, dash: 'dot' },
            hovertemplate: '%{y:.2f}%<extra></extra>'
        });

        const layout = {
            ...getThemeLayout(),
            title: showAggregate ? 'County-Wide Trends vs Rates' : 'Town-Level Trends vs Rates',
            height: 500,
            yaxis: {
                ...getThemeLayout().yaxis,
                title: 'Avg Price (3yr MA)',
                tickformat: '$,.0f'
            },
            yaxis2: {
                title: 'Mortgage Rate (%)',
                overlaying: 'y',
                side: 'right',
                range: [0, 20],
                showgrid: false,
                titlefont: { color: '#ef4444' },
                tickfont: { color: '#ef4444' },
                ticksuffix: '%'
            },
            legend: {
                ...getThemeLayout().legend,
                y: -0.3 // Push legend down a bit more for double scale
            }
        };

        Plotly.newPlot('chart-market-trends', traces, layout, { responsive: true });

    } catch (e) { console.error("Market Trends Error", e); }
};

async function renderMarketHistory() {
    try {
        // Initial call for each module
        await window.renderMarketTrends(true);
        await renderBuyerOriginsModule();
        await renderSeasonalityHeatmap();
        await renderSupplyGrowth();
    } catch (e) { console.error("Market History Wrapper Error", e); }
}

async function renderBuyerOriginsModule() {
    try {
        const owners = await (await fetch('data/owner_trends.json')).json();
        const types = ['Local (In-County)', 'In-State (Non-Local)', 'Out-of-State'];

        window.renderBuyerOrigins = (showPercentage) => {
            if (showPercentage) {
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
                    ...getThemeLayout(),
                    title: 'Buyer Origin Percentage',
                    height: 500,
                    yaxis: { ...getThemeLayout().yaxis, title: 'Percentage (%)', ticksuffix: '%' }
                }, { responsive: true });
            } else {
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
                    ...getThemeLayout(),
                    title: 'Buyer Count by Origin',
                    height: 500,
                    yaxis: { ...getThemeLayout().yaxis, title: 'Number of Buyers' }
                }, { responsive: true });
            }
        };

        // Initial render
        window.renderBuyerOrigins(false);

        // Set up toggle
        const buyerToggle = document.getElementById('toggle-buyer-percentage');
        if (buyerToggle) {
            buyerToggle.onchange = (e) => window.renderBuyerOrigins(e.target.checked);
        }
    } catch (e) { console.error("Buyer Origins Error", e); }
}

async function renderSeasonalityHeatmap() {
    try {
        const seasonal = await (await fetch('data/seasonality.json')).json();
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const uniqueCities = [...new Set(seasonal.map(d => d.city))];
        const zData = months.map(m => uniqueCities.map(c => {
            const d = seasonal.find(x => x.month_name === m && x.city === c);
            return d ? d.sales_count : 0;
        }));
        Plotly.newPlot('chart-seasonality', [{ z: zData, x: uniqueCities, y: months, type: 'heatmap', colorscale: 'RdBu', reversescale: true }], {
            ...getThemeLayout(),
            title: 'Monthly Sales Volume by City',
            height: 500
        }, { responsive: true });
    } catch (e) { console.error("Seasonality Error", e); }
}

async function renderSupplyGrowth() {
    try {
        const supply = await (await fetch('data/supply_growth.json')).json();
        const densityData = supply.map(d => {
            const pop = getPop(d.year_blt);
            return {
                year: d.year_blt,
                sqft: d.cumulative_sqft,
                pop: pop,
                per_capita: d.cumulative_sqft / pop
            };
        }).filter(d => d.year >= 1960);

        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

        Plotly.newPlot('chart-supply', [
            {
                x: densityData.map(d => d.year),
                y: densityData.map(d => d.sqft),
                name: 'Total SqFt Supply',
                type: 'scatter',
                fill: 'tozeroy',
                line: { color: isDark ? '#94a3b8' : CONFIG.colors.primary },
                fillcolor: isDark ? 'rgba(148, 163, 184, 0.2)' : 'rgba(15, 27, 44, 0.1)'
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
            ...getThemeLayout(),
            title: 'Housing Supply Growth vs Population',
            height: 500,
            yaxis: { ...getThemeLayout().yaxis, title: 'Total Residential SqFt' },
            yaxis2: {
                ...getThemeLayout().yaxis,
                title: 'SqFt / Capita',
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                titlefont: { color: CONFIG.colors.teal },
                tickfont: { color: CONFIG.colors.teal }
            }
        }, { responsive: true });
    } catch (e) { console.error("Supply Growth Error", e); }
}



async function renderCorrelationMatrix() {
    try {
        const corrResp = await fetch('data/correlation_matrix.json');
        const data = await corrResp.json();

        const labels = data.map(d => d.index);
        const zData = labels.map(rowLabel => labels.map(colLabel => data.find(d => d.index === rowLabel)[colLabel] || 0));

        const chartId = 'chart-correlations';
        Plotly.newPlot(chartId, [{
            z: zData, x: labels, y: labels,
            type: 'heatmap', colorscale: 'RdBu', zmin: -1, zmax: 1
        }], {
            ...getThemeLayout(),
            title: 'Feature Correlation Matrix',
            height: 500
        }, { responsive: true });

        // Click Handler for Drill-down
        document.getElementById(chartId).on('plotly_click', (data) => {
            if (data.points && data.points.length > 0) {
                const varX = data.points[0].x;
                const varY = data.points[0].y;
                renderDrillDown(varX, varY);
            }
        });

    } catch (e) {
        console.error('Failed to load correlation matrix:', e);
    }
}

async function renderDrillDown(varX, varY) {
    const drilldownEl = document.getElementById('correlation-drilldown');
    if (!drilldownEl) return;

    drilldownEl.classList.remove('hidden');
    document.getElementById('drilldown-var-x').textContent = varX;
    document.getElementById('drilldown-var-y').textContent = varY;

    // Scroll to view
    drilldownEl.scrollIntoView({ behavior: 'smooth', block: 'center' });

    try {
        const resp = await fetch('data/records_sample_curated.json');
        const records = await resp.json();

        // Map display labels to JSON keys
        const labelToKey = {
            'Price': 'totactval',
            'SqFt': 'sfla',
            'Beds': 'beds',
            'Baths': 'f_baths',
            'Year Built': 'year_blt',
            'Acres': 'acres',
            'Garage': 'garage_size',
            'Dist Breck': 'dist_breck'
        };

        const keyX = labelToKey[varX] || varX;
        const keyY = labelToKey[varY] || varY;

        const trace = {
            x: records.map(r => r[keyX]),
            y: records.map(r => r[keyY]),
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: CONFIG.colors.accent,
                opacity: 0.6,
                size: 8,
                line: { width: 1, color: 'white' }
            },
            text: records.map(r => `${r.address || 'N/A'}<br>${varX}: ${r[keyX]}<br>${varY}: ${r[keyY]}`),
            hoverinfo: 'text'
        };

        const reg = calculateRegression(trace.x, trace.y);
        const regTrace = {
            x: reg.x,
            y: reg.y,
            mode: 'lines',
            type: 'scatter',
            name: `Trend (β=${reg.slope.toFixed(2)})`,
            line: { color: '#ef4444', width: 2, dash: 'dot' }
        };

        const layout = {
            ...getThemeLayout(),
            title: `Correlation Detail: ${varX} vs ${varY}`,
            xaxis: { ...getThemeLayout().xaxis, title: varX },
            yaxis: { ...getThemeLayout().yaxis, title: varY },
            height: 500,
            showlegend: true
        };

        Plotly.newPlot('chart-correlation-drilldown', [trace, regTrace], layout);

    } catch (e) {
        console.error('Failed to render drilldown:', e);
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

        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

        const shapTrace = {
            y: shapData.map(d => d.feature),
            x: shapData.map(d => d.importance),
            type: 'bar',
            orientation: 'h',
            marker: { color: isDark ? CONFIG.colors.accent : CONFIG.colors.primary }
        };

        Plotly.newPlot('chart-shap-experiments', [shapTrace], {
            ...getThemeLayout(),
            title: `Feature Importance (SHAP):<br>${run.model_name} (Run #${run.run_id})`,
            xaxis: { ...getThemeLayout().xaxis, title: 'Mean |SHAP value|' },
            yaxis: { ...getThemeLayout().yaxis, automargin: true },
            height: 500
        }, { responsive: true });

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
                ...getThemeLayout(),
                title: `Marginal Effect: ${formatFeatureName(feature)}`,
                yaxis: { ...getThemeLayout().yaxis, title: 'Predicted Log Price' },
                xaxis: { ...getThemeLayout().xaxis, title: formatFeatureName(feature) },
                height: 500
            }, { responsive: true });
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
            const card = priceEl.closest('.metric-card');
            if (card) {
                card.classList.remove('pulse-update');
                void card.offsetWidth; // Force reflow
                card.classList.add('pulse-update');
            }
        }

        if (lastUpdatedEl) {
            lastUpdatedEl.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
        }

        await updatePredictorCharts(price);
        await renderSHAPExplainer(price);
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
    }], {
        ...getThemeLayout(),
        margin: { t: 10, r: 10, l: 50, b: 40 },
        xaxis: { ...getThemeLayout().xaxis, title: 'SqFt' },
        yaxis: { ...getThemeLayout().yaxis, title: 'Price ($)' },
        showlegend: false
    }, { responsive: true });

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
    }], {
        ...getThemeLayout(),
        margin: { t: 10, r: 10, l: 50, b: 40 },
        xaxis: { ...getThemeLayout().xaxis, title: 'Interest Rate (%)' },
        yaxis: { ...getThemeLayout().yaxis, title: 'Price ($)' },
        showlegend: false
    }, { responsive: true });

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

    // Likely ranges
    const Range2X = xValues.filter(x => x >= mu - 2 * sigma && x <= mu + 2 * sigma);
    const Range2Y = yValues.filter((_, i) => xValues[i] >= mu - 2 * sigma && xValues[i] <= mu + 2 * sigma);

    const Range1X = xValues.filter(x => x >= mu - sigma && x <= mu + sigma);
    const Range1Y = yValues.filter((_, i) => xValues[i] >= mu - sigma && xValues[i] <= mu + sigma);

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

    Plotly.newPlot('main-chart', [
        {
            x: xValues,
            y: yValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Probability',
            line: { color: isDark ? '#94a3b8' : '#64748b', width: 2, dash: 'solid' },
            fill: 'tozeroy',
            fillcolor: isDark ? 'rgba(148, 163, 184, 0.1)' : 'rgba(241, 245, 249, 0.5)',
            hoverinfo: 'none'
        },
        // 2 Sigma Range (95%)
        {
            x: Range2X,
            y: Range2Y,
            type: 'scatter',
            mode: 'lines',
            line: { width: 0 },
            fill: 'tozeroy',
            fillcolor: isDark ? 'rgba(59, 130, 246, 0.1)' : 'rgba(37, 99, 235, 0.05)',
            hoverinfo: 'none',
            showlegend: false
        },
        // 1 Sigma Range (68%)
        {
            x: Range1X,
            y: Range1Y,
            type: 'scatter',
            mode: 'lines',
            line: { width: 0 },
            fill: 'tozeroy',
            fillcolor: isDark ? 'rgba(59, 130, 246, 0.2)' : 'rgba(37, 99, 235, 0.15)',
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
        ...getThemeLayout(),
        height: 450,
        margin: { t: 40, b: 40, l: 40, r: 40 },
        showlegend: false,
        xaxis: {
            ...getThemeLayout().xaxis,
            title: 'Estimated Market Value',
            showgrid: false,
            zeroline: false,
            tickformat: '$,.0f'
        },
        yaxis: {
            ...getThemeLayout().yaxis,
            showgrid: false,
            zeroline: false,
            showticklabels: false
        },
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
                font: { color: isDark ? '#94a3b8' : '#64748b', size: 12 }
            },
            {
                x: mu + sigma / 2,
                y: Math.max(...yValues) * 0.2,
                xref: 'x',
                yref: 'y',
                text: '68%',
                showarrow: false,
                font: { color: CONFIG.colors.accent, size: 12, weight: 'bold' }
            },
            {
                x: mu + 1.5 * sigma,
                y: Math.max(...yValues) * 0.1,
                xref: 'x',
                yref: 'y',
                text: '95%',
                showarrow: false,
                font: { color: CONFIG.colors.accent, size: 11 }
            },
            {
                x: mu + sigma,
                y: Math.max(...yValues) * 0.4,
                xref: 'x',
                yref: 'y',
                text: `+$${Math.round(displayMae / 1000)}k confidence`,
                showarrow: false,
                font: { color: isDark ? '#94a3b8' : '#64748b', size: 12 }
            }
        ]
    }, { responsive: true, displayModeBar: false });
}

async function renderSHAPExplainer(currentPrice) {
    const activeModel = state.activeModel;
    const meta = state.metadata[activeModel];
    if (!meta) return;

    // Features to explain
    const numericFeatures = [
        { key: 'sfla', label: 'SqFt' },
        { key: 'year_blt', label: 'Year Built' },
        { key: 'garage_size', label: 'Garage' },
        { key: 'acres', label: 'Lot Size' },
        { key: 'mortgage_rate', label: 'Mortgage Rate' },
        { key: 'cpi', label: 'CPI (Inflation)' },
        { key: 'sp500', label: 'S&P 500' },
        { key: 'summit_pop', label: 'County Pop.' },
        { key: 'grade_numeric', label: 'Grade' },
        { key: 'cond_numeric', label: 'Condition' },
        { key: 'scenic_view', label: 'View Score' },
        { key: 'beds', label: 'Beds' },
        { key: 'baths', label: 'Baths' }
    ];

    const categoricalFeatures = [
        { key: 'city', label: 'Location' },
        { key: 'prop_type', label: 'Property Type' }
    ];

    // Baseline: Use num_means for numeric, and reasonable defaults for categorical
    const baselineInputs = { ...state.inputs };

    // Set numeric baselines from metadata indices
    meta.input_features.numeric.forEach((feat, idx) => {
        baselineInputs[feat] = meta.num_means[idx];
    });

    // Static baselines for categorical (most common)
    baselineInputs.city = "BRECKENRIDGE";
    baselineInputs.prop_type = "Single Family";

    const contributions = [];

    // Calculate numeric contributions using perturbation
    for (const feat of numericFeatures) {
        if (state.inputs[feat.key] === undefined) continue;

        const altInputs = { ...state.inputs, [feat.key]: baselineInputs[feat.key] };
        const altPrice = await runInference(altInputs);
        const delta = currentPrice - altPrice;

        if (Math.abs(delta) > 100) {
            contributions.push({ label: feat.label, delta: delta });
        }
    }

    // Calculate categorical contributions
    for (const feat of categoricalFeatures) {
        if (state.inputs[feat.key] === undefined) continue;

        const altInputs = { ...state.inputs, [feat.key]: baselineInputs[feat.key] };
        const altPrice = await runInference(altInputs);
        const delta = currentPrice - altPrice;

        if (Math.abs(delta) > 100) {
            contributions.push({ label: feat.label, delta: delta });
        }
    }

    // Sort by absolute impact
    contributions.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

    // Limit to top results
    const plotData = contributions.slice(0, 10);

    const labels = plotData.map(c => c.label);
    const deltas = plotData.map(c => c.delta);
    const colors = deltas.map(d => d >= 0 ? '#10b981' : '#ef4444');

    const chartId = 'chart-shap';
    const el = document.getElementById(chartId);
    if (!el) return;

    Plotly.newPlot(chartId, [{
        type: 'bar',
        x: deltas,
        y: labels,
        orientation: 'h',
        marker: { color: colors },
        text: deltas.map(d => (d >= 0 ? '+' : '') + new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(d)),
        textposition: 'auto',
        hovertemplate: '%{y}: %{x:$,.0f}<extra></extra>'
    }], {
        ...getThemeLayout(),
        margin: { t: 20, r: 50, l: 120, b: 40 },
        xaxis: {
            title: 'Value Impact ($)',
            zeroline: true,
            zerolinecolor: '#94a3b8',
            tickformat: '$,.0s'
        },
        yaxis: {
            autorange: 'reversed'
        }
    }, { responsive: true });
}

let inferenceMutex = false;

async function runInference(customInputs = null) {
    // Prevent concurrent inference calls (causes Session mismatch in ORT)
    while (inferenceMutex) {
        await new Promise(resolve => setTimeout(resolve, 30));
    }
    inferenceMutex = true;

    try {
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
    } finally {
        inferenceMutex = false;
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
