# Dashboard Refactor Plan (Story-First, DS/ML Focus)

## Narrative Tabs
1) **Data & Limits**
   - Dataset summary, schema snippet, sample rows.
   - Limits/bias: current-owner survivorship, scraped-sales quality, missing past buyer demographics.
   - QC/coverage: missing stats, counts. Add seasonality chart (monthly sales volume/price).

2) **Market Factors (Context)**
   - Price trends (3yr MA) by city + volume bubbles, mortgage overlay.
   - Supply context: cumulative housing vs hotel units.
   - Buyer mix: location pie, owner entity type, portfolio size; top out-of-state locations table.

3) **Drivers & Pricing (Feature Importance)**
   - Property type segmentation (price + volume by type).
   - Geospatial: ski-lift proximity map.
   - SHAP feature importance (bar), PDPs for top features, correlation matrix for features.

4) **Price Predictor (Inference)**
   - Interactive simulator (sqft, beds, baths, year, location, type).
   - Sensitivity readout (±10% sqft/year deltas) and brief model card (data window, features, limits, target definition, leakage checks, perf metrics MAE/R², temporal CV note).

## Gaps to Add Next
- Seasonality chart (monthly sales volume/price) using analytics.get_seasonality_stats().
- Sensitivity panel in predictor tab.
- Short explanations under PDP/Correlation for interpretability + actionability (what to do with findings).
- Model validation: report MAE/R², use temporal cross-validation/backtesting; add data/feature versioning note.
- Data quality gates: missing/coverage thresholds, drift checks on key features (sfla, price, rates, dist_to_lift).

## Additional Charts to Add
- Seasonality: monthly sales volume and median price/PPSF (peak months).
- Holding Period & Sales Velocity: histogram of days_held; distribution of times_sold per property.
- PPSF vs Size: scatter or box/LOESS to show diminishing returns on square footage.
- Renovation Premium: compare price/PPSF for is_renovated vs not (adj_year_blt > year_blt).
- Cohort Appreciation: track appreciation by purchase-year cohorts.
- Luxury/Outlier Tail: violin/box for top decile price/PPSF.
- Price-to-Rate Sensitivity: overlay mortgage rate with PPSF YoY change; simple correlation metric.
- SHAP Dependence Plots: feature-level (sfla, year_blt, dist_to_lift) non-linear effects.
- Supply vs Volume: cumulative housing units vs annual sales volume (absorption vs stock).

## Data Sources Already Leveraged
- owner_location_summary.csv, owner_type_summary.csv, multi_property_owner_breakdown.csv, top_out_of_state_locations.csv
- local_housing_supply_timeline.csv, hotel_growth_timeline.csv
- sp500_annual.csv

## Notes
- No cross-filtering reintroduced.
- Survivor bias note stays in Buyer Origins.
- Supply chart added in Macro Comparison.
- Prioritize for ship: (1) Seasonality, (2) Validation metrics + model card, (3) Sensitivity panel; others are nice-to-have.
