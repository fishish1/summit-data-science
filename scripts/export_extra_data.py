import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from summit_housing.queries import MarketAnalytics
from summit_housing.ml import train_macro_model, get_shap_values, get_pdp_data, get_available_pdp_features

# Configure export path
# Priority: 1) Environment variable, 2) Side-by-side repos, 3) Hardcoded fallback
def get_export_path():
    """
    Determine where to export data files.
    
    Returns the export directory path. Tries in order:
    1. SUMMIT_EXPORT_PATH environment variable
    2. ../brian.fishman.info/public/projects/summit/data (if repos are side-by-side)
    3. Hardcoded path as fallback
    """
    # Check environment variable first
    env_path = os.getenv('SUMMIT_EXPORT_PATH')
    if env_path:
        return Path(env_path)
    
    # Try side-by-side repos (most common setup)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # /path/to/summit
    sibling_path = project_root.parent / "brian.fishman.info/public/projects/summit/data"
    
    if sibling_path.exists():
        return sibling_path
    
    # Fallback to hardcoded path
    return Path("/Users/brian/Documents/brian.fishman.info/public/projects/summit/data")

EXPORT_BASE_PATH = get_export_path()

def export_to_json(df, filename):
    output_path = EXPORT_BASE_PATH / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to list of dicts and handle NaN -> null accurately
    data = df.to_dict(orient='records')
    
    import math
    def clean(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
        if isinstance(obj, np.int64): return int(obj)
        if isinstance(obj, np.float64): return float(obj)
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(x) for x in obj]
        return obj
    
    data = clean(data)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Exported {len(data)} rows to {output_path}")

def main():
    print(f"📊 Summit Housing Data Export")
    print(f"=" * 60)
    print(f"Export destination: {EXPORT_BASE_PATH}")
    print(f"=" * 60)
    print()
    
    analytics = MarketAnalytics()
    
    # 1. Standard Analytics
    print("Exporting Market Trends...")
    trends = analytics.get_market_trends(exclude_multiunit=True)
    export_to_json(trends, "market_trends.json")
    
    print("Exporting Owner Trends...")
    owner_trends = analytics.get_owner_purchase_trends()
    export_to_json(owner_trends, "owner_trends.json")
    
    print("Exporting Seasonality...")
    seasonality = analytics.get_seasonality_stats()
    export_to_json(seasonality, "seasonality.json")
    
    print("Exporting Cumulative Supply...")
    supply = analytics.get_cumulative_supply()
    export_to_json(supply, "supply_growth.json")
    
    # 2. Raw Data (Cleaned)
    print("Exporting Raw Sample (Summit Only)...")
    try:
        # Load Raw Data (to keep all original columns)
        df_raw = pd.read_csv("data/records.csv", low_memory=False)
        target_cities = ['BRECKENRIDGE', 'FRISCO', 'SILVERTHORNE', 'DILLON', 'KEYSTONE', 'COPPER MOUNTAIN', 'BLUE RIVER']
        df_clean = df_raw[df_raw['city'].isin(target_cities)].copy()
        
        # --- ENRICHMENT (Geo + Macro) ---
        print("   Enriching Raw Sample...")
        
        # 1. Geo
        from summit_housing.geo import enrich_with_geo_features
        try:
             df_clean = enrich_with_geo_features(df_clean)
        except Exception as e:
             print(f"   Warning: Geo enrichment failed within export: {e}")

        # 2. Macro (Manually merge to preserve raw columns)
        try:
            df_clean['sale_date'] = pd.to_datetime(df_clean['sale_date'], errors='coerce')
            df_clean = df_clean.sort_values('sale_date')
            
            macro_files = {
                'mortgage_rate': 'data/mortgage_rate.csv',
                'cpi': 'data/cpi.csv',
                'sp500': 'data/sp500.csv'
            }
            
            for name, path in macro_files.items():
                if os.path.exists(path):
                    macro_df = pd.read_csv(path)
                    macro_df['date'] = pd.to_datetime(macro_df['date'])
                    macro_df = macro_df.sort_values('date').rename(columns={'value': name})
                    df_clean = pd.merge_asof(df_clean, macro_df[['date', name]], left_on='sale_date', right_on='date', direction='backward')
        except Exception as e:
            print(f"   Warning: Macro enrichment failed: {e}")

        # -------------------------------

        # Take a robust sample
        if len(df_clean) > 200:
            df_curated = df_clean.sample(200, random_state=42)
        else:
            df_curated = df_clean
            
        # Basic columns for default view
        basic_cols = ['schno', 'address', 'city', 'year_blt', 'sfla', 'beds', 'f_baths', 'totactval']
        
        # Export ALL columns for "Show All" functionality
        print(f"   Exporting ALL {len(df_curated.columns)} columns...")
        
        # Handle NaN for JSON export (fill numeric with 0, strings with "")
        # Actually, let's just use fillna(0) for simplicity or let JSON encoder handle nulls (pandas to_json handles nulls usually)
        # But our export_to_json helper uses orient='records'.
        
        # Let's replace NaNs with None so they show as empty/null in JSON
        df_export = df_curated.where(pd.notnull(df_curated), None)
        
        export_to_json(df_export, "records_sample_curated.json")
    except Exception as e:
        print(f"Error exporting raw sample: {e}")

    # 3. ML Explainability (SHAP & PDP)
    print("Exporting ML Explainability (SHAP & PDP)...")
    try:
        # Get Model Environment
        # train_macro_model returns: pipeline, ip, X_test, y_test, input_cols, shap_cols
        pipeline, ip, X_test, y_test, features, shap_cols = train_macro_model()
        
        # A. SHAP Values
        print("   Calculating SHAP...")
        explainer, shap_vals = get_shap_values(pipeline, X_test)
        
        # Summarize for the chart (Feature Name -> Mean(|SHAP|))
        # Ensure we handle the matrix correctly
        if isinstance(shap_vals, list): 
             # For some models/outputs it might be a list, take the first one (regression)
             shap_vals = shap_vals[0]
             
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_summary = pd.DataFrame({
            'feature': shap_cols,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False).head(20)
        
        export_to_json(shap_summary, "shap_summary.json")
        
        # B. PDP Data
        print("   Calculating PDP...")
        
        # Get all available features dynamically
        pdp_features = get_available_pdp_features()
        print(f"   Found {len(pdp_features)} features for PDP analysis")
        
        pdp_results = []
        for feat in pdp_features:
            print(f"      PDP for {feat}...")
            # Fix append deprecation
            X_sample = pd.concat([X_test, X_test])
            df_pdp = get_pdp_data(feat)
            
            # Format for export
            if not df_pdp.empty:
                df_pdp['feature'] = feat
                pdp_results.append(df_pdp)
        
        if pdp_results:
            all_pdp = pd.concat(pdp_results)
            export_to_json(all_pdp, "pdp_data.json")
            
    except Exception as e:
        print(f"Error exporting ML metrics: {e}")

    # 4. Geo-Analytics (Distance Map)
    print("Exporting Geo Data...")
    try:
        # Load from CSV directly to ensure columns exist
        df_geo = pd.read_csv("data/records.csv", low_memory=False)
        target_cities = ['BRECKENRIDGE', 'FRISCO', 'SILVERTHORNE', 'DILLON', 'KEYSTONE', 'COPPER MOUNTAIN']
        df_geo = df_geo[df_geo['city'].isin(target_cities)].copy()
        
        # Enrich
        from summit_housing.geo import enrich_with_geo_features
        df_geo = enrich_with_geo_features(df_geo)
        
        lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
        valid_lifts = [c for c in lift_cols if c in df_geo.columns]
        
        if valid_lifts:
            df_geo['dist_to_lift'] = df_geo[valid_lifts].min(axis=1)
            # Ensure price column exists (totactval)
            if 'totactval' not in df_geo.columns:
                df_geo['totactval'] = 0
                
            df_geo = df_geo.dropna(subset=['LATITUDE', 'LONGITUDE', 'dist_to_lift'])
            
            # Keep only lightweight cols
            out_geo = df_geo[['LATITUDE', 'LONGITUDE', 'dist_to_lift', 'address', 'city', 'totactval']]
            # Rename for frontend consistency
            out_geo = out_geo.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'totactval': 'price'})
            
            export_to_json(out_geo, "geo_distances.json")
            
    except Exception as e:
        print(f"Error exporting geo data: {e}")

    # 5. Experiment History
    print("Exporting Experiment History...")
    try:
        import shutil
        history_src = "models/experiment_history.json"
        
        if os.path.exists(history_src):
            # Load and verify it's valid JSON
            with open(history_src, 'r') as f:
                history_data = json.load(f)
                
            export_to_json(pd.DataFrame(history_data), "experiment_history.json")
        else:
            print("   ⚠️ No experiment history found at models/experiment_history.json")
            
    except Exception as e:
        print(f"Error exporting experiment history: {e}")

if __name__ == "__main__":
    main()
