import pandas as pd
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# Legacy Fallback Points (used if GeoJSON fails)
RESORT_LIFTS_FALLBACK = {
    'dist_breck': [(39.4805, -106.0666)],
    'dist_keystone': [(39.605, -105.9439)],
    'dist_copper': [(39.5022, -106.1506)],
    'dist_abasin': [(39.6425, -105.8719)],
    'dist_dillon': [(39.6050, -106.0660)]
}

def load_address_points(data_dir: str = ".") -> pd.DataFrame:
    """
    Loads the Address Points CSV and selects relevant columns.
    """
    p = Path(data_dir)
    # Search recursively for the address file if not found immediately
    files = list(p.rglob("Address_Points*.csv"))
    if not files:
        # Try finding anywhere in likely dirs if current dir fails (e.g. running from root vs src)
        common_paths = [Path("data"), Path("/Users/brian/Documents/summit/data"), Path(".")]
        for cp in common_paths:
            if cp.exists():
                files = list(cp.rglob("Address_Points*.csv"))
                if files: break
                
        if not files:
             raise FileNotFoundError(f"Could not find Address_Points csv file in {data_dir} or common paths.")
    
    file_path = files[0]
    
    df = pd.read_csv(file_path, low_memory=False)
    cols = ['PropertySchedule', 'LATITUDE', 'LONGITUDE', 'FullAddress']
    df_clean = df[cols].dropna(subset=['PropertySchedule', 'LATITUDE', 'LONGITUDE']).copy()
    
    def clean_schno(x):
        try:
            return str(int(float(x)))
        except:
            return str(x)
            
    df_clean['schno'] = df_clean['PropertySchedule'].apply(clean_schno)
    df_clean = df_clean.groupby('schno').first().reset_index()
    
    return df_clean[['schno', 'LATITUDE', 'LONGITUDE', 'FullAddress']]

def enrich_with_geo_features(sales_df: pd.DataFrame, data_dir: str = ".") -> pd.DataFrame:
    """
    Takes a dataframe with 'schno' and adds:
    1. LATITUDE, LONGITUDE
    2. Distances to Resorts (Min distance to any LIFT LINE for that resort)
    """
    geo_df = load_address_points(data_dir)
    
    sales_df['schno'] = sales_df['schno'].astype(str)
    merged_df = pd.merge(sales_df, geo_df, on='schno', how='left')
    
    # Check for valid lat/lon rows
    mask = merged_df['LATITUDE'].notna() & merged_df['LONGITUDE'].notna()
    if not mask.any():
        return merged_df

    # Create GeoDataFrame for Properties
    # Start with Points in WGS84 (EPSG:4326)
    gdf_props = gpd.GeoDataFrame(
        merged_df[mask], 
        geometry=gpd.points_from_xy(merged_df.loc[mask, 'LONGITUDE'], merged_df.loc[mask, 'LATITUDE']),
        crs="EPSG:4326"
    )

    # Load Ski Lift Data if available
    lift_file_path = Path("data/ski_lifts.geojson")
    # Handle path search if running from src
    if not lift_file_path.exists():
        lift_file_path = Path("/Users/brian/Documents/brian.fishman.info/public/projects/summit/data/ski_lifts.geojson")

    use_real_geometry = False
    if lift_file_path.exists():
        try:
            gdf_lifts = gpd.read_file(lift_file_path)
            # Project to a metric CRS for accurate distance (UTM Zone 13N for Colorado: EPSG:32613)
            gdf_props = gdf_props.to_crs(epsg=32613)
            gdf_lifts = gdf_lifts.to_crs(epsg=32613)
            use_real_geometry = True
            print("Using REAL lift geometry for distance calculation.")
        except Exception as e:
            print(f"Failed to load lift geometry: {e}. Falling back to points.")
    
    # Define Resort Logic (Name mapping)
    resorts = {
        'dist_breck': ['Breckenridge', 'Peak 8', 'Peak 9', 'Peak 7', 'Peak 6', 'Beaver Run', 'Quicksilver', 'SuperConnect', 'Imperial', 'Falcon'],
        'dist_keystone': ['Keystone', 'River Run', 'Montezuma', 'Peru', 'Santiago', 'Outback', 'Wayback'],
        'dist_copper': ['Copper', 'American Eagle', 'American Flyer', 'Super Bee', 'Kokomo', 'Union Creek', 'Timberline'],
        'dist_abasin': ['Arapahoe', 'Black Mountain', 'Pallavicini', 'Lenawee', 'Zuma'],
        # Dillon is just a point usually (Marina), keep as fallback or specific logic if line exists
        'dist_dillon': ['Dillon Marina'] 
    }

    for feature_name, keywords in resorts.items():
        merged_df[feature_name] = np.nan
        
        if use_real_geometry:
            # Filter lifts for this resort
            # We look for a keyword match in 'name' or 'aerialway' tags if name is missing
            # This is a bit fuzzy.
            
            # Simple keyword match in 'name' column if it exists
            if 'name' in gdf_lifts.columns:
                # Create regex pattern from keywords
                import re
                pattern = '|'.join(keywords)
                resort_lifts = gdf_lifts[gdf_lifts['name'].str.contains(pattern, case=False, na=False)]
            else:
                resort_lifts = gpd.GeoDataFrame() # Empty if no name col
            
            # If we found lifts, calculate distance
            if not resort_lifts.empty:
                # Calculate distance from every property to the nearest lift of this resort
                # .distance returns meters in UTM
                # We want the distance to the UNION of all loop lifts
                combined_lift_geom = resort_lifts.unary_union
                
                # distance() element-wise
                # Convert meters to miles (1 meter = 0.000621371 miles)
                dists_meters = gdf_props.distance(combined_lift_geom)
                dists_miles = dists_meters * 0.000621371
                
                # Assign back using the index alignment (gdf_props matches merged_df[mask])
                merged_df.loc[mask, feature_name] = dists_miles.values
                continue
        
        # Fallback (Static Points) if no geometry found or file missing
        # Only calculated if not already set by real geometry
        if pd.isna(merged_df.loc[mask, feature_name]).all():
            fallback_pts = RESORT_LIFTS_FALLBACK.get(feature_name, [])
            if not fallback_pts: continue
            
            # Use original Haversine logic on fallback points
             # haversine function needs to be re-included or imported. 
             # For simplicity, putting it inline or assuming it exists if I didn't delete it.
             # Wait, I am replacing the file, so I need to include haversine logic here.
            
            valid_lons = merged_df.loc[mask, 'LONGITUDE'].values
            valid_lats = merged_df.loc[mask, 'LATITUDE'].values
            final_dists = np.full(len(valid_lons), np.inf)

            for (lat_poi, lon_poi) in fallback_pts:
                 # Re-implement simple haversine here since I am replacing the file
                lon1, lat1, lon2, lat2 = map(np.radians, [valid_lons, valid_lats, lon_poi, lat_poi])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                km = 6367 * c
                d_miles = km * 0.621371
                final_dists = np.minimum(final_dists, d_miles)

            merged_df.loc[mask, feature_name] = final_dists

    return merged_df
