import pandas as pd
import numpy as np
import os
from pathlib import Path

# POI Coordinates
# Key: Feature Name
# Value: List of (Lat, Lon) tuples. 
# We calculate the MIN distance to ANY of the points in the list.
RESORT_LIFTS = {
    'dist_breck': [
        (39.4851, -106.0666), # Peak 8 Base (Colorado SuperChair)
        (39.4792, -106.0465), # Peak 9 Base (Quicksilver)
        (39.4938, -106.0682), # Peak 7 Base (Independence)
        (39.4759, -106.0526), # Beaver Run
        (39.4788, -106.0531)  # Snowflake
    ],
    'dist_keystone': [
        (39.6105, -105.9439), # River Run Gondola
        (39.6074, -105.9567)  # Mountain House Base
    ],
    'dist_copper': [
        (39.5022, -106.1506), # American Eagle (Center)
        (39.5020, -106.1517), # American Flyer (Center)
        (39.5056, -106.1384), # Super Bee (East)
        (39.4975, -106.1627)  # Kokomo (West)
    ],
    'dist_abasin': [
        (39.6425, -105.8719)  # Black Mountain Express (Base)
    ],
    'dist_dillon': [
        (39.6050, -106.0660)  # Dillon Marina / Reservoir Center
    ]
}

def load_address_points(data_dir: str = ".") -> pd.DataFrame:
    """
    Loads the Address Points CSV and selects relevant columns.
    """
    # Find the address file dynamically if possible
    p = Path(data_dir)
    files = list(p.glob("Address_Points*.csv"))
    if not files:
        # Fallback to check if it's in the root when data_dir is data
        p_root = Path(".")
        files = list(p_root.glob("Address_Points*.csv"))
        if not files:
            raise FileNotFoundError("Could not find Address_Points csv file.")
    
    file_path = files[0]
    
    df = pd.read_csv(file_path, low_memory=False)
    
    # Keep only what we need
    # PropertySchedule seems to be the Join Key (schno).
    # LATITUDE, LONGITUDE
    cols = ['PropertySchedule', 'LATITUDE', 'LONGITUDE', 'FullAddress']
    
    # Filter for valid PropertySchedule and Coords
    df_clean = df[cols].dropna(subset=['PropertySchedule', 'LATITUDE', 'LONGITUDE']).copy()
    
    # Clean Join Key: ensure it's a string, strip decimals if any (though usually integer-like)
    # Some PropertySchedule might be like '1234567.0' if loaded as float
    # But read_csv might load as object or float.
    
    # Helper to clean schno
    def clean_schno(x):
        try:
            return str(int(float(x)))
        except:
            return str(x)
            
    df_clean['schno'] = df_clean['PropertySchedule'].apply(clean_schno)
    
    # Group by schno to remove duplicates (same property might have multiple address points?)
    # We take the first one.
    df_clean = df_clean.groupby('schno').first().reset_index()
    
    return df_clean[['schno', 'LATITUDE', 'LONGITUDE', 'FullAddress']]

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def enrich_with_geo_features(sales_df: pd.DataFrame, data_dir: str = ".") -> pd.DataFrame:
    """
    Takes a dataframe with 'schno' and adds:
    1. LATITUDE, LONGITUDE
    2. Distances to POIs (Min distance to any base area)
    """
    geo_df = load_address_points(data_dir)
    
    # Ensure join key consistency
    sales_df['schno'] = sales_df['schno'].astype(str)
    
    # Merge
    merged_df = pd.merge(sales_df, geo_df, on='schno', how='left')
    
    # Calculate Distances
    # Vectorized calculation for each feature (min distance to any base in the list)
    for feature_name, coords_list in RESORT_LIFTS.items():
        # Handle missing lat/lon
        mask = merged_df['LATITUDE'].notna()
        
        merged_df[feature_name] = np.inf
        
        if not mask.any():
            continue
            
        valid_lons = merged_df.loc[mask, 'LONGITUDE'].values
        valid_lats = merged_df.loc[mask, 'LATITUDE'].values
        
        final_dists = np.full(len(valid_lons), np.inf)
        
        for (lat_poi, lon_poi) in coords_list:
            d = haversine_np(valid_lons, valid_lats, lon_poi, lat_poi)
            final_dists = np.minimum(final_dists, d)
            
        merged_df.loc[mask, feature_name] = final_dists
        
        # Cleanup: Replace remaining Infs with NaN
        # (Though mask logic usually prevents this, safeguarding for ML)
        merged_df.loc[merged_df[feature_name] == np.inf, feature_name] = np.nan
        
    return merged_df
