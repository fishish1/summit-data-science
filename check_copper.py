
import pandas as pd
import numpy as np
from summit_housing.queries import MarketAnalytics
from summit_housing.geo import enrich_with_geo_features

try:
    print("Loading data...")
    analytics = MarketAnalytics()
    df = analytics.get_training_data()
    print(f"Initial shape: {df.shape}")
    
    print("Enriching...")
    df = enrich_with_geo_features(df)
    
    print("Columns:", [c for c in df.columns if 'dist_' in c])
    
    print("Unique Cities:", df['city'].unique())

    # Check Copper specifically
    copper_mask = df['city'] == 'COPPERMOUNTAIN'
    if 'dist_copper' in df.columns:
        copper_df = df[copper_mask]
        print("\n--- COPPER/COUNTY Analysis ---")
        print(f"Total Rows: {len(copper_df)}")
        print("dist_copper stats:")
        print(copper_df['dist_copper'].describe())
        
        # Calculate dist_to_lift manually
        lift_cols = ['dist_breck', 'dist_keystone', 'dist_copper', 'dist_abasin']
        valid_lifts = [c for c in lift_cols if c in df.columns]
        df['dist_to_lift'] = df[valid_lifts].min(axis=1)
        
        print("\ndist_to_lift stats for Copper/County:")
        print(df.loc[copper_mask, 'dist_to_lift'].describe())
        
        # Check for outliers
        far_copper = df[copper_mask & (df['dist_to_lift'] > 10)]
        print(f"\nProperties named Copper but >10km from lift: {len(far_copper)}")
        if len(far_copper) > 0:
            print(far_copper[['FullAddress', 'city', 'dist_to_lift', 'LATITUDE', 'LONGITUDE']].head())
            
    else:
        print("dist_copper column MISSING!")

except Exception as e:
    print(f"Error: {e}")
