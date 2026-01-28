import pandas as pd
import numpy as np
import os
from summit_housing.queries import MarketAnalytics
from summit_housing.geo import enrich_with_geo_features

def load_and_prep_data():
    """
    Fetches sales data and joins with local Macro-Economic indicators.
    """
    analytics = MarketAnalytics()
    df = analytics.get_training_data()
    try:
        df = enrich_with_geo_features(df)
    except Exception as e:
        print(f"⚠️ Geospatial Enrichment Failed: {e}")
    
    df['tx_date'] = pd.to_datetime(df['tx_date'])
    df['year'] = df['tx_date'].dt.year
    df['month'] = df['tx_date'].dt.month
    
    macro_files = {
        'mortgage_rate': 'data/mortgage_rate.csv',
        'cpi': 'data/cpi.csv',
        'sp500': 'data/sp500.csv',
        'summit_pop': 'data/summit_pop.csv'
    }
    
    macro_dfs = []
    for name, path in macro_files.items():
        if os.path.exists(path):
            macro_df = pd.read_csv(path)
            macro_df['date'] = pd.to_datetime(macro_df['date'])
            macro_df = macro_df.sort_values('date').rename(columns={'value': name})
            macro_dfs.append(macro_df[['date', name]])
    
    df = df.sort_values('tx_date')
    for m_df in macro_dfs:
        df = pd.merge_asof(df, m_df, left_on='tx_date', right_on='date', direction='backward')
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
            
    df = df.ffill().bfill()
    df['garage_size'] = df['garage_size'].fillna(0)
    df['year_blt'] = df['year_blt'].fillna(df['year_blt'].median())
    df['beds'] = df['beds'].fillna(0)
    df['baths'] = df['baths'].fillna(0)
    df['acres'] = df['acres'].fillna(0)
    
    quality_map = {'A': 6, 'X': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1}
    if 'grade' in df.columns:
        df['grade_numeric'] = df['grade'].map(quality_map).fillna(3)
    if 'cond' in df.columns:
        df['cond_numeric'] = df['cond'].map(quality_map).fillna(3)
    if 'scenic_view' in df.columns:
        df['scenic_view'] = df['scenic_view'].fillna(0)
    
    return df
