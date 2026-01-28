import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

try:
    from summit_housing.geo import enrich_with_geo_features, load_address_points
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

try:
    print("Testing load_address_points...")
    geo_df = load_address_points(".")
    print(f"Loaded {len(geo_df)} address points")
    print(geo_df.head())
except Exception as e:
    print(f"load_address_points failed: {e}")
    # print traceback
    import traceback
    traceback.print_exc()

# Create dummy sales df
sales_df = pd.DataFrame({'schno': ['100001', '100008', '999999']})
try:
    print("Testing enrich_with_geo_features...")
    enriched = enrich_with_geo_features(sales_df, ".")
    print("Enrichment successful")
    print(enriched.columns)
    print(enriched.head())
except Exception as e:
    print(f"enrich_with_geo_features failed: {e}")
    import traceback
    traceback.print_exc()
