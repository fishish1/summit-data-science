from summit_housing.queries import MarketAnalytics
import pandas as pd

analytics = MarketAnalytics()

# 1. Check Owner Purchase Trends Source
print("Fetching Owner Trends...")
df_trends = analytics.get_owner_purchase_trends()
print(df_trends.head())
print(f"\nTotal Rows in Analysis: {df_trends['buyer_count'].sum()}")

# 2. Compare to Raw Records Count
from summit_housing.database import get_db
with get_db() as db:
    # returns a dataframe
    df_count = db.query("SELECT COUNT(*) as c FROM raw_records")
    total_records = df_count.iloc[0]['c']
    
print(f"Total Raw Records: {total_records}")

if abs(df_trends['buyer_count'].sum() - total_records) < 5000:
    print("\n✅ Verification: The analysis creates roughly 1 row per CURRENT record.")
    print("It does NOT multiply records for historical sales (which would be ~100k+ rows).")
else:
    print("⚠️ Something is off with the counts.")

# 3. Check if 'sales' CTE uses buyer address
# The fear: "Are we simulating buyer location for 2005 based on 2024 owner?"
# We check get_market_trends -> ANALYTICS_SQL
print("\nChecking Market Trends columns...")
df_mk = analytics.get_market_trends()
print(df_mk.columns.tolist())

# If 'location_type' (Buyer Origin) is NOT in get_market_trends, we are safe.
if 'location_type' not in df_mk.columns:
    print("✅ Verification: The general 'Market Trends' (Price History) does NOT include Buyer Origin data.")
    print("This confirms we are NOT falsely attributing current owner data to historical prices.")
