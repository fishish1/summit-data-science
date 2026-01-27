from summit_housing.queries import MarketAnalytics
import pandas as pd

# Load data
analytics = MarketAnalytics()
df = analytics.get_training_data()

print("Columns:", df.columns.tolist())
if 'city' in df.columns:
    print("Unique Cities:", df['city'].unique())
    
# Check distance columns
dist_cols = [c for c in df.columns if 'dist_' in c]
print("Distance Columns:", dist_cols)

if 'city' in df.columns and dist_cols:
    print("\nSample Distance Map:")
    print(df.groupby('city')[dist_cols].mean())
