from summit_housing.queries import MarketAnalytics
import pandas as pd
import numpy as np

analytics = MarketAnalytics()
df_owners = analytics.get_owner_purchase_trends()

# 1. Calculate OOS Share per year
df_pivot = df_owners.pivot_table(index='purchase_year', columns='location_type', values='buyer_count', aggfunc='sum').fillna(0)
df_pivot['total'] = df_pivot.sum(axis=1)
df_pivot['oos_pct'] = (df_pivot['Out-of-State'] / df_pivot['total']) * 100
df_pivot.reset_index(inplace=True)
df_pivot['year'] = pd.to_numeric(df_pivot['purchase_year'])

# 2. Load Mortgage Data
df_nav = pd.read_csv("data/mortgage_rate.csv")
df_nav['year'] = pd.to_datetime(df_nav['date']).dt.year
df_rate = df_nav.groupby('year')['value'].mean().reset_index()

# 3. Merge
merged = pd.merge(df_pivot, df_rate, on='year', how='inner')

# 4. Correlations
corr_rate = merged['oos_pct'].corr(merged['value'])

print(f"Correlation (OOS % vs Mortgage Rate): {corr_rate:.4f}")
print(merged[['year', 'oos_pct', 'value']].tail(10))
