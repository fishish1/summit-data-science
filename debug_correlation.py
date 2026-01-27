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

# 2. Load S&P Data
df_sp = pd.read_csv("data/sp500_annual.csv")

# 3. Merge
merged = pd.merge(df_pivot, df_sp, on='year', how='inner')

# 4. Correlations
corr_return = merged['oos_pct'].corr(merged['return_pct'])
corr_value = merged['oos_pct'].corr(merged['avg_value'])

print(f"Correlation (OOS % vs S&P 500 Return): {corr_return:.4f}")
print(f"Correlation (OOS % vs S&P 500 Value): {corr_value:.4f}")

print("\nRecent Data:")
print(merged[['year', 'oos_pct', 'return_pct']].tail(10))
