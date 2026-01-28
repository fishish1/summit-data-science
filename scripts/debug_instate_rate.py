from summit_housing.queries import MarketAnalytics
import pandas as pd
import numpy as np

# 1. Get Owner Data
analytics = MarketAnalytics()
df_owners = analytics.get_owner_purchase_trends()

# 2. Pivot to get Shares
df_pivot = df_owners.pivot_table(index='purchase_year', columns='location_type', values='buyer_count', aggfunc='sum').fillna(0)
df_pivot.reset_index(inplace=True)
df_pivot['year'] = pd.to_numeric(df_pivot['purchase_year'])
df_pivot['Total Volume'] = df_pivot.sum(axis=1, numeric_only=True) - df_pivot['year']
df_pivot['In-State Share (%)'] = (df_pivot['In-State (Non-Local)'] / df_pivot['Total Volume']) * 100

# 3. Get Mortgage Rates
rates = pd.read_csv("data/mortgage_rate.csv")
rates['year'] = pd.to_datetime(rates['date']).dt.year
rates_annual = rates.groupby('year')['value'].mean().reset_index(name='Mortgage Rate')

# 4. Calculate Rate Acceleration (Delta)
rates_annual['Rate Delta'] = rates_annual['Mortgage Rate'].diff()

# 5. Merge
df = pd.merge(df_pivot, rates_annual, on='year', how='inner')

# 6. Correlations
corr_absolute = df['In-State Share (%)'].corr(df['Mortgage Rate'])
corr_delta = df['In-State Share (%)'].corr(df['Rate Delta'])

print(f"Correlation: In-State Share vs Absolute Rate: {corr_absolute:.4f}")
print(f"Correlation: In-State Share vs Rate Change: {corr_delta:.4f}")

print("\nRecent Data (Year, In-State %, Rate, Delta):")
print(df[['year', 'In-State Share (%)', 'Mortgage Rate', 'Rate Delta']].tail(10))
