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

# Calculate Shares
df_pivot['In-State Share'] = (df_pivot['In-State (Non-Local)'] / df_pivot['Total Volume']) * 100
df_pivot['Local Share'] = (df_pivot['Local (In-County)'] / df_pivot['Total Volume']) * 100
df_pivot['OOS Share'] = (df_pivot['Out-of-State'] / df_pivot['Total Volume']) * 100

# 3. Get Mortgage Rates
rates = pd.read_csv("data/mortgage_rate.csv")
rates['year'] = pd.to_datetime(rates['date']).dt.year
rates_annual = rates.groupby('year')['value'].mean().reset_index(name='Mortgage Rate')

# 4. Merge
df = pd.merge(df_pivot, rates_annual, on='year', how='inner')

# 5. Correlations
corr_local = df['Local Share'].corr(df['Mortgage Rate'])
corr_instate = df['In-State Share'].corr(df['Mortgage Rate'])
corr_oos = df['OOS Share'].corr(df['Mortgage Rate'])

print("\n--- HYPOTHESIS CHECK ---")
print(f"1. Local vs Rates Correlation:       {corr_local:.4f}")
print(f"2. In-State vs Rates Correlation:    {corr_instate:.4f}")
print(f"3. Out-of-State vs Rates Correlation: {corr_oos:.4f}")

# Check 3-year rolling correlations to see stability
# df['Local_Rolling'] = df['Local Share'].rolling(3).corr(df['Mortgage Rate'])
# print("\nRolling Correlation Check (Last 5 years):")
# print(df.tail(5))
