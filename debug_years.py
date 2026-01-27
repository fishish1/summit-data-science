from summit_housing.queries import MarketAnalytics
import pandas as pd

analytics = MarketAnalytics()
try:
    df = analytics.get_market_trends(exclude_multiunit=True)
    if not df.empty:
        print(f"Available years: {sorted(df['tx_year'].unique())}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.tail())
    else:
        print("Dataframe is empty")
except Exception as e:
    print(f"Error: {e}")
