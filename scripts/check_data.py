
import pandas as pd
import sys

try:
    df = pd.read_csv("records.csv", low_memory=False)
    
    total_rows = len(df)
    print(f"Total Records: {total_rows}")
    
    sales_cols = [
        ('1', 'recdate1', 'docfee1'),
        ('2', 'recdate2', 'docfee2'),
        ('3', 'recdate3', 'docfee3'),
        ('4', 'recdate4', 'docfee4')
    ]
    
    print("\n--- Sales Data Availability ---")
    for idx, date_col, fee_col in sales_cols:
        if date_col in df.columns and fee_col in df.columns:
            date_count = df[date_col].notna().sum()
            fee_count = df[fee_col].notna().sum()
            
            # Check for non-zero fees (indicating a likely sale price)
            # CO doc fee is typically $1 per $10,000, so 0.00 usually means no transfer tax paid (e.g. quit claim)
            market_sales = df[pd.to_numeric(df[fee_col], errors='coerce') > 0].shape[0]
            
            print(f"Slot {idx}:")
            print(f"  - Dates Present: {date_count} ({date_count/total_rows:.1%})")
            print(f"  - Fees Present:  {fee_count}")
            print(f"  - Market Sales (> $0 fee): {market_sales} ({market_sales/total_rows:.1%})")
        else:
            print(f"Slot {idx}: Columns not found!")

    # Check for multiple sales (needed for LAG analysis)
    # We need rows that have valid dates in consecutive slots, or at least multiple slots
    
    # Simple check: how many rows have at least 2 market sales?
    def count_market_sales(row):
        count = 0
        for _, _, fee_col in sales_cols:
            try:
                val = float(row[fee_col])
                if val > 0:
                    count += 1
            except:
                pass
        return count

    df['market_sale_count'] = df.apply(count_market_sales, axis=1)
    multi_sale_count = len(df[df['market_sale_count'] > 1])
    
    print(f"\nProperties with Multiple Market Sales (Good for LAG analysis):")
    print(f"  - Count: {multi_sale_count}")
    print(f"  - Percentage: {multi_sale_count/total_rows:.1%}")

except FileNotFoundError:
    print("Error: records.csv not found.")
