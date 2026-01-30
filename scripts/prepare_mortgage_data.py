import pandas as pd
import json
import os

def convert_mortgage():
    csv_path = '/Users/brian/Documents/summit/data/mortgage_rate.csv'
    dest_path = '/Users/brian/Documents/summit/static_dashboard/data/mortgage_history.json'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Average by year
    annual = df.groupby('year')['value'].mean().reset_index()
    
    # Filter to 1980 onwards to match market trends
    annual = annual[annual['year'] >= 1980]
    
    result = annual.to_dict(orient='records')
    
    with open(dest_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Exported to {dest_path}")

if __name__ == "__main__":
    convert_mortgage()
