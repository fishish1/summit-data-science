import pandas as pd
import json
import os
from pathlib import Path

def get_export_paths():
    """
    Determine where to export data files.
    
    Returns a list of export directory paths.
    1. Always includes local 'static_dashboard/data'
    2. Includes sibling repo path if it exists
    """
    paths = []
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # /path/to/summit
    
    # 1. Internal Static Dashboard (Guaranteed)
    internal_path = project_root / "static_dashboard/data"
    internal_path.mkdir(parents=True, exist_ok=True)
    paths.append(internal_path)
    
    # 2. Sibling Repo (website)
    sibling_path = project_root.parent / "brian.fishman.info/public/projects/summit/data"
    if sibling_path.exists():
        paths.append(sibling_path)
    
    return paths

def convert_mortgage():
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / 'data/mortgage_rate.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    # Average by year
    annual = df.groupby('year')['value'].mean().reset_index()
    
    # Filter to 1980 onwards to match market trends
    annual = annual[annual['year'] >= 1980]
    
    result = annual.to_dict(orient='records')
    
    # Export to all paths
    export_paths = get_export_paths()
    for dest_path in export_paths:
        output_file = dest_path / 'mortgage_history.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Exported mortgage history to {output_file}")

if __name__ == "__main__":
    convert_mortgage()
