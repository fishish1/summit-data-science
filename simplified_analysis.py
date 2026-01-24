import pandas as pd
import os

def load_and_process_data(records_path, owner_data_path, codes_path):
    """
    Loads, cleans, and merges all data sources for analysis.
    """
    # Load records data
    try:
        records_df = pd.read_csv(records_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: The records file was not found at {records_path}")
        return None

    # Clean and calculate 'sqft'
    records_df['groundflrs'] = pd.to_numeric(records_df['groundflrs'], errors='coerce')
    records_df['gnd_sqft'] = pd.to_numeric(records_df['gnd_sqft'], errors='coerce')
    records_df['sqft'] = records_df['groundflrs'] * records_df['gnd_sqft']

    # Load and merge owner data if it exists
    if os.path.exists(owner_data_path):
        owner_df = pd.read_csv(owner_data_path)
        records_df = pd.merge(records_df, owner_df, on="acct", how="left")
    else:
        print(f"Warning: Owner data not found at {owner_data_path}. Proceeding without it.")

    # Load and apply building type mapping from abstract codes
    if os.path.exists(codes_path):
        try:
            df_codes = pd.read_csv(codes_path)
            df_codes['code'] = pd.to_numeric(df_codes['code'], errors='coerce')
            df_codes.dropna(subset=['code'], inplace=True)
            df_codes['code'] = df_codes['code'].astype(int)
            code_map = df_codes.set_index('code')['building_type'].to_dict()
            
            # Apply the mapping
            records_df['abstract_code'] = pd.to_numeric(records_df['abstract_code'], errors='coerce')
            records_df['building_type'] = records_df['abstract_code'].map(code_map)
        except Exception as e:
            print(f"Warning: Could not process building codes from {codes_path}. Error: {e}")
    else:
        print(f"Warning: Abstract codes file not found at {codes_path}. Cannot map building types.")

    return records_df

def main():
    """
    Main function to run the simplified data analysis pipeline.
    """
    # Define file paths
    records_file = "records.csv"
    owner_data_file = "owner_data.csv"
    codes_file = "data/AbstractCodes.csv"

    # Load and process data
    processed_df = load_and_process_data(records_file, owner_data_file, codes_file)

    if processed_df is not None:
        print("Data processing complete.")
        
        # --- Perform Analysis ---
        # Example: Calculate and print the total square footage by building type
        if 'building_type' in processed_df.columns and 'sqft' in processed_df.columns:
            sqft_by_type = processed_df.groupby('building_type')['sqft'].sum().sort_values(ascending=False)
            print("\nTotal Square Footage by Building Type:")
            print(sqft_by_type)
        else:
            print("\nCould not perform analysis due to missing 'building_type' or 'sqft' columns.")

        # You can add more analysis here.
        # For example, showing the top 5 rows of the final dataframe:
        print("\nTop 5 rows of processed data:")
        print(processed_df.head())

if __name__ == "__main__":
    main()
