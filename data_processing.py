import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
FRED_API_KEY = os.getenv("FRED_KEY")

# FRED API details
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES_IDS = {
    "summit_pop": "COSUMM7POP",
    "us_pop": "POPTOTUSA647NWDB",
    "us_housing": "ETOTALUSQ176N",
}

DATA_DIR = "/Users/brian/Documents/summit/"


def fetch_fred_data(series_id, file_path):
    """Fetches data from FRED API and saves it to a CSV file."""
    if not FRED_API_KEY:
        print("FRED API key not found. Please set it in your .env file.")
        return

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "1970-01-01",
    }
    try:
        response = requests.get(FRED_API_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()["observations"]
        df = pd.DataFrame(data)
        df = df[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        # Convert value to numeric, coercing errors
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.to_csv(file_path, index=False)
        print(f"Successfully fetched and saved data for {series_id} to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {series_id}: {e}")
    except KeyError:
        print(f"Unexpected JSON structure for {series_id}.")


def load_and_prepare_data():
    """Loads, cleans, merges, and prepares the property data for analysis."""
    # --- Data Download ---
    for name, series_id in SERIES_IDS.items():
        file_path = os.path.join(DATA_DIR, f"{name}.csv")
        fetch_fred_data(series_id, file_path)

    # --- Data Merging and Cleaning ---
    print("Loading and merging data...")
    try:
        records_df = pd.read_csv(
            os.path.join(DATA_DIR, "records.csv"), low_memory=False
        )
        owner_df = pd.read_csv(os.path.join(DATA_DIR, "owner_data.csv"), low_memory=False)
        abstract_codes_df = pd.read_csv(
            os.path.join(DATA_DIR, "data/AbstractCodes.csv"), low_memory=False
        )
    except FileNotFoundError as e:
        print(f"Error: Missing required data file - {e}. Please run the script again.")
        return None

    # Merge records with owner data
    if "schno" in records_df.columns and "Schno" in owner_df.columns:
        records_df["schno"] = records_df["schno"].astype(str)
        owner_df["Schno"] = owner_df["Schno"].astype(str)
        records_df = pd.merge(
            records_df, owner_df, left_on="schno", right_on="Schno", how="left"
        )
        records_df.drop("Schno", axis=1, inplace=True)

    # --- Data Cleaning ---
    sqft_sum_cols = [
        "first_sqft", "second_sqft", "third_sqft", "addn",
        "fin_half", "fin_bsmt", "unfin_bsmt"
    ]
    for col in sqft_sum_cols + ["sfla", "sqft"]:
        if col in records_df.columns:
            records_df[col] = pd.to_numeric(records_df[col], errors='coerce').fillna(0)
        else:
            records_df[col] = 0

    records_df['sqft_sum'] = records_df[sqft_sum_cols].sum(axis=1)
    records_df['sfla'] = records_df.apply(
        lambda row: row['sfla'] if row['sfla'] > 0 else (row['sqft'] if row['sqft'] > 0 else row['sqft_sum']),
        axis=1
    )
    records_df.drop(columns=['sqft_sum'], inplace=True)

    if 'units' in records_df.columns:
        records_df['units'] = pd.to_numeric(records_df['units'], errors='coerce').fillna(1)
        records_df['units'] = records_df['units'].apply(lambda x: int(max(x, 1)))
    else:
        records_df['units'] = 1

    records_df['sfla'] = records_df['sfla'] / records_df['units']
    records_df = records_df.loc[records_df.index.repeat(records_df['units'])].reset_index(drop=True)
    records_df = records_df[records_df["sfla"] > 0].copy()

    # --- Feature Engineering ---
    print("Engineering features...")
    records_df["BuildingType"] = records_df.apply(
        _map_building_type_from_abstract, args=(abstract_codes_df,), axis=1
    )

    summit_cities = [
        "BRECKENRIDGE", "FRISCO", "DILLON", "SILVERTHORNE",
        "KEYSTONE", "COPPER MOUNTAIN", "HEENEY", "BLUE RIVER",
    ]
    records_df["city"] = records_df["city"].str.upper().str.strip()
    records_df["state"] = records_df["state"].str.upper().str.strip()
    records_df["LocationType"] = records_df.apply(_categorize_location, axis=1)

    print("Data preparation complete.")
    return records_df


def _map_building_type_from_abstract(row, abstract_codes_df):
    """
    Maps a property to a building type based on its abstract codes.
    Prioritizes 'Improvement' codes over 'Land' codes.
    """
    abstract_cols = ["abst1", "abst2", "abst3", "abst4", "more_abst"]

    # Get non-null abstract codes for the current property
    row_abstracts = [str(row[col]) for col in abstract_cols if pd.notna(row[col])]

    if not row_abstracts:
        return "Unknown"

    # Filter the main abstract codes dataframe for codes relevant to this row
    relevant_codes = abstract_codes_df[
        abstract_codes_df["abstract_code"].isin(row_abstracts)
    ]

    if relevant_codes.empty:
        return "Unknown"

    # Prioritize "Improvement" codes
    improvements = relevant_codes[relevant_codes["value_type"] == "Improvement"]
    if not improvements.empty:
        # Return the description of the first matched improvement code
        return improvements.iloc[0]["description"]

    # Fallback to "Land" codes
    lands = relevant_codes[relevant_codes["value_type"] == "Land"]
    if not lands.empty:
        # Return the description of the first matched land code
        return lands.iloc[0]["description"]

    return "Unknown"


def _categorize_location(row):
    """Categorizes a property based on its location."""
    summit_cities = [
        "BRECKENRIDGE", "FRISCO", "DILLON", "SILVERTHORNE",
        "KEYSTONE", "COPPER MOUNTAIN", "HEENEY", "BLUE RIVER",
    ]
    if pd.isna(row['city']) or pd.isna(row['state']):
        return "Out of State"
    if row["city"] in summit_cities and row["state"] == "CO":
        return "In-County"
    elif row["state"] == "CO":
        return "In-State (Out of County)"
    else:
        return "Out of State"
