import os
import pandas as pd
import requests
from dotenv import load_dotenv
from utils import map_building_class_to_type

# Load environment variables from .env file
load_dotenv()
FRED_API_KEY = os.getenv("FRED_KEY")

# FRED API details
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES_IDS = {
    "summit_pop": "COSUMM7POP",
    "us_pop": "POPTOTUSA647NWDB",
    "us_housing": "ETOTALUSQ176N",
    "fed_rate": "FEDFUNDS",
    "mortgage_rate": "MORTGAGE30US",
    "cpi": "CPIAUCSL",
    "sp500": "SP500",
    "summit_inventory": "ACTLISCOU08117"  # Realtor.com Active Listings
}

DATA_DIR = "data"


def fetch_fred_data(series_id, file_path):
    """Fetches data from FRED API and saves it to a CSV file."""
    if not FRED_API_KEY:
        print("FRED API key not found. Please set it in your .env file.")
        return

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
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


def calculate_historical_supply(df):
    """Calculates historical supply and value metrics."""
    # Local Housing Supply Timeline (Cumulative)
    supply_timeline = (
        df.groupby("year_blt").size().cumsum().reset_index(name="cumulative_units")
    )

    # Total Residential Square Footage (Cumulative)
    sqft_timeline = (
        df.groupby("year_blt")["sfla"]
        .sum()
        .cumsum()
        .reset_index(name="cumulative_sqft")
    )

    return supply_timeline, sqft_timeline


def analyze_owner_location_and_distribution(df):
    """Analyzes owner mailing address locations and property distribution."""
    # --- Location Analysis ---
    # The 'LocationType' column is now created in main() before this function is called.
    if "LocationType" not in df.columns:
        print("\nWarning: 'LocationType' column not found. Skipping location summary.")
        location_summary = pd.DataFrame(columns=["LocationType", "Percentage"])
    else:
        location_summary = (
            df["LocationType"].value_counts(normalize=True).mul(100).reset_index()
        )
        location_summary.columns = ["LocationType", "Percentage"]

    # --- Property Distribution Analysis ---
    # This now uses the 'Primary Owner' field from the merged owner_data.csv
    # It's more accurate than using mailing address as a proxy for owner identity.
    if "Primary Owner" in df.columns and "OwnerType" in df.columns:
        df_owned = df.dropna(subset=["Primary Owner"])

        # Group by owner to count properties per owner, preserving OwnerType
        properties_per_owner = df_owned.groupby("Primary Owner").agg(
            NumberOfPropertiesOwned=("schno", "size"),
            OwnerType=("OwnerType", "first"),  # Get the owner type for each owner
            LocationType=("LocationType", "first"),  # Get the location type for each owner
        ).reset_index()

        # --- Categorize into 1, 2, and 3+ properties ---
        def categorize_ownership(n):
            if n == 1:
                return "1 Property"
            elif n == 2:
                return "2 Properties"
            else:
                return "3+ Properties"

        properties_per_owner["OwnershipCategory"] = properties_per_owner[
            "NumberOfPropertiesOwned"
        ].apply(categorize_ownership)

        # --- Print schedule numbers for owners with > 10 properties ---
        owners_with_more_than_10 = properties_per_owner[
            properties_per_owner["NumberOfPropertiesOwned"] > 10
        ]
        if not owners_with_more_than_10.empty:
            print("\n--- Owners with more than 10 properties ---")
            owner_names = owners_with_more_than_10.index
            multi_owner_records = df[df["Primary Owner"].isin(owner_names)]

            for name, group in multi_owner_records.groupby("Primary Owner"):
                count = len(group)
                schnos = ", ".join(group["schno"].astype(str).tolist())
                print(f"\nOwner (owns {count} properties): {name}")
                print(f"  Schedule Numbers: {schnos}")
            print("-------------------------------------------")

        # Now, group by the number of properties and owner type to get the count of owners
        owner_distribution = (
            properties_per_owner.groupby(["OwnershipCategory", "OwnerType"])
            .size()
            .reset_index(name="NumberOfOwners")
        )

        # --- New: Create distribution by location ---
        owner_distribution_by_location = (
            properties_per_owner.groupby(["OwnershipCategory", "LocationType"])
            .size()
            .reset_index(name="NumberOfOwners")
        )

        # --- New: Breakdown of multi-property owners by type ---
        multi_property_owners = properties_per_owner[
            properties_per_owner["NumberOfPropertiesOwned"] > 1
        ]
        multi_owner_type_breakdown = (
            multi_property_owners["OwnerType"].value_counts().reset_index()
        )
        multi_owner_type_breakdown.columns = ["OwnerType", "NumberOfOwners"]

    else:
        print(
            "\nWarning: 'Primary Owner' or 'OwnerType' column not found. Skipping property distribution analysis."
        )
        # Create an empty dataframe to avoid errors downstream
        owner_distribution = pd.DataFrame(
            columns=["OwnershipCategory", "OwnerType", "NumberOfOwners"]
        )
        owner_distribution_by_location = pd.DataFrame(
            columns=["OwnershipCategory", "LocationType", "NumberOfOwners"]
        )
        multi_owner_type_breakdown = pd.DataFrame(
            columns=["OwnerType", "NumberOfOwners"]
        )

    return (
        location_summary,
        owner_distribution,
        owner_distribution_by_location,
        multi_owner_type_breakdown,
    )


def analyze_owner_language(df):
    """Analyzes the primary owner's name to categorize them."""
    if "Primary Owner" not in df.columns:
        print("\nWarning: 'Primary Owner' column not found. Skipping owner language analysis.")
        return df, None  # Return the original df and None for the summary

    def get_owner_type(name):
        if pd.isna(name):
            return "Unknown"
        name = name.upper()
        if "LLC" in name or "LIMITED LIABILITY" in name:
            return "LLC / Limited Liability"
        if "TRUST" in name or "TR" in name or "TRUSTEE" in name:
            return "Trust"
        if "INC" in name or "CORP" in name or "INCORPORATED" in name or "LTD" in name:
            return "Corporation"
        if "PARTNERS" in name or "LP" in name or "LLP" in name:
            return "Partnership"
        if "CHURCH" in name or "DIOCESE" in name or "MINISTRIES" in name or "TEMPLE" in name or "SYNAGOGUE" in name or "MOSQUE" in name:
            return "Church / Religious"
        return "Individual"

    df["OwnerType"] = df["Primary Owner"].apply(get_owner_type)

    owner_type_summary = (
        df["OwnerType"].value_counts(normalize=True).mul(100).reset_index()
    )
    owner_type_summary.columns = ["OwnerType", "Percentage"]
    return df, owner_type_summary


def load_abstract_code_map():
    """
    Loads AbstractCodes.csv and creates a mapping to building types,
    including value type and priority.
    """
    try:
        df_codes = pd.read_csv("data/AbstractCodes.csv")
        # Ensure 'code' is numeric, coercing errors to NaN which will be handled
        df_codes["code_num"] = pd.to_numeric(df_codes["code"], errors="coerce")
        df_codes["description"] = df_codes["description"].str.upper()
        # Standardize 'value type' column
        if "value type" in df_codes.columns:
            df_codes["value_type"] = df_codes["value type"].str.strip().str.title()
        else:
            print("Warning: 'value type' column not found in AbstractCodes.csv. Categorization may be inaccurate.")
            df_codes["value_type"] = "Unknown"

    except FileNotFoundError:
        print(
            "Warning: data/AbstractCodes.csv not found. Cannot determine building types."
        )
        return None

    code_map = {}
    for _, row in df_codes.iterrows():
        desc = row["description"]
        code = row["code"]
        code_num = row["code_num"]
        value_type = row["value_type"]
        cat = "Other"  # Default category

        # Vacant Land - highest priority check
        if "VACANT" in desc:
            cat = "Vacant Land"
        
        # Residential Properties (1000-1999 range)
        elif pd.notna(code_num) and 1000 <= code_num < 2000:
            if any(term in desc for term in ["SINGLE FAMILY", "SFR"]):
                cat = "Single Family Home"
            elif any(term in desc for term in ["CONDO", "CONDOMINIUM"]):
                cat = "Apartment/Condo/Townhome"
            elif any(term in desc for term in ["TOWNHOUSE", "TOWNHOME"]):
                cat = "Apartment/Condo/Townhome"
            elif any(term in desc for term in ["MULTI-UNIT", "DUPLEX", "TRIPLEX", "2 & 3 UNIT", "4-8", "9 &"]):
                cat = "Apartment/Condo/Townhome"
            elif any(term in desc for term in ["MOBILE HOME"]):
                cat = "Single Family Home"  # Mobile homes are typically single family
            elif "RESIDENTIAL" in desc:
                cat = "Single Family Home"  # Default residential to single family
            else:
                cat = "Single Family Home"  # Fallback for 1000s range
        
        # Commercial Properties (2000-2999 range)
        elif pd.notna(code_num) and 2000 <= code_num < 3000:
            if any(term in desc for term in ["LODGING", "BED & BREAKFAST"]):
                cat = "Hotel/Lodging"
            elif any(term in desc for term in ["MERCHANDISING", "OFFICE", "WAREHOUSE", "STORAGE"]):
                cat = "Commercial"
            elif any(term in desc for term in ["RECREATION", "ENTERTAINMENT"]):
                cat = "Recreation/Entertainment"
            elif "AIRPORT" in desc:
                cat = "Transportation/Infrastructure"
            else:
                cat = "Commercial"  # Default for 2000s range
        
        # Industrial Properties (3000-3999 range)
        elif pd.notna(code_num) and 3000 <= code_num < 4000:
            cat = "Industrial"
        
        # Agricultural Properties (4000-4999 range)
        elif pd.notna(code_num) and 4000 <= code_num < 5000:
            if any(term in desc for term in ["RESIDENCE", "MOBILE HOMES"]):
                cat = "Single Family Home"  # Agricultural residences
            else:
                cat = "Agricultural"
        
        # Mining/Natural Resources (5000-5999 range)
        elif pd.notna(code_num) and 5000 <= code_num < 6000:
            cat = "Mining/Natural Resources"
        
        # Oil/Gas Properties (7000-7999 range)
        elif pd.notna(code_num) and 7000 <= code_num < 8000:
            cat = "Oil/Gas"
        
        # Utilities/Transportation (8000-8999 range)
        elif pd.notna(code_num) and 8000 <= code_num < 9000:
            if any(term in desc for term in ["RAILROAD", "AIRLINE", "PIPELINE"]):
                cat = "Transportation/Infrastructure"
            elif any(term in desc for term in ["ELECTRIC", "TELEPHONE", "GAS", "WATER"]):
                cat = "Utilities"
            else:
                cat = "Utilities"  # Default for 8000s range
        
        # Exempt Properties (9000+ range) - now separated into Religious, Government subcategories, and Public Lands
        elif pd.notna(code_num) and code_num >= 9000:
            if any(term in desc for term in ["RESIDENTIAL", "SFR", "MULTI RES", "PARSONAGE", "HOUSING"]):
                cat = "Single Family Home"  # Exempt residential properties
            elif any(term in desc for term in ["CHURCH", "RELIGIOUS", "CAMP/RETREAT", "CONVENT", "MONASTERY", "PARSONAGE"]):
                cat = "Religious"
            elif "CEMETERY" in desc or "CEMETARY" in desc:
                cat = "Cemetary"  # Cemeteries are typically religious
            elif any(term in desc for term in ["NATIONAL PARKS", "NATIONAL FOREST", "BUREAU OF LAND", "PARKS AND RECREATION", "WILDLIFE", "OPEN SPACE", "COMMON AREA"]):
                cat = "Public Lands"
            elif any(term in desc for term in ["SCHOOL", "COLLEGE", "EDUCATION", "ELEMENTARY", "SECONDARY", "VOCATIONAL"]):
                cat = "Schools/Education"
            elif any(term in desc for term in ["FEDERAL", "GENERAL SERVICE ADMIN", "MILITARY", "INDIAN"]):
                cat = "Federal Government"
            elif any(term in desc for term in ["STATE", "ADMINISTRATION", "HIGHWAY DEPARTMENT", "INSTITUTIONS", "LAND COMMISSION"]):
                cat = "State Government"
            elif any(term in desc for term in ["COUNTY", "ROAD AND BRIDGE", "TAX TITLE", "HOUSING AUTHORITY"]):
                cat = "County Government"
            elif any(term in desc for term in ["TOWN", "FIRE", "WATER", "SANITATION", "DRAINAGE", "IRRIGATION", "LIBRARY", "CHARITABLE"]):
                cat = "Local Government"
            elif any(term in desc for term in ["HEALTH CARE"]):
                cat = "Health Care"
            else:
                cat = "Government"  # Default for other exempt properties
        
        # Additional keyword-based categorization for edge cases
        elif any(term in desc for term in ["CHURCH", "RELIGIOUS", "CEMETERY", "CEMETARY"]):
            cat = "Religious"
        elif any(term in desc for term in ["NATIONAL PARKS", "NATIONAL FOREST", "PARKS", "WILDLIFE", "OPEN SPACE"]):
            cat = "Public Lands"
        elif any(term in desc for term in ["SCHOOL", "COLLEGE", "EDUCATION"]):
            cat = "Schools/Education"
        elif "FEDERAL" in desc:
            cat = "Federal Government"
        elif "STATE" in desc:
            cat = "State Government"
        elif "COUNTY" in desc:
            cat = "County Government"
        elif "TOWN" in desc:
            cat = "Local Government"
        elif "HOTEL" in desc or "MOTEL" in desc or "LODGE" in desc:
            cat = "Hotel/Lodging"
        elif "RECREATION" in desc or "ENTERTAINMENT" in desc:
            cat = "Recreation/Entertainment"

        # Set priority: Improvement > Land > Other
        priority = 0
        if value_type == "Improvement":
            priority = 1
        elif value_type == "Land":
            priority = 2
        else:
            priority = 3

        code_map[code] = {"type": cat, "value_type": value_type, "priority": priority}
    return code_map


def determine_building_type(row, code_map):
    """
    Determines building type from abstract codes, prioritizing 'Improvement' over 'Land'.
    """
    found_codes = []
    # Check all relevant abstract code columns, including 'more_abst'
    for col in ["abst1", "abst2", "abst3", "abst4", "more_abst"]:
        code = row[col]
        if pd.notna(code) and code in code_map:
            # Append the full code info dictionary
            found_codes.append(code_map[code])

    if not found_codes:
        return "Unknown"

    # Sort the found codes by priority (lower number is higher priority)
    found_codes.sort(key=lambda x: x["priority"])

    # The best match is the first one in the sorted list
    best_match = found_codes[0]

    # Return the building type of the best match
    return best_match["type"]


def analyze_buyer_characteristics_by_location(df):
    """Analyzes housing characteristics by owner location type."""
    if "LocationType" not in df.columns:
        print(
            "\nWarning: 'LocationType' column not found. Skipping buyer characteristics analysis."
        )
        return None

    # --- New: Filter for residential properties for this analysis ---
    residential_types = ["Single Family Home", "Apartment/Condo/Townhome"]
    residential_df = df[df["BuildingType"].isin(residential_types)].copy()
    # --- End of New Filter ---

    # Create boolean flags for analysis
    # Assuming 'garage_size' > 0 means there is a garage.
    residential_df["HasGarage"] = residential_df["garage_size"] > 0

    # --- New: Group by both LocationType and BuildingType for more granular analysis ---
    # Calculate metrics like AvgSqFt and PctWithGarage per building type within each location
    location_analysis = (
        residential_df.groupby(["LocationType", "BuildingType"])
        .agg(
            AvgSqFt=("sfla", "mean"),
            PctWithGarage=("HasGarage", lambda x: x.mean() * 100),
            UnitCount=("sfla", "size"),  # Get a count of units for weighting/info
        )
        .reset_index()
    )
    # --- End of New Grouping Logic ---

    return location_analysis


def analyze_owner_type_property_portfolio(df):
    """
    Analyzes the distribution of property types for each owner type.
    """
    print("Analyzing property portfolios by owner type...")
    # --- Use the pre-calculated BuildingType column ---
    residential_types = ["Single Family Home", "Apartment/Condo/Townhome"]
    residential_df = df[df["BuildingType"].isin(residential_types)].copy()

    # Count properties by owner type and building type
    portfolio_counts = (
        residential_df.groupby(["OwnerType", "BuildingType"])
        .size()
        .reset_index(name="PropertyCount")
    )

    # Calculate the total number of properties for each owner type
    owner_totals = portfolio_counts.groupby("OwnerType")["PropertyCount"].transform("sum")

    # Calculate the percentage
    portfolio_counts["Percentage"] = (portfolio_counts["PropertyCount"] / owner_totals) * 100

    return portfolio_counts


def analyze_out_of_state_owners(df):
    """Analyzes the geographic distribution of out-of-state owners."""
    if "LocationType" not in df.columns or "state" not in df.columns:
        print("\nWarning: Required columns not found for out-of-state owner analysis.")
        return None

    out_of_state_df = df[df["LocationType"] == "Out of State"].copy()
    state_counts = out_of_state_df["state"].value_counts().reset_index()
    state_counts.columns = ["State", "NumberOfProperties"]
    return state_counts


def analyze_property_composition(df):
    """
    Analyzes and summarizes the distribution of building types by unit count and square footage.
    """
    if "BuildingType" not in df.columns:
        print("Warning: 'BuildingType' column not found. Skipping building type analysis.")
        return None, None

    # Distribution by unit count
    building_type_summary_by_units = (
        df["BuildingType"].value_counts(normalize=True).mul(100).reset_index()
    )
    building_type_summary_by_units.columns = ["BuildingType", "Percentage"]

    # Distribution by square footage
    if "sfla" not in df.columns:
        print("Warning: 'sfla' column not found. Skipping sqft distribution analysis.")
        building_type_summary_by_sqft = None
    else:
        sqft_by_type = df.groupby("BuildingType")["sfla"].sum()
        total_sqft = sqft_by_type.sum()
        building_type_summary_by_sqft = (
            (sqft_by_type / total_sqft) * 100
        ).reset_index(name="Percentage")

    return building_type_summary_by_units, building_type_summary_by_sqft


def analyze_hotel_growth(df):
    """Analyzes the construction rate of hotels and lodging."""
    print("Analyzing hotel/lodging growth timeline...")
    # Use the pre-calculated BuildingType column
    if "BuildingType" not in df.columns:
        print("Warning: 'BuildingType' column not found. Skipping hotel growth analysis.")
        return None

    hotel_df = df[df["BuildingType"] == "Commercial"].copy()

    # A more specific filter for hotels/lodging based on description
    # This part is tricky as we don't have original descriptions if we only have BuildingType.
    # Assuming 'Commercial' is sufficient for now. If more detail is needed,
    # the original data source for BuildingType would need to be consulted.
    # hotel_df = hotel_df[
    #     hotel_df["BuildingType"].str.contains("Hotel|Lodging", case=False, na=False)
    # ]

    if hotel_df.empty:
        print("No commercial properties found to analyze for hotel growth.")
        return None

    # Ensure 'units' column exists and is numeric, fill NaNs with 1
    if "units" not in hotel_df.columns:
        hotel_df["units"] = 1
    else:
        hotel_df["units"] = pd.to_numeric(hotel_df["units"], errors="coerce").fillna(1)

    hotel_growth = (
        hotel_df.groupby("year_blt")
        .agg(new_units=("units", "sum"), new_sfla=("sfla", "sum"))
        .reset_index()
    )

    hotel_growth = hotel_growth.sort_values("year_blt")
    hotel_growth["cumulative_units"] = hotel_growth["new_units"].cumsum()
    hotel_growth["cumulative_sfla"] = hotel_growth["new_sfla"].cumsum()

    # Filter to start from 1970 for relevance
    hotel_growth = hotel_growth[hotel_growth["year_blt"] >= 1970].copy()

    return hotel_growth


def create_unified_dataset(records_df, owner_data_exists=True):
    """
    Creates a comprehensive unified dataset for dashboard cross-filtering.
    This combines all property records with calculated metrics and categorizations.
    """
    print("Creating unified dataset for dashboard...")
    
    # Start with the main records dataframe
    unified_df = records_df.copy()

    # --- Commercial Lodging SqFt Calculation ---
    # Define columns for fallback calculation and ensure they are numeric
    sqft_sum_cols = [
        "first_sqft", "second_sqft", "third_sqft", "addn",
        "fin_half", "fin_bsmt", "unfin_bsmt"
    ]
    # This calculation is moved here from main() to be part of the unified dataset only
    unified_df['sqft_sum'] = unified_df[sqft_sum_cols].sum(axis=1)

    def is_commercial_lodging(row):
        """Check if any abstract code is in the commercial range (2000s)."""
        for col in ["abst1", "abst2", "abst3", "abst4", "more_abst"]:
            code = row.get(col)
            if pd.notna(code):
                try:
                    code_num = int(float(code)) # Handle potential float strings like '2000.0'
                    if 2000 <= code_num < 3000:
                        return True
                except (ValueError, TypeError):
                    continue # Ignore non-numeric codes
        return False

    unified_df['commercial_lodging_sqft'] = unified_df.apply(
        lambda row: row['sqft_sum'] if row['units'] > 0 and is_commercial_lodging(row) else 0,
        axis=1
    )
    unified_df.drop(columns=['sqft_sum'], inplace=True)
    # --- End Commercial Lodging SqFt Calculation ---
    
    # Add derived metrics for properties
    if 'garage_size' in unified_df.columns:
        unified_df['HasGarage'] = unified_df['garage_size'] > 0
    else:
        unified_df['HasGarage'] = False
    
    # Add property age calculation
    current_year = 2024
    unified_df['PropertyAge'] = current_year - unified_df['year_blt']
    
    # Add size categories
    def categorize_size(sqft):
        if pd.isna(sqft) or sqft <= 0:
            return "Unknown"
        elif sqft < 1000:
            return "Small (< 1,000 sqft)"
        elif sqft < 2000:
            return "Medium (1,000-2,000 sqft)"
        elif sqft < 3000:
            return "Large (2,000-3,000 sqft)"
        else:
            return "Very Large (3,000+ sqft)"
    
    unified_df['SizeCategory'] = unified_df['sfla'].apply(categorize_size)
    
    # Add decade built categories
    def categorize_decade(year):
        if pd.isna(year):
            return "Unknown"
        decade = int(year // 10) * 10
        return f"{decade}s"
    
    unified_df['DecadeBuilt'] = unified_df['year_blt'].apply(categorize_decade)
    
    # Add ownership concentration metric (properties per owner)
    if owner_data_exists and 'Primary Owner' in unified_df.columns:
        owner_property_counts = unified_df.groupby('Primary Owner').size()
        unified_df['Owner Portfolio Size'] = unified_df['Primary Owner'].map(owner_property_counts)
        
        def categorize_ownership_level(count):
            if pd.isna(count):
                return "Unknown"
            elif count == 1:
                return "Single Property Owner"
            elif count <= 3:
                return "Small Portfolio (2-3)"
            elif count <= 10:
                return "Medium Portfolio (4-10)"
            else:
                return "Large Portfolio (10+)"
        
        unified_df['OwnershipLevel'] = unified_df['Owner Portfolio Size'].apply(categorize_ownership_level)
        
        # Create Owner Portfolio Size Category for display purposes only
        def categorize_portfolio_size(count):
            if pd.isna(count):
                return "Unknown"
            elif count == 1:
                return "1 Property"
            elif count == 2:
                return "2 Properties"
            elif count == 3:
                return "3 Properties"
            elif count <= 5:
                return "4-5 Properties"
            elif count <= 10:
                return "6-10 Properties"
            else:
                return "10+ Properties"
        
        unified_df['Owner Portfolio Category'] = unified_df['Owner Portfolio Size'].apply(categorize_portfolio_size)
    else:
        unified_df['Owner Portfolio Size'] = 1
        unified_df['OwnershipLevel'] = "Unknown"
        unified_df['Owner Portfolio Category'] = "1 Property"
    
    # Add residential filter flag
    residential_types = ["Single Family Home", "Apartment/Condo/Townhome"]
    unified_df['IsResidential'] = unified_df['BuildingType'].isin(residential_types)
    
    # Add value tier based on square footage (proxy for value)
    unified_df['ValueTier'] = pd.qcut(unified_df['sfla'], 
                                    q=4, 
                                    labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'],
                                    duplicates='drop')
    
    # Clean up and select relevant columns for dashboard
    dashboard_columns = [
        'schno', 'year_blt', 'sfla', 'BuildingType', 'LocationType', 'OwnerType',
        'HasGarage', 'PropertyAge', 'SizeCategory', 'DecadeBuilt', 'OwnershipLevel',
        'IsResidential', 'ValueTier', 'OwnerPropertyCount', 'commercial_lodging_sqft'
    ]
    
    # Add optional columns if they exist
    optional_columns = ['city', 'state', 'Primary Owner', 'garage_size', 'units']
    for col in optional_columns:
        if col in unified_df.columns:
            dashboard_columns.append(col)
    
    # Add "Owner Portfolio Size" column to dashboard columns if it exists
    if 'Owner Portfolio Size' in unified_df.columns:
        dashboard_columns.append('Owner Portfolio Size')
    
    # Add "Owner Portfolio Category" column to dashboard columns if it exists
    if 'Owner Portfolio Category' in unified_df.columns:
        dashboard_columns.append('Owner Portfolio Category')
    
    # Filter to only include columns that exist
    available_columns = [col for col in dashboard_columns if col in unified_df.columns]
    unified_df = unified_df[available_columns].copy()
    
    return unified_df


def main():
    """Main function to run the data analysis pipeline."""
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Step 1: Fetch FRED data
    for name, series_id in SERIES_IDS.items():
        file_path = os.path.join(DATA_DIR, f"{name}.csv")
        fetch_fred_data(series_id, file_path)

    # Step 2: Load and clean local records
    try:
        records_df = pd.read_csv("records.csv", low_memory=False)

        # --- Updated SqFt Calculation Logic for Three Metrics ---
        # Define columns for fallback calculation and ensure they are numeric
        sqft_sum_cols = [
            "first_sqft", "second_sqft", "third_sqft", "addn",
            "fin_half", "fin_bsmt", "unfin_bsmt"
        ]
        # Ensure all needed columns exist and are numeric
        for col in sqft_sum_cols + ["sfla", "sqft", "units"]:
            if col in records_df.columns:
                records_df[col] = pd.to_numeric(records_df[col], errors='coerce').fillna(0)
            else:
                records_df[col] = 0

        # Calculate the sum of detailed sqft columns
        records_df['sqft_sum'] = records_df[sqft_sum_cols].sum(axis=1)

        # Metric 2: Total Square Footage Occupied - updated to include commercial lodging
        # For lodging, it's commercial_lodging_sqft. For others, it's the greater of sqft and sfla.
        records_df['total_sqft_occupied'] = records_df.apply(
            lambda row: (row['sqft_sum'] if row['units'] > 0 else 0) if 'Hotel/Lodging' in (row.get('BuildingType', '')) else
                        (max(row['sqft'], row['sfla']) if row['sqft'] > 0 or row['sfla'] > 0 else row['sqft_sum']),
            axis=1
        )

        # Metric 1: Square Feet of Living Area (SFLA) - keep original sfla, NO fallbacks
        # If sfla is 0, there is no living area on the property (e.g., vacant land, mining claims)
        records_df['sfla'] = records_df['sfla']

        records_df.drop(columns=['sqft_sum'], inplace=True)
        # --- End of Updated SqFt Calculation Logic ---

        # --- Handle Multi-Unit Properties ---
        if 'units' in records_df.columns:
            records_df['units'] = pd.to_numeric(records_df['units'], errors='coerce').fillna(1)
            # Ensure units are at least 1, and an integer
            records_df['units'] = records_df['units'].apply(lambda x: int(max(x, 1)))
        else:
            records_df['units'] = 1

        # Divide square footage by the number of units before exploding
        # This ensures each new 'unit' row has its proportional share of the sqft
        records_df['sfla'] = records_df['sfla'] / records_df['units']

        # Expand the DataFrame so each unit is a separate row
        # This is crucial for accurate unit-based counting in all subsequent analyses
        records_df = records_df.loc[records_df.index.repeat(records_df['units'])].reset_index(drop=True)
        # --- End of Multi-Unit Handling ---

        # Data Cleaning: Filter out records that are likely just land (no structure)
        records_df = records_df[records_df["sfla"] > 0].copy()

        # --- Join with Owner Data ---
        owner_data_path = "owner_data.csv"
        if os.path.exists(owner_data_path):
            print("Found owner_data.csv, joining with records...")
            owner_df = pd.read_csv(owner_data_path)
            # Ensure join keys are the same type (string)
            records_df["schno"] = records_df["schno"].astype(str)
            owner_df["Schno"] = owner_df["Schno"].astype(str)
            records_df = pd.merge(
                records_df, owner_df, left_on="schno", right_on="Schno", how="left"
            )
            records_df.drop("Schno", axis=1, inplace=True)  # Drop redundant column
        else:
            print("Warning: owner_data.csv not found. Some owner analyses will be skipped.")

        # --- Perform Column Creations for Analyses ---
        # This is done here so all analysis functions can use the new columns.
        # LocationType creation
        summit_cities = [
            "BRECKENRIDGE", "FRISCO", "DILLON", "SILVERTHORNE",
            "KEYSTONE", "COPPER MOUNTAIN", "HEENEY", "BLUE RIVER",
        ]
        records_df["city"] = records_df["city"].str.upper().str.strip()
        records_df["state"] = records_df["state"].str.upper().str.strip()
        def get_location_type(row):
            if pd.isna(row['city']) or pd.isna(row['state']):
                return "Out of State"
            if row["city"] in summit_cities and row["state"] == "CO":
                return "In-County"
            elif row["state"] == "CO":
                return "In-State (Out of County)"
            else:
                return "Out of State"
        records_df["LocationType"] = records_df.apply(get_location_type, axis=1)

        # BuildingType creation using abstract codes
        abstract_code_map = load_abstract_code_map()
        if abstract_code_map:
            records_df["BuildingType"] = records_df.apply(
                determine_building_type, axis=1, code_map=abstract_code_map
            )
        else:
            # Fallback if AbstractCodes.csv is not found
            records_df["BuildingType"] = "Unknown"

    except FileNotFoundError:
        print("Error: records.csv not found.")
        return
    except KeyError as e:
        print(
            f"Error: A required column was not found: {e}. Please check the column names in records.csv."
        )
        return

    # Step 3: Calculate metrics
    print("\nCalculating historical supply metrics...")
    supply_timeline, sqft_timeline = calculate_historical_supply(records_df)

    # --- MODIFICATION: Run Owner Language analysis first to get OwnerType ---
    print("Analyzing owner language...")
    records_df, owner_type_summary = analyze_owner_language(records_df)
    # --- END MODIFICATION ---

    print("Analyzing owner location and distribution...")
    (
        location_summary,
        owner_distribution,
        owner_distribution_by_location,
        multi_owner_type_breakdown,
    ) = analyze_owner_location_and_distribution(records_df)

    print("Analyzing buyer characteristics by location...")
    buyer_characteristics = analyze_buyer_characteristics_by_location(records_df)

    print("Analyzing out-of-state owner locations...")
    out_of_state_summary = analyze_out_of_state_owners(records_df)

    print("Analyzing overall building type distribution...")
    (
        building_type_summary_by_units,
        building_type_summary_by_sqft,
    ) = analyze_property_composition(records_df)

    print("Analyzing hotel construction growth...")
    hotel_growth_timeline = analyze_hotel_growth(records_df)

    # --- New Analysis ---
    owner_portfolio_df = analyze_owner_type_property_portfolio(records_df)
    owner_portfolio_df.to_csv(
        os.path.join(DATA_DIR, "owner_property_portfolio.csv"), index=False
    )

    # Step 4: Create unified dataset
    unified_dataset = create_unified_dataset(records_df, owner_data_exists=os.path.exists(owner_data_path))
    unified_dataset.to_csv(
        os.path.join(DATA_DIR, "unified_dataset.csv"), index=False
    )
    print("Unified dataset saved.")

    # Step 5: Save processed data
    supply_timeline.to_csv(
        os.path.join(DATA_DIR, "local_housing_supply_timeline.csv"), index=False
    )
    sqft_timeline.to_csv(
        os.path.join(DATA_DIR, "total_residential_sqft_timeline.csv"), index=False
    )
    location_summary.to_csv(
        os.path.join(DATA_DIR, "owner_location_summary.csv"), index=False
    )
    owner_distribution.to_csv(
        os.path.join(DATA_DIR, "property_ownership_distribution.csv"), index=False
    )

    if owner_distribution_by_location is not None:
        owner_distribution_by_location.to_csv(
            os.path.join(DATA_DIR, "property_ownership_by_location.csv"), index=False
        )
        print("Owner distribution by location data saved.")

    if multi_owner_type_breakdown is not None:
        multi_owner_type_breakdown.to_csv(
            os.path.join(DATA_DIR, "multi_property_owner_breakdown.csv"), index=False
        )
        print("Multi-property owner breakdown data saved.")

    if owner_type_summary is not None:
        owner_type_summary.to_csv(
            os.path.join(DATA_DIR, "owner_type_summary.csv"), index=False
        )
        print("Owner language analysis saved.")

    if buyer_characteristics is not None:
        buyer_characteristics.to_csv(
            os.path.join(DATA_DIR, "buyer_characteristics_by_location.csv"), index=False
        )
        print("Buyer characteristics analysis saved.")

    if out_of_state_summary is not None:
        out_of_state_summary.to_csv(
            os.path.join(DATA_DIR, "top_out_of_state_locations.csv"), index=False
        )
        print("Out-of-state owner analysis saved.")

    if building_type_summary_by_units is not None:
        building_type_summary_by_units.to_csv(
            os.path.join(DATA_DIR, "building_type_summary.csv"), index=False
        )
        print("Overall building type distribution by unit saved.")

    if building_type_summary_by_sqft is not None:
        building_type_summary_by_sqft.to_csv(
            os.path.join(DATA_DIR, "building_type_sqft_summary.csv"), index=False
        )
        print("Overall building type distribution by sqft saved.")

    if hotel_growth_timeline is not None:
        hotel_growth_timeline.to_csv(
            os.path.join(DATA_DIR, "hotel_growth_timeline.csv"), index=False
        )
        print("Hotel growth analysis saved.")

    print("Historical supply and value data saved.")
    print("Owner location and distribution data saved.")

    # Step 6: Combine for contextual metrics
    # Load the FRED data we just saved
    summit_pop_df = pd.read_csv(os.path.join(DATA_DIR, "summit_pop.csv"))
    us_pop_df = pd.read_csv(os.path.join(DATA_DIR, "us_pop.csv"))
    us_housing_df = pd.read_csv(os.path.join(DATA_DIR, "us_housing.csv"))
    fed_rate_df = pd.read_csv(os.path.join(DATA_DIR, "fed_rate.csv"))

    # Rename columns for clarity before merging
    summit_pop_df.rename(
        columns={"value": "summit_population", "year": "year_blt"}, inplace=True
    )
    us_pop_df.rename(columns={"value": "us_population", "year": "year_blt"}, inplace=True)
    us_housing_df.rename(
        columns={"value": "us_total_housing_units", "year": "year_blt"}, inplace=True
    )
    fed_rate_df.rename(columns={"value": "fed_interest_rate", "year": "year_blt"}, inplace=True)

    # Merge datasets
    contextual_df = pd.merge(
        supply_timeline,
        summit_pop_df[["year_blt", "summit_population"]],
        on="year_blt",
        how="left",
    )
    contextual_df = pd.merge(
        contextual_df, us_pop_df[["year_blt", "us_population"]], on="year_blt", how="left"
    )
    contextual_df = pd.merge(
        contextual_df,
        us_housing_df[["year_blt", "us_total_housing_units"]],
        on="year_blt",
        how="left"
    )
    contextual_df = pd.merge(
        contextual_df,
        fed_rate_df[["year_blt", "fed_interest_rate"]],
        on="year_blt",
        how="left"
    )

    # Forward fill the national data to align with annual local data
    contextual_df.fillna(method="ffill", inplace=True)

    contextual_df.to_csv(
        os.path.join(DATA_DIR, "contextual_growth_metrics.csv"), index=False
    )
    print("Contextual growth metrics data saved.")
    print("\nAnalysis complete. All data is in the 'data' directory.")


if __name__ == "__main__":
    main()
