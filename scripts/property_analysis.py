import pandas as pd


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


def analyze_building_type_distribution(df):
    """
    Analyzes and summarizes the distribution of building types.
    """
    if "BuildingType" not in df.columns:
        print("Warning: 'BuildingType' column not found. Skipping building type analysis.")
        return None
    building_type_summary = (
        df["BuildingType"].value_counts(normalize=True).mul(100).reset_index()
    )
    building_type_summary.columns = ["BuildingType", "Percentage"]
    return building_type_summary


def analyze_hotel_growth(df):
    """Analyzes the construction rate of hotels and lodging."""
    print("Analyzing hotel/lodging growth timeline...")
    # Use the pre-calculated BuildingType column
    if "BuildingType" not in df.columns:
        print("Warning: 'BuildingType' column not found. Skipping hotel growth analysis.")
        return None

    hotel_df = df[df["BuildingType"] == "Commercial"].copy()

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
