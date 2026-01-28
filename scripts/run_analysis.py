import os
import pandas as pd

# Import functions from the new modules
from data_processing import load_and_prepare_data
from owner_analysis import (
    analyze_owner_language,
    analyze_owner_location_and_distribution,
    analyze_buyer_characteristics_by_location,
    analyze_out_of_state_owners,
    analyze_owner_type_property_portfolio,
)
from property_analysis import (
    calculate_historical_supply,
    analyze_building_type_distribution,
    analyze_hotel_growth,
)

# Define the data directory
DATA_DIR = "/Users/brian/Documents/summit/data"


def main():
    """Main function to run the data analysis pipeline."""
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Step 1: Load and prepare data
    print("Loading and preparing data...")
    records_df = load_and_prepare_data()

    if records_df is None:
        print("Data loading failed. Exiting analysis.")
        return

    # Step 2: Run all analyses
    print("\nAnalyzing owner language...")
    records_df, owner_type_summary = analyze_owner_language(records_df)

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
    building_type_summary = analyze_building_type_distribution(records_df)

    print("Analyzing hotel construction growth...")
    hotel_growth_timeline = analyze_hotel_growth(records_df)

    print("Analyzing property portfolios by owner type...")
    owner_portfolio_df = analyze_owner_type_property_portfolio(records_df)

    print("Calculating historical supply metrics...")
    supply_timeline, sqft_timeline = calculate_historical_supply(records_df)

    # Step 3: Save all generated dataframes to CSV
    print("\nSaving analysis results...")

    # Helper function to save CSVs
    def save_df(df, filename):
        if df is not None:
            df.to_csv(os.path.join(DATA_DIR, filename), index=False)
            print(f"Successfully saved {filename}")
        else:
            print(f"Warning: DataFrame for {filename} is None, not saving.")

    save_df(owner_type_summary, "owner_type_summary.csv")
    save_df(location_summary, "owner_location_summary.csv")
    save_df(owner_distribution, "property_ownership_distribution.csv")
    save_df(owner_distribution_by_location, "property_ownership_by_location.csv")
    save_df(multi_owner_type_breakdown, "multi_property_owner_breakdown.csv")
    save_df(buyer_characteristics, "buyer_characteristics_by_location.csv")
    save_df(out_of_state_summary, "top_out_of_state_locations.csv")
    save_df(building_type_summary, "building_type_summary.csv")
    save_df(hotel_growth_timeline, "hotel_growth_timeline.csv")
    save_df(owner_portfolio_df, "owner_property_portfolio.csv")
    save_df(supply_timeline, "local_housing_supply_timeline.csv")
    save_df(sqft_timeline, "total_residential_sqft_timeline.csv")

    # Step 4: Combine for contextual metrics
    print("\nCreating contextual growth metrics...")
    try:
        summit_pop_df = pd.read_csv(os.path.join(DATA_DIR, "summit_pop.csv"))
        us_pop_df = pd.read_csv(os.path.join(DATA_DIR, "us_pop.csv"))
        us_housing_df = pd.read_csv(os.path.join(DATA_DIR, "us_housing.csv"))

        summit_pop_df.rename(
            columns={"value": "summit_population", "year": "year_blt"}, inplace=True
        )
        us_pop_df.rename(columns={"value": "us_population", "year": "year_blt"}, inplace=True)
        us_housing_df.rename(
            columns={"value": "us_total_housing_units", "year": "year_blt"}, inplace=True
        )

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

        contextual_df.fillna(method="ffill", inplace=True)
        save_df(contextual_df, "contextual_growth_metrics.csv")

    except FileNotFoundError as e:
        print(f"Error creating contextual metrics: Could not find required file - {e}")

    print("\nAnalysis complete. All data is in the 'data' directory.")


if __name__ == "__main__":
    main()
