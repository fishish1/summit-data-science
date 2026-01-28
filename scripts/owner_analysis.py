import pandas as pd


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
