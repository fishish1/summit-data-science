import pandas as pd
import os

# --- Data Loading for Mapping ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
codes_df = pd.read_csv(os.path.join(DATA_DIR, "AbstractCodes.csv"), skipinitialspace=True)

# TODO: It looks like the "Description" column does not exist in the dataframe.
# The line below will print the available columns so you can identify the correct one.
print(codes_df.columns)

def classify_description(description):
    """Classifies a building description into a general type."""
    description = str(description).lower()
    if any(
        keyword in description
        for keyword in ["residential", "dwelling", "apartment", "residence", "res"]
    ):
        return "Residential"
    if any(
        keyword in description
        for keyword in [
            "commercial",
            "store",
            "office",
            "hotel",
            "lodging",
            "retail",
            "bank",
        ]
    ):
        return "Commercial"
    if any(
        keyword in description for keyword in ["industrial", "warehouse", "factory"]
    ):
        return "Industrial"
    if any(
        keyword in description
        for keyword in ["public", "school", "church", "hospital", "government"]
    ):
        return "Public/Institutional"
    return "Other"


# Create a mapping from building class code to a more general type
codes_df["BuildingType"] = codes_df["description"].apply(classify_description)
building_class_to_type_map = pd.Series(
    codes_df.BuildingType.values, index=codes_df.code
).to_dict()


def map_building_class_to_type(bldg_class):
    """Maps a building class code to its general building type."""
    return building_class_to_type_map.get(bldg_class, "Other")
