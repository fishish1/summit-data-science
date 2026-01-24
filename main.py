import pandas as pd
from analysis import analyze_building_type_distribution, analyze_year_built, analyze_neighborhood_distribution
from report import generate_report

def main():
    # Load data
    data = pd.read_csv("real_estate_data.csv")

    # Perform analysis
    building_type_summary = analyze_building_type_distribution(data)
    year_built_summary = analyze_year_built(data)
    neighborhood_summary = analyze_neighborhood_distribution(data)

    # Generate report
    summaries = {}
    if building_type_summary is not None:
        summaries["Building Type Distribution"] = building_type_summary
    else:
        summaries["Building Type Distribution"] = "No data available."
    if year_built_summary is not None:
        summaries["Year Built Analysis"] = year_built_summary
    else:
        summaries["Year Built Analysis"] = "No data available."
    if neighborhood_summary is not None:
        summaries["Neighborhood Distribution"] = neighborhood_summary
    else:
        summaries["Neighborhood Distribution"] = "No data available."

    generate_report(summaries, "Real_Estate_Analysis_Report.md")


if __name__ == "__main__":
    main()