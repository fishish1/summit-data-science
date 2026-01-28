import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def scrape_property_data(schno):
    """
    Scrapes primary and secondary owner data for a given schedule number (Schno).
    """
    base_url = "https://gis.summitcountyco.gov/map/DetailData.aspx"
    params = {'Schno': schno}
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Could not retrieve data for Schno {schno}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Check for an indicator of an empty/invalid page.
    if soup.find(string=lambda text: text and "no record found" in text.lower()):
        print(f"Skipping empty or invalid page for Schno {schno}")
        return None

    property_data = {'Schno': schno}

    # Helper to get text from a beautiful soup element, if it exists.
    def get_text(element):
        return element.get_text(strip=True) if element else None

    # --- Find Owner Data ---
    detail_table = soup.find('table', class_='DetailData')
    if detail_table:
        primary_label = detail_table.find('td', string='Primary:')
        property_data['Primary Owner'] = get_text(primary_label.find_next_sibling('td')) if primary_label else None
        
        secondary_label = detail_table.find('td', string='Secondary:')
        property_data['Secondary Owner'] = get_text(secondary_label.find_next_sibling('td')) if secondary_label else None
    else:
        # If table not found, set keys to None
        property_data['Primary Owner'] = None
        property_data['Secondary Owner'] = None

    return property_data

def main():
    """
    Main function to iterate through Schno values, scrape data, and save to CSV.
    """
    input_filename = 'records.csv'
    output_filename = 'owner_data.csv'

    try:
        # Use low_memory=False for mixed dtypes in records.csv
        schno_df = pd.read_csv(input_filename, low_memory=False)
        all_schnos = schno_df['schno'].astype(str).unique().tolist()
    except FileNotFoundError:
        print(f"Input file not found: {input_filename}")
        return
    except Exception as e:
        print(f"Error reading {input_filename}: {e}")
        return

    processed_schnos = set()
    # Check if output file exists and read processed Schnos
    try:
        if pd.io.common.file_exists(output_filename):
            processed_df = pd.read_csv(output_filename)
            if 'Schno' in processed_df.columns:
                processed_schnos = set(processed_df['Schno'].astype(str))
            print(f"Resuming scrape. {len(processed_schnos)} properties already scraped.")
    except Exception as e:
        print(f"Could not read existing output file {output_filename}. Starting from scratch. Error: {e}")


    schnos_to_scrape = [s for s in all_schnos if s not in processed_schnos]
    
    if not schnos_to_scrape:
        print("All properties from input file have already been scraped.")
        return

    print(f"Starting scrape for {len(schnos_to_scrape)} properties from {input_filename}...")

    for i, schno in enumerate(schnos_to_scrape):
        schno = str(schno).strip()
        data = scrape_property_data(schno)
        
        if data:
            print(f"({i+1}/{len(schnos_to_scrape)}) Successfully scraped data for Schno {schno}")
            
            # Convert the single property data to a DataFrame
            df = pd.DataFrame([data])
            
            # Append to CSV, write header only if file doesn't exist
            try:
                header = not pd.io.common.file_exists(output_filename)
                df.to_csv(output_filename, mode='a', header=header, index=False)
            except IOError as e:
                print(f"Error writing to {output_filename}: {e}")

        # Be respectful to the server and avoid getting blocked
        # A randomized delay is better than a fixed one.
        time.sleep(random.uniform(1, 2.5))

    print(f"\nScraping complete. Data saved to {output_filename}")


if __name__ == '__main__':
    main()