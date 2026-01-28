import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def scrape_property_data(schno):
    """
    Scrapes property data for a given schedule number (Schno).
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
    # Invalid pages have a "No record found for Account" message.
    if soup.find(string=lambda text: text and "no record found" in text.lower()):
        print(f"Skipping empty or invalid page for Schno {schno}")
        return None

    property_data = {'Schno': schno}

    # Helper to get text from a beautiful soup element, if it exists.
    def get_text(element):
        return element.get_text(strip=True) if element else None

    # --- DetailData Table ---
    detail_table = soup.find('table', class_='DetailData')
    if detail_table:
        rows = detail_table.find_all('tr')
        
        # Schedule #
        sched_header = detail_table.find('td', class_='style1aSched')
        if sched_header:
            property_data['Schedule #'] = get_text(sched_header).replace('Schedule #', '').strip()

        # Property Desc, Subdiv, Filing, Phase, Block, Lot
        prop_desc_label = detail_table.find(lambda tag: tag.name == 'td' and 'Property Desc' in tag.get_text())
        if prop_desc_label:
            property_data['Property Desc'] = get_text(prop_desc_label.find_next_sibling('td'))
            
            subdiv_row = prop_desc_label.parent
            subdiv_labels = subdiv_row.find_all('td', class_='style1a')
            subdiv_values_row = subdiv_row.find_next_sibling('tr')
            if subdiv_values_row:
                subdiv_values = subdiv_values_row.find_all(['td', 'th'])
                # This relies on a specific structure, adjust if it's not consistent
                value_offset = len(subdiv_values_row.find_all(class_=['styleBlank', 'style1']))
                
                labels = ['Subdiv', 'Filing', 'Phase', 'Block', 'Lot']
                for i, label_text in enumerate(labels):
                    # Find the header cell for the label
                    label_cell = next((cell for cell in subdiv_labels if label_text in cell.get_text()), None)
                    if label_cell:
                        # Find its index to get the corresponding value
                        try:
                            idx = list(subdiv_row.children).index(label_cell)
                            value_cell = list(subdiv_values_row.children)[idx]
                            property_data[label_text] = get_text(value_cell)
                        except (ValueError, IndexError):
                             pass


        # Phys. Address, PPI
        phys_addr_label = detail_table.find(lambda tag: tag.name == 'td' and 'Phys. Address' in tag.get_text())
        if phys_addr_label:
            property_data['Phys. Address'] = get_text(phys_addr_label.find_next_sibling('td'))
            ppi_label = phys_addr_label.parent.find('td', string='PPI:')
            if ppi_label:
                property_data['PPI'] = get_text(ppi_label.find_next_sibling('td'))

        # Ownership, Econ, Nhood, TaxArea, AssdVal, etc.
        # This part is tricky due to layout. We'll find labels and get next appropriate sibling.
        labels_map = {
            'Primary:': 'Primary Owner',
            'Secondary:': 'Secondary Owner',
            'C/O': 'Mailing C/O',
            'Addr:': 'Mailing Addr',
            'CSZ': 'Mailing CSZ',
            'Econ:': 'Econ',
            'Nhood:': 'Nhood',
            'TaxArea:': 'TaxArea',
            'AssdVal:': 'AssdVal',
            'Tship:': 'Tship',
            'Range:': 'Range',
        }
        for label_text, key in labels_map.items():
            label_cell = detail_table.find('td', string=label_text)
            if label_cell:
                property_data[key] = get_text(label_cell.find_next_sibling('td'))

        # Sales History
        sales_history = []
        sales_header_row = detail_table.find('tr', string=lambda t: t and 'Reception' in t)
        if sales_header_row:
            current_row = sales_header_row.find_next_sibling('tr')
            while current_row:
                cells = current_row.find_all('td', class_='style2b')
                if len(cells) == 4:
                    sale = {
                        'Reception': get_text(cells[0]),
                        'Sale Date': get_text(cells[1]),
                        'Document Type': get_text(cells[2]),
                        'Sale Price': get_text(cells[3]),
                    }
                    sales_history.append(sale)
                # Stop if we are no longer in the sales section
                if not current_row.find('td', class_='style2b'):
                    break
                current_row = current_row.find_next_sibling('tr')
        if sales_history:
            property_data['Sales History'] = sales_history

    # --- ValueData Table ---
    value_table = soup.find('table', class_='ValueData')
    if value_table:
        # 2025 Actual Value
        val_2025_header = value_table.find('td', string=lambda t: t and '2025 Actual Value' in t)
        if val_2025_header:
            val_row = val_2025_header.parent.find_next_sibling('tr')
            if val_row:
                cells = val_row.find_all('td', class_='style2c')
                if len(cells) == 3:
                    property_data['2025 Value Type'] = get_text(cells[1])
                    property_data['2025 Actual Value'] = get_text(cells[2])
        # 2024 Actual Value
        val_2024_header = value_table.find('td', string=lambda t: t and '2024 Actual Value' in t)
        if val_2024_header:
            val_row = val_2024_header.parent.find_next_sibling('tr')
            if val_row:
                cells = val_row.find_all('td', class_='style2d')
                if len(cells) == 3:
                    property_data['2024 Value Type'] = get_text(cells[1])
                    property_data['2024 Actual Value'] = get_text(cells[2])

    # --- ImpData Table ---
    imp_table = soup.find('table', class_='ImpData')
    if imp_table:
        imp_labels = imp_table.find_all('td', class_='Impstyle1')
        for label_cell in imp_labels:
            key = get_text(label_cell).replace(':', '')
            if key:
                value_cell = label_cell.find_next_sibling('td', class_='Impstyle2')
                property_data[key] = get_text(value_cell)

    # --- LandData Table ---
    land_table = soup.find('table', class_='LandData')
    if land_table:
        land_labels = land_table.find_all('td', class_='Impstyle1')
        for label_cell in land_labels:
            key = get_text(label_cell).replace(':', '')
            if key:
                value_cell = label_cell.find_next_sibling('td', class_='Impstyle2')
                property_data[key] = get_text(value_cell)

    return property_data

def main():
    """
    Main function to iterate through Schno values, scrape data, and save to CSV.
    """
    input_filename = 'summit_county_properties.csv'
    output_filename = 'scraped_summit_county_properties.csv'

    try:
        schno_df = pd.read_csv(input_filename)
        all_schnos = schno_df.iloc[:, 0].astype(str).tolist()
        if not all_schnos[0].isdigit():
            all_schnos = all_schnos[1:]
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