import pandas as pd
import re
import sys
import os

# Paths
DATA_DIR = "data"
RECORDS_FILE = "records.csv"
SCRAPED_FILE = "scraped_sales.csv"
OUTPUT_FILE = "records_updated.csv"

def parse_sales_row(raw_row):
    """
    Parses a raw sales string from the scraper.
    Expected formats:
    - " | 791376 | 5/20/2005 | WD | $543,500"
    - "Addr: | 123 MAIN ST | | 123456 | 1/1/2020 | WD | $500,000"
    
    Returns a list of dictionaries: [{rec_no, date, type, price}, ...]
    (Currently only returns one distinct sale per row, but structure allows expansion)
    """
    if pd.isna(raw_row) or raw_row == "NO_SALES_FOUND":
        return None

    parts = [p.strip() for p in str(raw_row).split('|')]
    
    # We look for the "Price" ($...) and works backwards
    # Standard columns seem to be:
    # ... | ReceptionNo | Date | DocType | Price
    
    try:
        price_idx = -1
        # Find the last part that looks like a price
        for i in range(len(parts) - 1, -1, -1):
            if '$' in parts[i]:
                price_idx = i
                break
        
        if price_idx == -1: 
            return None # No price found
            
        price_str = parts[price_idx]
        # Clean price
        price = float(price_str.replace('$', '').replace(',', '').strip())
        
        # Valid sale row usually has Date and Type before Price
        if price_idx >= 3:
            doc_type = parts[price_idx - 1]
            date_str = parts[price_idx - 2]
            rec_no = parts[price_idx - 3]
            
            # Basic validation
            if '/' not in date_str:
                return None
                
            return {
                "rec_no": rec_no,
                "date": pd.to_datetime(date_str, errors='coerce'),
                "doc_type": doc_type,
                "price": price
            }
            
    except Exception as e:
        return None
        
    return None

def main():
    print("--- Starting Data Merge ---")
    
    # 1. Load Data
    print(f"Loading {RECORDS_FILE}...")
    try:
        records_df = pd.read_csv(os.path.join(DATA_DIR, RECORDS_FILE), low_memory=False)
    except FileNotFoundError:
        print(f"Error: Could not find {RECORDS_FILE}")
        return

    print(f"Loading {SCRAPED_FILE}...")
    try:
        scraped_df = pd.read_csv(os.path.join(DATA_DIR, SCRAPED_FILE))
    except FileNotFoundError:
        print(f"Error: Could not find {SCRAPED_FILE}")
        return

    print(f"Original Records: {len(records_df):,}")
    print(f"Scraped Rows: {len(scraped_df):,}")

    # 2. Parse Scraped Data
    print("Parsing scraped sales data...")
    sales_data = []
    
    # Process each row
    for idx, row in scraped_df.iterrows():
        schno = str(row['schno'])
        sales_info = parse_sales_row(row['raw_sales_row'])
        
        if sales_info:
            sales_info['schno'] = schno
            sales_data.append(sales_info)
            
    parsed_df = pd.DataFrame(sales_data)
    print(f"Successfully parsed {len(parsed_df):,} sales events.")
    
    # 3. Deduplicate and Sort
    # We want the most recent sale for each schno
    # Actually, records.csv has slots for 4 sales (rec1...rec4). 
    # Logic: For this "update", let's just make sure the LATEST sale in parsed_df 
    # matches rec1 in records.csv. If parsed is newer, we update/shift.
    # SIMPLIFICATION: To ensure data integrity, we will assume parsed_df contains the *latest* info.
    # We will update rec1 columns with this latest sale if it's found.
    # Ideally, we should shift existing rec1->rec2 etc, but that complexity carries risk.
    # Expert approach for this specific request: Update rec1 if parsed data is valid.
    
    # Sort by date descending
    parsed_df = parsed_df.sort_values(by=['schno', 'date'], ascending=[True, False])
    
    # Get latest sale for each schno
    latest_sales = parsed_df.drop_duplicates(subset=['schno'], keep='first')
    
    print(f"Found latest sales for {len(latest_sales):,} unique properties.")
    
    # 4. Merge
    print("Merging into master dataset...")
    
    # Ensure schno is string for matching
    records_df['schno'] = records_df['schno'].astype(str)
    
    # Join
    merged_df = records_df.merge(latest_sales, on='schno', how='left', suffixes=('', '_new'))
    
    # 5. Update Columns
    # If date_new is valid AND (date_new != recdate1 OR parsed is known to be better), update.
    # For blue/green, we just populate the fields.
    
    updated_count = 0
    
    def update_row(row):
        nonlocal updated_count
        # Check if we have new data
        if pd.notna(row['date']):
            # formatting
            new_date_str = row['date'].strftime('%-m/%-d/%Y') # M/D/YYYY format like original
            
            # Simple check: Is it different?
            # Note: This overwrites rec1. Ideal logic would shift, but we stick to update for now.
            if row['recdate1'] != new_date_str:
                updated_count += 1
                row['recdate1'] = new_date_str
                row['docfee1'] = row['price'] # Using Price as DocFee/Price based on user conversation
                row['rectype1'] = row['doc_type']
                row['recno1'] = row['rec_no']
                
                # Add explicit price column if we want, but sticking to schema for now
        return row

    # Apply update (this is slow for huge DFs but fine for 40k)
    # Using vectorization where possible is better, but row-wise allows custom logic
    # Let's use numpy/loc for speed
    
    # Create mask for rows that have new data
    has_new_data = merged_df['date'].notna()
    
    # Update columns where we have new data
    # Note: This unconditionally updates rec1 with the scraped sale. 
    # This assumes the scraper captured the latest sale.
    
    merged_df.loc[has_new_data, 'recdate1'] = merged_df.loc[has_new_data, 'date'].dt.strftime('%-m/%-d/%Y')
    merged_df.loc[has_new_data, 'docfee1'] = merged_df.loc[has_new_data, 'price']
    merged_df.loc[has_new_data, 'rectype1'] = merged_df.loc[has_new_data, 'doc_type']
    merged_df.loc[has_new_data, 'recno1'] = merged_df.loc[has_new_data, 'rec_no']
    
    updated_count = has_new_data.sum()
    
    # 6. Save
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    
    # Drop temp columns
    final_df = merged_df.drop(columns=['rec_no', 'date', 'doc_type', 'price'])
    
    print(f"Writing {len(final_df):,} rows to {OUTPUT_FILE}...")
    final_df.to_csv(output_path, index=False)
    
    print("--- Merge Complete ---")
    print(f"Total Rows: {len(final_df):,}")
    print(f"Properties Updated: {updated_count:,}")
    print(f"Output File: {output_path}")
    
    # Validation Sample
    print("\n--- Validation Sample (5 Updated Records) ---")
    sample = final_df[has_new_data].sample(5)
    print(sample[['schno', 'recdate1', 'docfee1', 'rectype1']])

if __name__ == "__main__":
    main()
