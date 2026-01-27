import pandas as pd
import numpy as np

# Load records
df = pd.read_csv('data/records.csv', low_memory=False)

# Columns of interest
# Mailing Address: 'address', 'city', 'state', 'zip'
# Physical Address Parts: 'num', 'pdir', 'name', 'suff', 'sdir', 'unit'

def construct_phy_address(row):
    parts = []
    
    # 1. Number
    if pd.notna(row['num']) and row['num'] != 0:
        parts.append(str(int(row['num'])))
        
    # 2. Pre-Direction
    if pd.notna(row['pdir']):
        parts.append(str(row['pdir']).strip())
        
    # 3. Street Name
    if pd.notna(row['name']):
        parts.append(str(row['name']).strip())
        
    # 4. Suffix
    if pd.notna(row['suff']):
        parts.append(str(row['suff']).strip())
        
    # 5. Post-Direction
    if pd.notna(row['sdir']):
        parts.append(str(row['sdir']).strip())
        
    # 6. Unit
    if pd.notna(row['unit']):
        parts.append("#" + str(row['unit']).strip())

    return " ".join(parts).upper()

# Construct Physical Address
df['physical_address_constructed'] = df.apply(construct_phy_address, axis=1)

# Normalize Mailing Address
df['mailing_address_norm'] = df['address'].astype(str).str.upper().str.strip()

# Town Code Mapping (from queries.py)
town_map = {
    'B': 'BRECKENRIDGE',
    'D': 'DILLON',
    'F': 'FRISCO',
    'S': 'SILVERTHORNE',
    'K': 'KEYSTONE',
    'C': 'COPPER MOUNTAIN',
    'M': 'MONTEZUMA',
    'R': 'RURAL',
    'BR': 'BLUE RIVER',
    'H': 'HEENEY'
}

df['physical_city'] = df['town'].map(town_map).fillna('OTHER')
df['mailing_city_norm'] = df['city'].astype(str).str.upper().str.strip()

# Check for Match
# 1. City Match
df['city_match'] = df['physical_city'] == df['mailing_city_norm']

# 2. Street Match (Fuzzy)
# Check if the Mailing Address contains the Physical "Number + Name"
# e.g. Physical "123 MAIN" in Mailing "123 MAIN STREET"
def is_address_match(row):
    phy = row['physical_address_constructed']
    mail = row['mailing_address_norm']
    
    if not phy or not mail: return False
    
    # Simple check: Does Mailing Address start with the Physical Address string?
    # (Assuming constructed address is concise like "123 MAIN")
    if mail.startswith(phy): return True
    
    # Check if number matches AND street name is present
    try:
        phy_num = str(int(row['num']))
        phy_name = str(row['name']).strip()
        
        if pd.notna(phy_num) and pd.notna(phy_name):
            # Mailing address must start with number
            if mail.startswith(phy_num):
                 # And contain street name
                 if phy_name in mail:
                     return True
    except:
        pass
        
    return False

df['fuzzy_match'] = df.apply(is_address_match, axis=1)

# Check PO Boxes
df['is_po_box'] = df['mailing_address_norm'].str.contains('PO BOX|P.O. BOX', regex=True)

# Count
total = len(df)
city_matches = df['city_match'].sum()
fuzzy_matches = df['fuzzy_match'].sum()
po_boxes = df['is_po_box'].sum()

print(f"Total Records: {total}")
print(f"Mailing Address is PO Box: {po_boxes} ({po_boxes/total:.1%})")
print(f"Mailing City == Physical City: {city_matches} ({city_matches/total:.1%})")
print(f"Mailing Address MATCHES Physical House: {fuzzy_matches} ({fuzzy_matches/total:.1%})")

# Sample mismatches
print("\n--- Sample Fuzzy Matches ---")
print(df[df['fuzzy_match']][['physical_address_constructed', 'mailing_address_norm']].head())

print("\n--- Sample Local Owners (City Match) but Address Mismatch ---")
mask_local = df['city_match'] & ~df['fuzzy_match'] & ~df['is_po_box']
print(df[mask_local][['physical_address_constructed', 'mailing_address_norm', 'physical_city', 'mailing_city_norm']].head(10))


# Check In-State vs Out-of-State for mismatches
df['is_colorado'] = df['state'].astype(str).str.upper().str.strip() == 'CO'

# Recalculate masks since I removed them in previous edit
mask_mismatch = ~df['fuzzy_match']

# Refine Local Definition: Any Summit County Town
summit_towns = [
    'BRECKENRIDGE', 'DILLON', 'FRISCO', 'SILVERTHORNE', 
    'KEYSTONE', 'COPPER MOUNTAIN', 'BLUE RIVER', 'MONTEZUMA', 'HEENEY'
]
df['is_summit_local'] = df['mailing_city_norm'].isin(summit_towns)

owners_local_city = df[df['city_match']]
owners_summit_local = df[df['is_summit_local']]
owners_in_state = df[mask_mismatch & df['is_colorado']]
# Exclude Summit Locals from the "In-State" bucket to separate "Denver" from "Locals"
owners_in_state_non_local = df[mask_mismatch & df['is_colorado'] & ~df['is_summit_local']]
owners_out_state = df[mask_mismatch & ~df['is_colorado']]

print(f"\nOwner Location Analysis:")
print(f"1. Strict Local (Same City): {len(owners_local_city)} ({len(owners_local_city)/total:.1%})")
print(f"2. Broad Local (Any Summit Town): {len(owners_summit_local)} ({len(owners_summit_local)/total:.1%})")
print(f"3. Front Range / Other CO (Non-Summit): {len(owners_in_state_non_local)} ({len(owners_in_state_non_local)/total:.1%})")
print(f"4. Out-of-State: {len(owners_out_state)} ({len(owners_out_state)/total:.1%})")

