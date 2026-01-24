import csv
import pandas as pd
import os
from pydantic import ValidationError
from summit_housing.database import SummitDB
from summit_housing.models import PropertyRecord

BATCH_SIZE = 1000
DATA_DIR = "data"
RECORDS_FILE = "records.csv"
REJECTS_FILE = "rejected_records.csv"
DB_FILE = "summit_housing.db"

# Schema Definition
SCHEMA_SQL = """
DROP TABLE IF EXISTS raw_records;
CREATE TABLE IF NOT EXISTS raw_records (
    schno TEXT PRIMARY KEY,
    recdate1 TEXT, docfee1 REAL,
    recdate2 TEXT, docfee2 REAL,
    recdate3 TEXT, docfee3 REAL,
    recdate4 TEXT, docfee4 REAL,
    sfla REAL,
    year_blt INTEGER,
    adj_year_blt INTEGER,
    units INTEGER,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    town TEXT,
    abst1 TEXT,
    abst2 TEXT,
    beds REAL,
    bath_tot REAL,
    garage_size REAL,
    lot_size REAL
);

CREATE INDEX IF NOT EXISTS idx_records_city ON raw_records(town); -- Index on Property Location Code
CREATE INDEX IF NOT EXISTS idx_records_year ON raw_records(year_blt);
"""

def init_db():
    print("Initializing Database Schema...")
    with SummitDB(DB_FILE) as db:
        db.execute_script(SCHEMA_SQL)

def ingest_records():
    records_path = os.path.join(DATA_DIR, RECORDS_FILE)
    rejects_path = os.path.join(DATA_DIR, REJECTS_FILE)

    if not os.path.exists(records_path):
        print(f"Error: {records_path} not found.")
        return

    print(f"Starting ingestion from {records_path}...")
    
    # Read CSV with pandas first to handle encoding/quoting issues easily, 
    # then convert to dict stream
    # using chunksize to handle memory efficiently
    chunk_iterator = pd.read_csv(records_path, chunksize=BATCH_SIZE, low_memory=False, dtype=str)
    
    total_processed = 0
    total_inserted = 0
    total_rejected = 0

    # Prepare Rejects File
    with open(rejects_path, 'w', newline='') as f_reject:
        writer = csv.writer(f_reject)
        writer.writerow(['schno', 'error_message', 'raw_data']) # Header

        with SummitDB(DB_FILE) as db:
            for chunk in chunk_iterator:
                batch_data = []
                
                # Convert chunk to list of dicts: replace NaN with None
                records = chunk.where(pd.notnull(chunk), None).to_dict('records')

                for row in records:
                    total_processed += 1
                    try:
                        # Pydantic Validation
                        # rename 'zip' to 'zip_code' for model if needed, or let alias handle it
                        # The alias in model is 'zip', so passing 'zip' in dict works.
                        record = PropertyRecord(**row)
                        
                        # Prepare for SQL
                        # usage of model_dump() (v2) or dict() (v1)
                        data = record.model_dump()
                        batch_data.append((
                            data['schno'],
                            data['recdate1'], data['docfee1'],
                            data['recdate2'], data['docfee2'],
                            data['recdate3'], data['docfee3'],
                            data['recdate4'], data['docfee4'],
                            data['sfla'], data['year_blt'], data['adj_year_blt'], data['units'],
                            data['address'], data['city'], data['state'], data['zip_code'], str(data['town']),
                            data['abst1'], data['abst2'],
                            data['beds'], data['bath_tot'], data['garage_size'], data['lot_size']
                        ))

                    except ValidationError as e:
                        total_rejected += 1
                        # DLQ Logic: Write to reject file
                        # Just grabbing a few identifier columns for context or raw dump
                        schno_val = row.get('schno', 'UNKNOWN')
                        writer.writerow([schno_val, str(e), str(row)])
                    except Exception as e:
                         # Catch unexpected errors to keep pipeline alive
                         total_rejected += 1
                         writer.writerow([row.get('schno', 'UNKNOWN'), f"Unexpected: {str(e)}", str(row)])


                if batch_data:
                    placeholders = ", ".join(["?"] * 24)
                    sql = f"""
                    INSERT OR REPLACE INTO raw_records 
                    (schno, recdate1, docfee1, recdate2, docfee2, recdate3, docfee3, recdate4, docfee4, 
                     sfla, year_blt, adj_year_blt, units, address, city, state, zip_code, town, abst1, abst2,
                     beds, bath_tot, garage_size, lot_size)
                    VALUES ({placeholders})
                    """
                    db.execute_many(sql, batch_data)
                    total_inserted += len(batch_data)
                
                print(f"Processed: {total_processed} | Inserted: {total_inserted} | Rejected: {total_rejected}", end='\r')

    print("\nIngestion Complete.")
    print(f"Total Inserted: {total_inserted}")
    print(f"Total Rejected: {total_rejected} (See {rejects_path})")

if __name__ == "__main__":
    init_db()
    ingest_records()
