import pytest
import pandas as pd
import sqlite3
from summit_housing.database import SummitDB
from summit_housing.queries import SALES_EVENTS_SQL

class PersistentDB(SummitDB):
    """Keeps connection open for in-memory tests."""
    def __enter__(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Do not close automatically
        pass

schema = """
CREATE TABLE raw_records (
    schno TEXT,
    recdate1 TEXT, docfee1 REAL,
    recdate2 TEXT, docfee2 REAL,
    recdate3 TEXT, docfee3 REAL,
    recdate4 TEXT, docfee4 REAL
);
"""

@pytest.fixture
def db():
    # Use in-memory DB wrapper
    db = PersistentDB(":memory:")
    with db:
        db.execute_script(schema)
        # Seed Data
        # Prop 1: Bought for 10k, Sold for 20k
        data = [
            ("PROP1", "2010-01-01", 1.0, "2015-01-01", 2.0, None, None, None, None),
            ("PROP2", "2020-01-01", 5.0, None, None, None, None, None, None)
        ]
        db.execute_many("INSERT INTO raw_records VALUES (?,?,?,?,?,?,?,?,?)", data)
    return db

def test_sales_unpivot(db):
    """Test that SALES_EVENTS_SQL correctly unpivots columns."""
    with db:
        df = pd.read_sql_query(SALES_EVENTS_SQL, db.conn)
    
    assert len(df) == 3 # Prop1x2, Prop2x1
    
    # Check Prop 1 Sales
    prop1 = df[df['schno'] == "PROP1"].sort_values('tx_date')
    assert len(prop1) == 2
    assert prop1.iloc[0]['estimated_price'] == 10000 # 1.0 * 10000
    assert prop1.iloc[1]['estimated_price'] == 20000 # 2.0 * 10000
