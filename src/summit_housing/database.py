import sqlite3
import pandas as pd
from typing import Optional, List, ContextManager

DB_PATH = "summit_housing.db"

class SummitDB:
    """
    Context manager for SQLite database interactions.
    Ensures connections are closed and transactions are committed.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "SummitDB":
        self.conn = sqlite3.connect(self.db_path)
        # Enable foreign keys and other helpful pragmas
        self.conn.execute("PRAGMA foreign_keys = ON;")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()

    def execute_script(self, script: str) -> None:
        """Executes a raw SQL script (multiple statements)."""
        if not self.conn:
            raise RuntimeError("Database connection is not open.")
        self.conn.executescript(script)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Executes a single SQL statement."""
        if not self.conn:
            raise RuntimeError("Database connection is not open.")
        return self.conn.execute(sql, params)

    def execute_many(self, sql: str, params: List[tuple]) -> sqlite3.Cursor:
        """Executes a bulk SQL statement."""
        if not self.conn:
            raise RuntimeError("Database connection is not open.")
        return self.conn.executemany(sql, params)

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Executes a query and returns the result as a Pandas DataFrame."""
        if not self.conn:
            raise RuntimeError("Database connection is not open.")
        return pd.read_sql_query(sql, self.conn, params=params)

def get_db() -> ContextManager[SummitDB]:
    """Helper to get a database context manager."""
    return SummitDB()
