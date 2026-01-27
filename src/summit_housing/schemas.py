from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
import pandas as pd
from datetime import datetime

class PropertyRecord(BaseModel):
    """
    Represents a single property record from records.csv.
    Includes validation logic to standardize dates for SQLite.
    """
    # Key Identifiers
    schno: str = Field(..., description="Schedule Number (Unique ID)")
    
    # Dates & Fees (The critical sales history)
    recdate1: Optional[str] = None
    docfee1: Optional[float] = None
    recdate2: Optional[str] = None
    docfee2: Optional[float] = None
    recdate3: Optional[str] = None
    docfee3: Optional[float] = None
    recdate4: Optional[str] = None
    docfee4: Optional[float] = None

    # Property Attributes
    sfla: Optional[float] = 0.0  # Square Footage Living Area
    year_blt: Optional[int] = None
    adj_year_blt: Optional[int] = Field(None, description="Adjusted Year Built (effective age)")
    units: Optional[int] = 1
    
    # Location
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = Field(None, alias="zip")
    town: Optional[str] = None # Critical: Property Location Code (B, F, K, etc.)

    # Raw abstract codes for building type logic
    abst1: Optional[str] = None
    abst2: Optional[str] = None

    # ML Features
    beds: Optional[float] = 0.0
    bath_tot: Optional[float] = 0.0
    garage_size: Optional[float] = 0.0
    lot_size: Optional[float] = Field(0.0, alias="acres") # CSV header says 'acres'
    
    # Validators
    @field_validator('recdate1', 'recdate2', 'recdate3', 'recdate4', mode='before')
    @classmethod
    def standardize_date(cls, v: Optional[str]) -> Optional[str]:
        """
        Converts dates from '8/10/2012' or '8/10/2012 9:13:25 AM' 
        to ISO 8601 'YYYY-MM-DD'.
        Returns None if date is invalid or missing.
        """
        if not v or pd.isna(v) or str(v).strip() == "":
            return None
        
        v_str = str(v).strip()
        
        # Try parsing with pandas which is very robust
        try:
            # Setting dayfirst=False for US dates (month/day/year)
            dt = pd.to_datetime(v_str, dayfirst=False)
            if pd.isna(dt):
                return None
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            # If pandas fails, return None (will be caught by type check if strict, 
            # but here we allow None)
            return None

    @field_validator('docfee1', 'docfee2', 'docfee3', 'docfee4', 'sfla', 'units', 'year_blt', mode='before')
    @classmethod
    def clean_numeric(cls, v: Any) -> Optional[float]:
        """Coerces numeric strings to float/int, handling errors."""
        if v is None or pd.isna(v) or str(v).strip() == "":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    @field_validator('year_blt', mode='after')
    @classmethod
    def valid_year(cls, v: Optional[float]) -> Optional[int]:
        if v is None: 
            return None
        v_int = int(v)
        if 1800 <= v_int <= datetime.now().year + 5:
            return v_int
        return None  # Treat wild years (e.g. 0 or 9999) as None via logic, or keep? 
                     # For now, let's keep strict logical range or None
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore"
    }
