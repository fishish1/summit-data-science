from summit_housing.models import PropertyRecord
from pydantic import ValidationError
import pytest

def test_property_record_valid():
    """Test valid record creation."""
    data = {
        "schno": "12345",
        "recdate1": "8/10/2012",
        "docfee1": "10.00",
        "year_blt": "2000"
    }
    record = PropertyRecord(**data)
    assert record.schno == "12345"
    assert record.recdate1 == "2012-08-10" # ISO Conversion
    assert record.docfee1 == 10.0
    assert record.year_blt == 2000

def test_property_record_invalid_date():
    """Test standard date handling - invalid date should be None or Error depending on strictness."""
    # Our validator returns None for bad dates
    data = {
        "schno": "12345",
        "recdate1": "NOT A DATE"
    }
    record = PropertyRecord(**data)
    assert record.recdate1 is None 

def test_property_record_numeric_cleaning():
    """Test coercion of numeric strings."""
    data = {
        "schno": "12345",
        "docfee1": "  25.50  "
    }
    record = PropertyRecord(**data)
    assert record.docfee1 == 25.5

def test_missing_required():
    with pytest.raises(ValidationError):
        PropertyRecord(recdate1="2020-01-01") # Missing schno
