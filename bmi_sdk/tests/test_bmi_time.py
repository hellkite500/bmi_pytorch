from ..bmi_time import BmiTime
from pydantic import ValidationError
import pytest

@pytest.mark.parametrize('unit', ["s", "sec", "second", "seconds", 
                       "min", "minute", "minutes"
                       "h", "hr", "hour", "hours", 
                       "d", "day", "days"])
def test_good_time_units(unit: str) -> None:
    """Test validation of time units

    Args:
        unit (str): unit of time
    """
    t = BmiTime(units=unit)
    assert t.units == unit

@pytest.mark.parametrize('unit', ["secs", "summer", 
                       "mins", "minut",
                       "hrs", "huors", 
                       "year", "years"])
def test_bad_time_units(unit: str) -> None:
    """Test validation of time units

    Args:
        unit (str): unit of time
    """
    with pytest.raises(ValidationError):
        t = BmiTime(units=unit)