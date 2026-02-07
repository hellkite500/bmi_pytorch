import numpy as np
import pytest
from pydantic import ValidationError

from ..bmi_variable import BmiVariable


def test_variable_create() -> None:
    """Test creation of default BmiVariable

    """
    name = "test"
    t = BmiVariable(name=name)
    assert t.name == name

def test_variable_invalid_grid() -> None:
    """Test creating BmiVariable with a value which doesn't
    match the grid shape.

    """
    value = np.zeros((1,1))
    with pytest.raises(ValidationError):
        t = BmiVariable(name="test", value=value)
