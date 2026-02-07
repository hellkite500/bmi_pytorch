from __future__ import annotations

from enum import Enum

# Once python 3.8 is no longer supported, just get Annotated directly
# from typing import Annotated, Any, Literal, Optional, Self, Union
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    model_validator,
)

# Only needed on python 3.8
from typing_extensions import Annotated, Self

from .bmi_grid import Grid

Integer = RootModel[Literal["int", "int16", "int32", "integer"]]
Long = RootModel[Literal["int", "int64", "longlong"]]
Float = RootModel[
    Literal[
        "float",
        "float32",
        "np.float32",
        "numpy.float32",
        "np.single",
        "numpy.single",
    ]
]
Double = RootModel[Literal["float", "float64", "np.float64", "numpy.float64"]]

DataType = Union[Integer, Long, Float, Double]


class GridLocation(str, Enum):
    """Enumeration of variable grid location strings supported by BMI.

    See https://bmi.readthedocs.io/en/stable/#get-var-location

    """

    node = "node"
    edge = "edge"
    face = "face"


class BmiVariable(BaseModel):
    """Abstraction of a BMI variable which contains all the standard BMI variable metadata."""

    # Required for Grid attribute
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    grid: Annotated[Grid, "Bmi Grid object"] = Field(
        default_factory=lambda: Grid(0, 0, "scalar")
    )
    type: DataType = Field(default="float")
    units: str = Field(default="-")
    # By default, value is a numpy scalar of float type
    value: Optional[Any] = Field(
        default_factory=lambda: np.zeros(tuple()), dtype=np.float32
    )
    location: GridLocation = Field(default="node")

    @computed_field
    @property
    def rank(self) -> int:
        """Rank of the grid associated with the variable

        Returns:
            int: rank
        """
        return self.grid.rank

    @computed_field
    @property
    def itemsize(self) -> int:
        """Size of single variable instance, in bytes

        Returns:
            int: itemsize
        """
        # FIXME assumes value is numpy array, or at least has an itemsize property
        return self.value.itemsize

    @computed_field
    @property
    def nbytes(self) -> int:
        """Total number of bytes required to represent the variable

        Returns:
            int: nbytes
        """
        # FIXME assumes value is numpy array, or at least has an nbytes property
        return self.value.nbytes

    @model_validator(mode="after")
    def check_variable(self) -> Self:
        """Checks variable rank, grid, and size/nbytes are all compatible

        Returns:
            Self: BmiVariable with consistent sizes for grid
        """
        # FIXME several assumptions of shape attribute (e.g. numpy)
        if self.grid.type == "scalar":
            assert (
                self.value.shape == ()
            ), f"{self.name} is associated with scalar grid {self.grid.id}, but has value shape {self.value.shape}"
        else:
            # check the number of dims
            assert (
                len(self.value.shape) == self.rank
            ), f"{self.name} is associated with grid {self.grid.id} of rank {self.grid.rank}, but value has {len(self.value.shape)} dimensions"
            assert (
                self.grid.shape == self.value.shape
            ), f"{self.name} is associated with grid of shape {self.grid.shape}, but has value of shape {self.value.shape}"
            assert (
                self.nbytes == self.grid.size() * self.itemsize
            ), f"{self.name} grid size {self.grid.size()} does not match required number of items for nbytes"
            # TODO check compatiblity of dtype with value.dtype?
            # this would require mapping all "bmi" type strings supported...
