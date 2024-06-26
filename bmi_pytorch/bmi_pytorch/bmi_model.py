from typing import List, Tuple

import torch
from bmi_sdk import UnknownBMIVariable
from bmi_sdk.bmi_grid import Grid, GridType
from bmi_sdk.bmi_minimal import Bmi_Minimal
from numpy import ndarray
from torch import Tensor

from .config import Config
from .model import Model


class Bmi_Model(Bmi_Minimal):
    """BMI composition wrapper for Model

    Args:
        Bmi (_type_): _description_
    """

    def __init__(self):
        super(Bmi_Model, self).__init__()
        # Grid 0 is a 0 dimension "grid" for scalars
        self.grid_0: Grid = Grid(0, 0, GridType.scalar)
        self.input_names: Tuple[str] = ("precipitation",)
        self.output_names: Tuple[str] = ("runoff",)
        # all inputs and outputs map to scalar grid
        self.grid_map = {k: self.grid_0 for k in self.input_names + self.output_names}
        self._grids: List[Grid] = [self.grid_0]

        self.units = {k: "-" for k in self.input_names + self.output_names}

        self.input = Tensor()
        self.output = Tensor()
        self._values = {}
        for name in self.input_names:
            self._values[name] = self.input
        for name in self.output_names:
            self._values[name] = self.output

    def initialize(self, config_file: str):
        """_summary_

        Args:
            config (str): _description_
        """
        _config = Config()  # Hack to get the default values. We can change later
        self.model = Model(_config)
        # TODO should these be attributes of the Bmi_Model, or the underlying Model?
        self.learning_rate = _config.learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)

    def update(self):
        """Update the model for the internal timestep duration"""
        self.output = self.model(self.input)

    def finalize(self):
        """Clean up any internal resources of the model"""
        pass

    def get_component_name(self) -> str:
        """Name of this BMI module component.

        Returns:
            str: Model Name
        """
        return "Tensor Test"

    def get_input_item_count(self) -> int:
        """Number of model input variables

        Returns:
            int: number of input variables
        """
        return len(self.input_names)

    def get_input_var_names(self) -> Tuple[str, ...]:
        """The names of each input variables

        Returns:
            tuple[str, ...]: iterable tuple of input variable names
        """
        return self.input_names

    def get_output_item_count(self) -> int:
        """Number of model output variables

        Returns:
            int: number of output variables
        """
        return len(self.output_names)

    def get_output_var_names(self) -> Tuple[str, ...]:
        """The names of each output variable

        Returns:
            tuple[str, ...]: iterable tuple of output variable names
        """
        return self.output_names

    # BMI Variable Query
    def get_value_ptr(self, name: str) -> ndarray:
        # np_array will share memory with the Tensor's
        # numeric array, but won't have any other attributes
        # of the Tensor.
        np_array = self._values[name].numpy()
        shape = np_array.shape
        try:
            # see if raveling is possible without a copy
            np_array.shape = (-1,)
            # reset original shape
            np_array.shape = shape
        except ValueError as e:
            raise RuntimeError(
                "Cannot flatten array without copying -- " + str(e).split(": ")[-1]
            )
        return np_array.ravel()

    # BMI Variable Information Functions
    def get_var_grid(self, name: str) -> int:
        """Get the grid identiferier associated with a given variable

        Args:
            name (str): name of the variable

        Raises:
            UnknownBMIVariable: name is not recognized, grid unknown

        Returns:
            int: grid identifier associated with @p name
        """
        if name in (self.input_names + self.output_names):
            return 0  # should these be on a grid???

        raise (UnknownBMIVariable(f"No known variable in BMI model: {name}"))

    def get_var_itemsize(self, name: str) -> int:
        """Size, in bytes, of a single element of the variable name

        Args:
            name (str): variable name

        Returns:
            int: number of bytes representing a single variable of @p name
        """
        return self.get_value_ptr(name).itemsize

    def get_var_nbytes(self, name: str) -> int:
        """_summary_

        Args:
            name (str): Name of variable.

        Returns:
            int: Size of data array in bytes.
        """
        return self.get_value_ptr(name).nbytes

    def get_var_type(self, name: str) -> str:
        """Data type of variable.

        Args:
            name (str): Name of variable.

        Returns:
            str: Data type.
        """
        return str(self.get_value_ptr(name).dtype)
