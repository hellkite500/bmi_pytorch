import torch
from bmipy import Bmi
from torch import Tensor

from bmi_grid import Grid, GridType
from config import Config
from model import Model


class Bmi_Model(Bmi):
    """BMI composition wrapper for Model

    Args:
        Bmi (_type_): _description_
    """

    def __init__(self):
        super(Bmi_Model, self).__init__()
        # Grid 0 is a 0 dimension "grid" for scalars
        self.grid_0: Grid = Grid(0, 0, GridType.scalar)
        self.input_names = ["precipitation"]
        self.output_names = ["runoff"]
        # all inputs and outputs map to scalar grid
        self.grid_map = {k: self.grid_0 for k in self.input_names + self.output_names}

        self.units = {k: "-" for k in self.input_names + self.output_names}

        self.input = Tensor()
        self.ouptut = Tensor()
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
        self.learning_rate = _config.learning_rate
        self.optimzer = torch.optim.SGD(self.model.parameters(), self.learning_rate)

    def update(self):
        """Update the model for the internal timestep duration
        """
        self.output = self.model(self.input)

    def update_until(self, time: float) -> None:
        """Update model from current_time until current_time + time

        Args:
            time (float): duration of time to advance model till
        """
        raise NotImplementedError()

    def finalize(self):
        """Clean up any internal resources of the model"""
        pass
