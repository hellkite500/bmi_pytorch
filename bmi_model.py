from bmipy import Bmi
from bmi_grid import Grid, GridType
import torch
from torch import Tensor
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
        self.input_names = ['Input_Tensor']
        self.output_names = ['Output_Tensor']
        # all inputs and outputs map to scalar grid
        self.grid_map = { k:self.grid_0 for k in self.input_names+self.output_names }

        self.units = {k:'-' for k in self.input_names+self.output_names}

        self.input = Tensor()
        self.ouptut = Tensor()
        self._values = {}
        for name in self.input_names:
            self._values[name] = self.input
        for name in self.output_names:
            self._values[name] = self.output