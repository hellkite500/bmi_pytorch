from typing import Tuple

import numpy as np
from pathlib import Path
import pytest
import torch

from ..config import Config
from ..model import Model
from ..utils import load_data


def test_single_layer(config: Config, input: torch.Tensor):
    """_summary_

    Args:
        input (Tensor): an Nx1 set of inputs
    """
    # Construct the model, 1 input, 1 output\
    model = Model(config)
    # Pass input through the model
    tmp = model(input)


def test_two_layer(config: Config, input, compare):
    """_summary_

    Args:
        input (_type_): _description_
        compare (_type_): _description_
    """
    # Construct the model
    config.hidden_size = [10]
    model = Model(config)
    # Pass input through the model
    tmp = model(input)


def test_three_layer(config: Config, input, compare):
    """_summary_

    Args:
        input (_type_): _description_
        compare (_type_): _description_
    """
    # Construct the model
    config.hidden_size = [10, 15]
    model = Model(config)
    # Pass input through the model
    tmp = model(input)


def test_data_load(data_dims: Tuple[int, int]):
    """_summary_

    Args:
        data_dims (_type_): _description_
    """
    # Load the data
    runoff_mean, precip_mean = load_data(Path(__file__).parent/"data/CAMELS")
    # Check the shapes and dtypes
    assert isinstance(runoff_mean, np.ndarray), "Runoff mean is not a numpy array"
    assert isinstance(precip_mean, np.ndarray), "Precip mean is not a numpy array"
    assert runoff_mean.shape == data_dims, "Runoff mean shape is incorrect"
    assert precip_mean.shape == data_dims, "Precip mean shape is incorrect"
