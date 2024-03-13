from typing import Tuple

import pytest
from pytest import fixture
import torch
import numpy as np

from config import Config
from model import Model
from src.utils import load_data


@fixture
def input():
    np.random.seed(42)
    data = np.random.uniform(0, 10, 24).reshape(24, 1)
    return torch.Tensor(data)

@fixture 
def config():
    return Config(
        input_size=1,
        output_size=1,
        hidden_size=[],
        learning_rate=0.005,
        epochs=800,
    )
    
@fixture
def data_dims() -> Tuple[int, int]:
    return (671, 1)

@fixture
def compare():
    np.random.seed(4242)
    data = np.random.uniform(0, 10, 24).reshape(24, 1)
    return torch.Tensor(data)

@pytest.mark.usefixtures("input", "config")
def test_single_layer(config: Config, input: torch.Tensor):
    """_summary_

    Args:
        input (Tensor): an Nx1 set of inputs
    """
    #Construct the model, 1 input, 1 output\
    model = Model(config)
    #Pass input through the model
    tmp = model(input)

@pytest.mark.usefixtures("input", "config")
def test_two_layer(config: Config, input, compare):
    """_summary_

    Args:
        input (_type_): _description_
        compare (_type_): _description_
    """
    #Construct the model
    config.hidden_size = [10]
    model = Model(config)
    #Pass input through the model
    tmp = model(input)

@pytest.mark.usefixtures("input", "config")
def test_three_layer(config: Config, input, compare):
    """_summary_

    Args:
        input (_type_): _description_
        compare (_type_): _description_
    """
    print( input.shape )
    #Construct the model
    config.hidden_size = [10]
    model = Model(config)
    #Pass input through the model
    tmp = model(input)
    
    
@pytest.mark.usefixtures("input", "config")
def test_three_layer(config: Config, input, compare):
    """_summary_

    Args:
        input (_type_): _description_
        compare (_type_): _description_
    """
    print( input.shape )
    #Construct the model
    config.hidden_size = [10]
    model = Model(config)
    #Pass input through the model
    tmp = model(input)
    
@pytest.mark.usefixtures("data_dims")
def test_data_load(data_dims: Tuple[int, int]):
    """_summary_

    Args:
        data_dims (_type_): _description_
    """
    #Load the data
    runoff_mean, precip_mean = load_data()
    #Check the shapes
    assert runoff_mean.shape == data_dims, "Runoff mean shape is incorrect"
    assert precip_mean.shape == data_dims, "Precip mean shape is incorrect"