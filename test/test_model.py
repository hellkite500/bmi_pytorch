import pytest
from pytest import fixture
import torch
import numpy as np
from model import Model

@fixture
def input():
    np.random.seed(42)
    data = np.random.uniform(0, 10, 24).reshape(24, 1)
    return torch.Tensor(data)

@fixture
def compare():
    np.random.seed(4242)
    data = np.random.uniform(0, 10, 24).reshape(24, 1)
    return torch.Tensor(data)

@pytest.mark.usefixtures("input")
def test_single_layer(input):
    """_summary_

    Args:
        input (Tensor): an Nx1 set of inputs
    """
    #Construct the model, 1 input, 1 output
    model = Model(1,1)
    #Pass input through the model
    tmp = model(input)

@pytest.mark.usefixtures("input")
def test_two_layer(input, compare):
    """_summary_

    Args:
        input (_type_): _description_
        compare (_type_): _description_
    """
    #Construct the model
    layers = [10]
    model = Model(1,1, layers)
    #Pass input through the model
    tmp = model(input)

@pytest.mark.usefixtures("input")
def test_three_layer(input, compare):
    """_summary_

    Args:
        input (_type_): _description_
        compare (_type_): _description_
    """
    print( input.shape )
    #Construct the model
    layers = [10, 15]
    model = Model(1,1, layers)
    #Pass input through the model
    tmp = model(input)