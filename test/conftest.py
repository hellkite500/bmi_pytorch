import pytest
import torch
import numpy as np
from config import Config
from typing import Tuple

@pytest.fixture
def input():
    np.random.seed(42)
    data = np.random.uniform(0, 10, 24).reshape(24, 1)
    return torch.Tensor(data)

@pytest.fixture
def config():
    return Config(
        input_size=1,
        output_size=1,
        hidden_size=[],
        learning_rate=0.005,
        epochs=800,
    )

@pytest.fixture
def data_dims() -> Tuple[int, int]:
    return (671, 1)

@pytest.fixture
def compare():
    np.random.seed(4242)
    data = np.random.uniform(0, 10, 24).reshape(24, 1)
    return torch.Tensor(data)
