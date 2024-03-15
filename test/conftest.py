from contextlib import contextmanager
from os import PathLike, chdir, getcwd, system
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pytest
import torch

from config import Config


@contextmanager
def pushd(path: Union[str, PathLike]) -> None:
    """Change current working directory to the given path.

    Parameters
    ----------
    path : New directory path

    Returns
    ----------
    None

    """
    # Save current working directory
    cwd = getcwd()

    # Change the directory
    chdir(path)
    try:
        yield
    finally:
        chdir(cwd)


def pytest_sessionstart(session) -> None:
    """attempt to download data before starting tests if it doesn't exist

    Args:
        session (_type_): _description_
    """
    path = Path(__file__).parent.parent / "data/CAMELS"
    if not path.exists():
        with pushd(Path(__file__).parent.parent):
            system("scripts/download_camels.sh")


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

@pytest.fixture
def bmi_model() -> Bmi_Model:
    return Bmi_Model()

@pytest.fixture
def bmi_model_initialized(config, bmi_model):
    bmi_model.initialize(config)
    return bmi_model
