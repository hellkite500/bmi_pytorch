from ..bmi_model import Bmi_Model
from contextlib import contextmanager
from os import PathLike, chdir, getcwd, system
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pytest
import torch

from ..config import Config


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
    data_path = Path(__file__).parent/"data"
    print(data_path)
    camels = data_path / "CAMELS"
    if not camels.exists():
        Path.mkdir(data_path, exist_ok=True)
        with pushd(data_path):
            url = 'https://drive.google.com/uc?export=download&id=1ZeX-M2fA-HKNg1nWwDDsI66O6seUwpz4'
            dest = 'CAMELS.zip'
            #TODO replace with requestlib?
            system(f"wget --no-check-certificate '{url}' -O '{dest}'")
            system("unzip 'CAMELS.zip' && rm CAMELS.zip")


@pytest.fixture
def input() -> torch.Tensor:
    np.random.seed(42)
    data = np.random.uniform(0, 10, 24).reshape(24, 1)
    return torch.Tensor(data)


@pytest.fixture
def config() -> Config:
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
def compare() -> torch.Tensor:
    np.random.seed(4242)
    data = np.random.uniform(0, 10, 24).reshape(24, 1)
    return torch.Tensor(data)

@pytest.fixture
def bmi_model() -> Bmi_Model:
    return Bmi_Model()

@pytest.fixture
def bmi_model_initialized(config, bmi_model) -> Bmi_Model:
    bmi_model.initialize(config)
    return bmi_model
