import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
from bmi_pytorch.bmi_model import Bmi_Model
from bmi_pytorch.config import Config


class TestDataDownloadError(Exception): ...


def pytest_configure(config: pytest.Config) -> None:
    """Download CAMELS data if it doesn't already exist"""

    data_path = Path(__file__).parent / "data"
    camels = data_path / "CAMELS"
    if not camels.exists():
        Path.mkdir(data_path, exist_ok=True)
        print(f"Downloading CAMELS test data to {data_path}")
        url = "https://drive.google.com/uc?export=download&id=1ZeX-M2fA-HKNg1nWwDDsI66O6seUwpz4"
        dest = data_path / "CAMELS.zip"
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            raise TestDataDownloadError("failed to download CAMELS test data") from e

        with zipfile.ZipFile(dest, "r") as fp:
            fp.extractall(data_path)
        os.remove(dest)

    else:
        print(f"CAMELS data already exists in {data_path}")


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
