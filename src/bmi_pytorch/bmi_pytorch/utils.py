import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(__file__).parent.parent / "data/CAMELS"

    # TODO add the input and output vars to config file
    # Read the CSV files and drop the "Year" column
    runoff_pd = pd.read_csv(data_dir / "runoff_mm.csv").drop(columns=["Year"])
    precip_pd = pd.read_csv(data_dir / "precipitation_mm.csv").drop(columns=["Year"])
    # attributes_pd = pd.read_csv(data_dir / 'attributes.csv')

    # Calculate the mean values
    runoff_mean = np.array(runoff_pd.mean())
    precip_mean = np.array(precip_pd.mean())
    # attributes_mean = attributes_pd["pet_mean"].to_numpy()[1:]

    # Expanding dimensions to match the expected input shape for PyTorch.
    runoff_mean = runoff_mean.reshape(-1, 1)
    precip_mean = precip_mean.reshape(-1, 1)
    # attributes_mean = np.expand_dims(attributes_mean, 1)

    return runoff_mean, precip_mean
