import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import Config

log = logging.getLogger(__name__)


def normalize(x: np.ndarray) -> np.ndarray:

    return (x - np.mean(x)) / np.std(x)


def load_data() -> Tuple[np.ndarray, np.ndarray]:

    data_dir = Path(__file__).parent.parent / "data/CAMELS"

    # TODO add the input and output vars to config file
    runoff_pd = pd.read_csv(data_dir / "runoff_mm.csv")
    precip_pd = pd.read_csv(data_dir / "precipitation_mm.csv")
    # attributes_pd = pd.read_csv(data_dir / 'attributes.csv')

    # Calculating the mean for each basin across years.
    # Removing the first column which is year column `.values[1:]`
    runoff_mean = runoff_pd.mean().values[1:]
    precip_mean = precip_pd.mean().values[1:]
    # attributes_mean = attributes_pd["pet_mean"].to_numpy()[1:]

    # Expanding dimensions to match the expected input shape for PyTorch.
    runoff_mean = np.expand_dims(runoff_mean, 1)
    precip_mean = np.expand_dims(precip_mean, 1)
    # attributes_mean = np.expand_dims(attributes_mean, 1)

    return runoff_mean, precip_mean
