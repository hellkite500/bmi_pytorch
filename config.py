import logging
from typing import List

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class Config(BaseModel):
    input_size: int = 1
    output_size: int = 1
    hidden_size: List[None | int] = Field(default_factory=lambda: [10, 10])
    learning_rate: float = 0.005
    epochs: int = 800
