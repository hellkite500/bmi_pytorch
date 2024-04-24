from typing import Literal

from pydantic import BaseModel, Field

TimeUnits = Literal[
    "s",
    "sec",
    "second",
    "seconds",
    "min",
    "minute",
    "minutes",
    "h",
    "hr",
    "hour",
    "hours",
    "d",
    "day",
    "days",
]


class BmiTime(BaseModel):
    """
    Time components required for BMI.

    BMI doesn't have much for specific time implementaiton, but some hints can be found
    in the time funciton docs: https://bmi.readthedocs.io/en/stable/#time-functions

    e.g.
    "Model time is always expressed as a floating point value."
    "The start time in BMI is typically defined to be 0.0."
    "If the model doesn’t define an end time, a large number (e.g., the largest floating point number supported on a platform) is typically chosen."
    "A time step is typically a positive value. However, if the model permits it, a negative value can be used (running the model backward)."
    """

    current_time: float = Field(default=0.0)
    start_time: float = Field(default=0.0)
    end_time: float = Field(default=float("inf"))
    # TODO: extracted from looking through the xml files
    # https://docs.unidata.ucar.edu/udunits/current/#Database
    # need to verify this is a complete set...
    units: TimeUnits = Field(default="s")
    time_step: float = Field(default=3600.0)
