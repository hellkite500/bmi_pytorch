from pydantic import BaseModel, Field, field_validator, ValidationInfo

class BmiTime(BaseModel):
    """Time components required for BMI.

    BMI doesn't have much for specific time implementaiton, but some hints can be found
    in the time funciton docs: https://bmi.readthedocs.io/en/stable/#time-functions

    e.g.
    "Model time is always expressed as a floating point value."
    "The start time in BMI is typically defined to be 0.0."
    "If the model doesn’t define an end time, a large number (e.g., the largest floating point number supported on a platform) is typically chosen."
    "A time step is typically a positive value. However, if the model permits it, a negative value can be used (running the model backward)."

    """
    current_time: float = Field(default=0)
    start_time: float = Field(default=0)
    end_time: float = Field(default=float('inf'))
    units: str = Field(default="s")
    time_step: float = Field(default=3600)

    @field_validator('units')
    @classmethod
    def check_time_units(cls, v: str, info: ValidationInfo) -> str:
        """_summary_

        Args:
            v (str): unit string to check
            info (ValidationInfo): Context for the class being validated.

        Returns:
            str: valid udunits2 time unit string
        """
        # TODO extracted from looking throug the xml files
        # https://docs.unidata.ucar.edu/udunits/current/#Database
        # need to verify this is a complete set...
        valid_units = ["s", "sec", "second", "seconds", 
                       "min", "minute", "minutes"
                       "h", "hr", "hour", "hours", 
                       "d", "day", "days"]
        if(not v in valid_units):
            raise ValueError(f'{info.field_name} must be one of {valid_units}')
        # could use assert here, but it will be disabled if python is run with optimization (-O) flag
        # assert v in valid_units, f'{info.field_name} must be one of {valid_units}'
        return v
