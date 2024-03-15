from config import Config
from bmi_grid import GridType
from bmi_model import Bmi_Model


def test_bmi_model_construct() -> None:
    """Tests the model default construction with no custom configuration

       Currently tests input_names, output_names, and grid are "correct"
       Also validates that input and output are not None

       Additional tests of bmi/model states at construction can be
       added here as the model is developed.
    """

    # Construct the model, 1 input, 1 output\
    model: Bmi_Model = Bmi_Model()

    assert len(model.input_names) == 1
    assert model.input_names[0] == "precipitation"

    assert len(model.output_names) == 1
    assert model.output_names[0] == "runoff"

    assert len(model._grids) == 1
    assert model._grids[0].id == 0
    assert model._grids[0].type == GridType.scalar

    assert model.input != None
    assert model.output != None

def test_bmi_initialize(config: Config) -> None:
    """Test bmi initialization function from config

       Currently tests model attribute, learning_rate, and optimizer
       are set after initialization

       Additional tests of bmi/model states at/after initialization can be
       added here as the model is developed.

    Args:
        config (Config): configuration to initialize the model with
    """
    model: Bmi_Model = Bmi_Model()
    assert not hasattr(model, "model")

    model.initialize(config)

    assert hasattr(model, "model")
    assert model.model != None

    assert model.learning_rate == config.learning_rate
    assert model.optimizer != None

