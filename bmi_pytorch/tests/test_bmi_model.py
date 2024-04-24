from ..config import Config
from bmi_sdk.bmi_grid import GridType
from ..bmi_model import Bmi_Model, UnknownBMIVariable
import pytest
from torch import Tensor, tensor, float16, float32, float64, int16, int32, int64, int8


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


def test_bmi_initialize(config: Config, bmi_model) -> None:
    """Test bmi initialization function from config

       Currently tests model attribute, learning_rate, and optimizer
       are set after initialization

       Additional tests of bmi/model states at/after initialization can be
       added here as the model is developed.

    Args:
        config (Config): configuration to initialize the model with
    """
    assert not hasattr(bmi_model, "model")

    bmi_model.initialize(config)

    assert hasattr(bmi_model, "model")
    assert bmi_model.model != None

    assert bmi_model.learning_rate == config.learning_rate
    assert bmi_model.optimizer != None


@pytest.mark.parametrize("model", ["bmi_model", "bmi_model_initialized"])
def test_bmi_component_name(model, request):
    m = request.getfixturevalue(model)
    assert m.get_component_name() == "Tensor Test"


@pytest.mark.parametrize("model", ["bmi_model", "bmi_model_initialized"])
def test_bmi_input_item_count(model, request):
    m = request.getfixturevalue(model)
    assert m.get_input_item_count() == 1


@pytest.mark.parametrize("model", ["bmi_model", "bmi_model_initialized"])
def test_bmi_input_var_names(model, request):
    m = request.getfixturevalue(model)
    assert m.get_input_item_count() == 1
    names = m.get_input_var_names()
    assert len(names) == 1
    assert names[0] == "precipitation"


@pytest.mark.parametrize("model", ["bmi_model", "bmi_model_initialized"])
def test_bmi_output_item_count(model, request):
    m = request.getfixturevalue(model)
    assert m.get_output_item_count() == 1


@pytest.mark.parametrize("model", ["bmi_model", "bmi_model_initialized"])
def test_bmi_output_var_names(model, request):
    m = request.getfixturevalue(model)
    assert m.get_output_item_count() == 1
    names = m.get_output_var_names()
    assert len(names) == 1
    assert names[0] == "runoff"


@pytest.mark.parametrize("var,expected", [("precipitation", 0), ("runoff", 0)])
def test_bmi_var_grid(bmi_model_initialized, var, expected):
    m = bmi_model_initialized
    assert m.get_var_grid(var) == expected


@pytest.mark.parametrize("var", ["var1", "var2"])
def test_bmi_var_grid_2(bmi_model_initialized, var):
    m = bmi_model_initialized
    with pytest.raises(UnknownBMIVariable):
        m.get_var_grid(var)


@pytest.mark.parametrize(
    "input",
    [
        Tensor([0, 0]),
        Tensor([0, 1, 2]),
        Tensor([2, 1, 0]),
        Tensor([[0]]),
        Tensor([[0, 1]]),
        Tensor([[0, 1], [2, 3]]),
        Tensor([[3, 2], [1, 0]]),
    ],
)
def test_bmi_get_var_ptr(bmi_model_initialized, input):
    name = "precipitation"
    m = bmi_model_initialized
    m.input = input
    m._values[name] = m.input
    data = m.get_value_ptr(name)
    # Can't test this since data is flattened...
    # assert data.shape == m.input.shape
    assert data.shape == m.input.flatten().shape
    for expected, val in zip(m.input.flatten(), data.flatten()):
        assert expected == val


def test_bmi_get_var_ptr_1(bmi_model_initialized, input):
    name = "precipitation"
    expected = (4, 5, 6)
    m = bmi_model_initialized
    m.input = Tensor([1, 2, 3])
    m._values[name] = m.input
    data = m.get_value_ptr(name)
    data[:] = expected[:]

    for expected, val in zip(m.input.flatten(), expected):
        assert expected == val


@pytest.mark.parametrize(
    "input",
    [
        Tensor([0.0, 0]),
        Tensor([0, 1.0, 2]),
        Tensor([2, 1, 0]),
        Tensor([[0]]),
        Tensor([[0, 1.0]]),
        Tensor([[0, 1.0], [2, 3.0]]),
        Tensor([[3, 2], [1.0, 0]]),
    ],
)
def test_bmi_get_var_itemsize(bmi_model_initialized, input):
    name = "precipitation"
    m = bmi_model_initialized
    m.input = input
    m._values[name] = m.input
    data = m.get_var_itemsize(name)

    assert input.itemsize == data


@pytest.mark.parametrize(
    "input",
    [
        Tensor([0.0, 0]),
        Tensor([0, 1.0, 2]),
        Tensor([2, 1, 0]),
        Tensor([[0]]),
        Tensor([[0, 1.0]]),
        Tensor([[0, 1.0], [2, 3.0]]),
        Tensor([[3, 2], [1.0, 0]]),
    ],
)
def test_bmi_get_var_nybtes(bmi_model_initialized, input):
    name = "precipitation"
    m = bmi_model_initialized
    m.input = input
    m._values[name] = m.input
    data = m.get_var_nbytes(name)

    assert input.nbytes == data


@pytest.mark.parametrize(
    "input",
    [
        tensor([0, 0], dtype=int8),
        tensor([0, 1.0, 2], dtype=int16),
        tensor([2, 1, 0], dtype=int32),
        tensor([[0]], dtype=int64),
        tensor([[0, 1.0]], dtype=float16),
        tensor([[0, 1.0], [2, 3.0]], dtype=float32),
        tensor([[3, 2], [1.0, 0]], dtype=float64),
    ],
)
def test_bmi_get_var_type(bmi_model_initialized, input):
    name = "precipitation"
    m = bmi_model_initialized
    m.input = input
    m._values[name] = m.input
    data = m.get_var_type(name)

    assert str(input.dtype).split(".")[1] == data
