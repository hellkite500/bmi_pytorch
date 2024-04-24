[![PyTest Suite](https://github.com/hellkite500/bmi_pytorch/actions/workflows/pytest.yml/badge.svg)](https://github.com/hellkite500/bmi_pytorch/actions/workflows/pytest.yml)

# Testing BMI with torch tensors

## Installation

### From Github

```shell
# install `bmi_pytorch` package
pip install "git+https://github.com/hellkite500/bmi_pytorch@main#egg=bmi_pytorch&subdirectory=bmi_pytorch"

# install `bmi_sdk` package
pip install "git+https://github.com/hellkite500/bmi_pytorch@main#egg=bmi_sdk&subdirectory=bmi_sdk"
```

### From Source

```shell
# clone repo
git clone https://github.com/hellkite500/bmi_pytorch && cd bmi_pytorch

# create python virtual environment (python >= 3.8 required)
python -m venv venv
source ./venv/bin/activate

# install `bmi_pytorch` package
pip install ./bmi_pytorch

# install `bmi_sdk` package
pip install ./bmi_sdk/
```

### Local Development

#### `pre-commit` hooks

We use `pre-commit` hooks to run tasks like code formatting and linting.
After cloning the repo and setting up a python virtual environment, install `pre-commit` with:

```python
pip install pre-commit
```

Next, run `pre-commit install` (you must from a directory in your repo clone).
That's all!
Now, whenever you make a commit, `pre-commit` will run hooks to ensure that your code is properly formatted and linted!

#### Running Tests

We use `pytest` to write and run our tests. Do the following to build and run tests:

```shell
# clone repo
git clone https://github.com/hellkite500/bmi_pytorch && cd bmi_pytorch

# create python virtual environment (python >= 3.8 required)
python -m venv venv
source ./venv/bin/activate

# install `bmi_pytorch` package
pip install "./bmi_pytorch[test]"
# install `bmi_sdk` package
pip install "./bmi_sdk/[test]"

# run tests
pytest
```

## model.py
Contains an intitial multi-layer torch neural network for testing.

> [!NOTE]
> TODO add a `model_bmi.py` module

> [!NOTE]
> TODO add a Model.update() and connect to BMI update function
