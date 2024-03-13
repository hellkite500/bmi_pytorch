"""
Torch model for a multi-layer Neural Network with weights and bias

@author Nels Frazier (nfrazier@lynker.com)
@version 0.1.1

@date 2024-03-11

@copyright Copyright (c) 2024
"""

import torch
from torch.nn import Parameter
from math import sqrt
#Typing imports
from typing import Callable, List, Optional
from torch import Tensor   

from config import Config

class Model(torch.nn.Module):
    """Multi-layer Neural Network Model

    """

    def __init__(self, config: Config, activation: Optional[Callable] = None, dropout_rate: Optional[float] = 0.0):
        """Initialize the model weights and bias for each required layer

        Args:
            input_size (int): number of inputs
            output_size (int): number of outputs
            hidden_sizes (Iterable, optional): Size of each hidden layer to apply. Defaults to [], or no hidden layers.
            activation (Callable, optional): Activation function to use. Defaults to None.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(Model, self).__init__()
        input_size: int = config.input_size
        output_size: int = config.output_size
        hidden_sizes: List[None|int] = config.hidden_size
        
        # Start with empty Parameter layers
        self.weights: List[Parameter] = []
        self.bias: List[Parameter] = []
        
        # Compute the distribution to initialize from based on input size
        self.std_deviation: float = 1.0 / sqrt(input_size)

        current_in: int = input_size
        for layer_size in hidden_sizes[1:]:
            # Create each layer weights and bias and initialize the parameter data
            # from a uniform distribtion
            self.weights.append( Parameter(torch.randn(current_in, layer_size)) )
            self.bias.append( Parameter(torch.randn(layer_size)) )
            current_in = layer_size
            #set initial parameter data for this layer
            self.weights[-1].data.uniform_(-self.std_deviation, self.std_deviation)
            self.bias[-1].data.uniform_(-self.std_deviation, self.std_deviation)
        # Create final layer (or only layer)
        self.weights.append( Parameter(torch.randn(current_in, output_size)) )
        self.bias.append( Parameter(torch.randn(output_size)) )
        #set initial parameter data for the final layer
        self.weights[-1].data.uniform_(-self.std_deviation, self.std_deviation)
        self.bias[-1].data.uniform_(-self.std_deviation, self.std_deviation)
        
        # Hold activation function and dropout rate for use in forward pass
        self.activation: Callable = activation
        self.dropout_rate: float = dropout_rate
        
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the network layers

        Args:
            input (Tensor): Input tensor

        Returns:
            Tensor: Result of tensor applyed to each layer of the Model
        """
        result: Tensor = input
        # Iterate each layer's weight and bias
        for weight, bias in zip(self.weights, self.bias):
            result = torch.matmul(result, weight)+bias
        # Apply activation, if requested
        if self.activation:
            result = self.activation(result)
        result = torch.dropout( result, self.dropout_rate, self.training )
        return result
    