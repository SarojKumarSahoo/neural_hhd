import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradSineLayer(nn.Module):
    """
    Defines a modified linear layer with sinusoidal activation function.
    """

    def __init__(
        self, input_dim: int, output_dim: int, omega: int = 30, include_bias: bool = True, first_layer: bool = False
    ):
        """
        Initialize the layer.

        Parameters:
        - input_dim: Number of input features.
        - output_dim: Number of output features.
        - omega: Scaling factor for sinusoidal activation.
        - include_bias: Whether to include a bias term.
        - first_layer: Whether this is the first layer of the network.
        """
        super(GradSineLayer, self).__init__()
        self.input_dim = input_dim
        self.omega = omega
        self.first_layer = first_layer

        # Define the linear layer
        self.linear = nn.Linear(input_dim, output_dim, bias=include_bias)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights based on whether it's the first layer or not.
        """
        with torch.no_grad():
            if self.first_layer:
                self.linear.weight.uniform_(-1 / self.input_dim, 1 / self.input_dim)
            else:
                bound = np.sqrt(6 / self.input_dim) / self.omega
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, activations: torch.Tensor, prev_derivatives: torch.Tensor):
        """
        Forward pass for the layer.

        Parameters:
        - activations: Activations from the previous layer.
        - prev_derivatives: Derivatives of activations from the previous layer.
        """
        # Compute pre-activations
        pre_activations = self.omega * self.linear(activations)

        # Apply sinusoidal activation
        new_activations = torch.sin(pre_activations)

        # Compute derivatives using the chain rule
        derivatives = self.omega * F.linear(prev_derivatives, self.linear.weight, None)
        new_derivatives = torch.cos(pre_activations).unsqueeze(-2) * derivatives

        return new_activations, new_derivatives


class GradSiren(nn.Module):
    """
    Neural network model using GradSineLayer modules.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_layers: int = 6, omega: int = 30):
        """
        Initialize the network.

        Parameters:
        - input_dim: Number of input features.
        - output_dim: Number of output features.
        - hidden_dim: Number of features in hidden layers.
        - num_layers: Total number of layers.
        - omega: Scaling factor for sinusoidal activation.
        """
        super(GradSiren, self).__init__()

        layers = []
        for idx in range(num_layers - 1):
            in_dim = input_dim if idx == 0 else hidden_dim
            out_dim = hidden_dim
            layers.append(GradSineLayer(in_dim, out_dim, omega, first_layer=(idx == 0)))

        # Add the final linear layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass for the network.
        """
        activations = inputs
        # Initialize derivatives with identity matrix for input dimensions
        derivatives_shape = list(inputs.shape) + [inputs.shape[-1]]
        derivatives = torch.zeros(derivatives_shape, dtype=inputs.dtype, device=inputs.device)
        derivatives[..., torch.arange(inputs.shape[-1]), torch.arange(inputs.shape[-1])] = 1

        for layer in self.network:
            if isinstance(layer, GradSineLayer):
                activations, derivatives = layer(activations, derivatives)
            else:
                derivatives = F.linear(derivatives, layer.weight, None)

        return derivatives.squeeze(-1)
