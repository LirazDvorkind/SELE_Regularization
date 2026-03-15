from typing import Tuple

import torch
import torch.nn as nn


class ScoreNetwork(nn.Module):
    """
    Score network for diffusion model.

    Takes as input the noisy data concatenated with time and outputs
    the score (gradient of log probability).

    Args:
        input_dim: Dimension of input data (data_dim + 1 for time)
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output (same as data_dim)
    """

    def __init__(
            self,
            input_dim: int = 33,
            hidden_dims: Tuple[int, ...] = (64, 128, 256, 256, 128, 64),
            output_dim: int = 32
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Softplus()  # More stable than LogSigmoid
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the score network.

        Args:
            x: Noisy data tensor of shape (batch_size, data_dim)
            t: Time tensor of shape (batch_size, 1)

        Returns:
            Score tensor of shape (batch_size, data_dim)
        """
        xt = torch.cat([x, t], dim=-1)
        return self.network(xt)