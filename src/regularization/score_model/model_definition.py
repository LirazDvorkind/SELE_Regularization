from typing import Tuple

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar diffusion time to a higher-dimensional sinusoidal embedding."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor of shape (batch_size, 1)
        Returns:
            Embedding of shape (batch_size, embed_dim)
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        # t: (B, 1), freqs: (half,) -> args: (B, half)
        args = t * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Hidden layer with a skip connection."""

    def __init__(self, in_dim: int, out_dim: int, use_layer_norm: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.act = nn.Softplus()
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.linear(x))) + self.skip(x)


class ScoreNetwork(nn.Module):
    """
    Score network for diffusion model.

    Takes as input the noisy data concatenated with time and outputs
    the score (gradient of log probability).

    Args:
        input_dim: Dimension of input data (data_dim + 1 for time)
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimension of output (same as data_dim)
        use_layer_norm: Whether to use LayerNorm in hidden layers
        use_residual: Use residual blocks instead of plain sequential layers
        use_time_embedding: Use sinusoidal time embedding (only when use_residual=True)
        time_embed_dim: Dimension of the sinusoidal time embedding
    """

    def __init__(
            self,
            input_dim: int = 33,
            hidden_dims: Tuple[int, ...] = (64, 128, 256, 256, 128, 64),
            output_dim: int = 32,
            use_layer_norm: bool = False,
            use_residual: bool = False,
            use_time_embedding: bool = False,
            time_embed_dim: int = 128,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_time_embedding = use_time_embedding and use_residual

        if self.use_time_embedding:
            self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
            # data_dim = input_dim - 1 (the original +1 was for raw time scalar)
            effective_input_dim = (input_dim - 1) + time_embed_dim
        else:
            self.time_embed = None
            effective_input_dim = input_dim

        if use_residual:
            blocks = []
            prev_dim = effective_input_dim
            for hidden_dim in hidden_dims:
                blocks.append(ResidualBlock(prev_dim, hidden_dim, use_layer_norm))
                prev_dim = hidden_dim
            blocks.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*blocks)
        else:
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Softplus())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights. Kaiming for residual mode, Xavier for legacy."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.use_residual:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                else:
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
        if self.use_time_embedding and self.time_embed is not None:
            t_emb = self.time_embed(t)
            xt = torch.cat([x, t_emb], dim=-1)
        else:
            xt = torch.cat([x, t], dim=-1)
        return self.network(xt)
