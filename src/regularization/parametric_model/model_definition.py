"""A network that reads an ELE measurement and returns a Gaussian over the five parameters.

The forward problem is strongly many-to-one: the feasibility work measured only about two
effective degrees of freedom resolvable against five real parameters. A network that emitted
five numbers would have to pick one point out of a ridge of equally good answers and could
not say so. Predicting a *full* covariance lets it report "these two trade off against each
other" -- a statement five independent error bars cannot make.

Trained on clean, noise-free data, so the spread it learns is the ambiguity of the physics
itself rather than measurement noise.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

# Log-amplitude scalar plus the amplitude-normalised log shape.
DEFAULT_INPUT_DIM = 29
N_PARAMETERS = 5
# Strictly positive floor on the Cholesky diagonal. Without it the likelihood is unbounded:
# the network can drive a variance to zero and collect infinite reward on one sample.
MIN_SCALE = 1e-4


def cholesky_entries(n_parameters: int = N_PARAMETERS) -> int:
    return n_parameters * (n_parameters + 1) // 2


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
        self.act = nn.SiLU()
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.linear(x))) + self.skip(x)


class ParametricPosteriorNetwork(nn.Module):
    """ELE -> ``(mu, scale_tril)`` of a Gaussian over normalised parameters."""

    def __init__(
            self,
            input_dim: int = DEFAULT_INPUT_DIM,
            hidden_dims: Tuple[int, ...] = (256, 256, 256),
            n_parameters: int = N_PARAMETERS,
            use_layer_norm: bool = True,
            use_residual: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_parameters = n_parameters

        layers = []
        prev = input_dim
        for width in hidden_dims:
            if use_residual:
                layers.append(ResidualBlock(prev, width, use_layer_norm))
            else:
                layers.append(nn.Linear(prev, width))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(width))
                layers.append(nn.SiLU())
            prev = width
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev, n_parameters)
        self.scale_head = nn.Linear(prev, cholesky_entries(n_parameters))

        self._tril_rows, self._tril_cols = torch.tril_indices(n_parameters, n_parameters)
        self._diagonal_mask = self._tril_rows == self._tril_cols

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Start near unit variance rather than near zero, so early training is not dominated
        # by an exploding log-determinant term.
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        mean = self.mean_head(features)
        scale_tril = self._assemble_scale_tril(self.scale_head(features))
        return mean, scale_tril

    def _assemble_scale_tril(self, raw: torch.Tensor) -> torch.Tensor:
        rows = self._tril_rows.to(raw.device)
        cols = self._tril_cols.to(raw.device)
        mask = self._diagonal_mask.to(raw.device)

        values = torch.where(mask, nn.functional.softplus(raw) + MIN_SCALE, raw)
        scale_tril = raw.new_zeros(raw.shape[0], self.n_parameters, self.n_parameters)
        scale_tril[:, rows, cols] = values
        return scale_tril


def build_parametric_network(model_config: dict) -> nn.Module:
    """Construct from a checkpoint's ``config`` dict, tolerating older/newer keys."""
    return ParametricPosteriorNetwork(
        input_dim=model_config.get("input_dim", DEFAULT_INPUT_DIM),
        hidden_dims=tuple(model_config.get("hidden_dims", (256, 256, 256))),
        n_parameters=model_config.get("n_parameters", N_PARAMETERS),
        use_layer_norm=model_config.get("use_layer_norm", True),
        use_residual=model_config.get("use_residual", True),
    )


def gaussian_nll(
        mean: torch.Tensor,
        scale_tril: torch.Tensor,
        target: torch.Tensor,
) -> torch.Tensor:
    """Mean negative log-likelihood of ``target`` under ``N(mean, LL^T)``.

    Trains the mean and the covariance together; a plain squared error would leave the
    covariance with no gradient at all.
    """
    distribution = torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
    return -distribution.log_prob(target).mean()
