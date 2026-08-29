from typing import Optional, Tuple

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


def _default_dilations(target_length: int) -> Tuple[int, ...]:
    """Dilation schedule sized to the curve length.

    Kernel size 5, two convs per block: receptive field grows by 8*dilation per
    block (see plans/score-network-conv1d-explanation.md, section 6). The d500
    list reaches RF 509 by the 6th block; d32 caps at 4 blocks (RF well past 32).
    """
    if target_length <= 64:
        return (1, 2, 4, 8)
    return (1, 2, 4, 8, 16, 32)


class TimeMLP(nn.Module):
    """Sinusoidal time embedding through a shared 2-layer MLP, read by every block's FiLM."""

    def __init__(self, sinusoidal_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(sinusoidal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.time_embed(t))


class FiLM(nn.Module):
    """Per-channel scale-and-shift from the time embedding, broadcast across depth.

    gamma/beta are shape (B, channels) -- one value per channel, not per depth --
    because noise level is a property of the whole curve, uniform across depth.
    """

    def __init__(self, t_emb_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(t_emb_dim, 2 * channels)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(t_emb).chunk(2, dim=-1)
        return h * gamma.unsqueeze(-1) + beta.unsqueeze(-1)


class DilatedConvBlock(nn.Module):
    """Conv1d -> GroupNorm -> SiLU -> FiLM(t) -> Conv1d -> GroupNorm -> SiLU, with an identity skip."""

    def __init__(
            self,
            channels: int,
            kernel_size: int,
            dilation: int,
            t_emb_dim: int,
            n_groups: int = 8,
            use_norm: bool = True,
    ):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        conv_kwargs = dict(kernel_size=kernel_size, dilation=dilation, padding=pad, padding_mode='replicate')
        self.conv1 = nn.Conv1d(channels, channels, **conv_kwargs)
        self.conv2 = nn.Conv1d(channels, channels, **conv_kwargs)
        self.norm1 = nn.GroupNorm(n_groups, channels) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(n_groups, channels) if use_norm else nn.Identity()
        self.film = FiLM(t_emb_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.film(h, t_emb)
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class Conv1dScoreNetwork(nn.Module):
    """
    1D dilated residual convolutional score network.

    Replaces ScoreNetwork's fully-connected layers -- which have no structural
    notion of depth adjacency -- with a stack of dilated convolutions sharing
    weights across depth, so smoothness is architectural rather than something
    the network must infer statistically. See
    plans/score-network-conv1d-explanation.md for the full rationale.

    Args:
        target_length: number of depth points (SELE curve length).
        channels: internal channel count carried at every depth point.
        n_blocks: number of dilated residual blocks. None = derive from
            target_length via _default_dilations.
        kernel_size: convolution kernel width (bidirectional, not causal --
            depth has no preferred direction).
        dilations: explicit dilation per block. None = _default_dilations(target_length),
            optionally truncated to n_blocks.
        use_norm: GroupNorm on/off. GroupNorm normalizes across the length axis and
            strips per-sample amplitude, which is this dataset's dominant variance
            direction -- disable if trained samples show collapsed amplitude spread.
    """

    def __init__(
            self,
            target_length: int = 500,
            channels: int = 128,
            n_blocks: Optional[int] = None,
            kernel_size: int = 5,
            dilations: Optional[Tuple[int, ...]] = None,
            sinusoidal_dim: int = 128,
            time_hidden_dim: int = 256,
            n_groups: int = 8,
            use_norm: bool = True,
    ):
        super().__init__()
        if dilations is None:
            dilations = _default_dilations(target_length)
        if n_blocks is not None:
            dilations = dilations[:n_blocks]
        self.target_length = target_length

        self.time_mlp = TimeMLP(sinusoidal_dim, time_hidden_dim)
        self.stem = nn.Conv1d(
            1, channels, kernel_size, padding=kernel_size // 2, padding_mode='replicate'
        )
        self.blocks = nn.ModuleList([
            DilatedConvBlock(channels, kernel_size, d, time_hidden_dim, n_groups, use_norm)
            for d in dilations
        ])
        self.head_norm = nn.GroupNorm(n_groups, channels) if use_norm else nn.Identity()
        self.head_act = nn.SiLU()
        self.head_conv = nn.Conv1d(channels, 1, kernel_size=1)

        # Zero-initialize the output conv: the network starts by predicting zero
        # score, so early training does not have to fight large random outputs.
        nn.init.zeros_(self.head_conv.weight)
        nn.init.zeros_(self.head_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy data tensor of shape (batch_size, target_length)
            t: Time tensor of shape (batch_size, 1)
        Returns:
            Score tensor of shape (batch_size, target_length)
        """
        t_emb = self.time_mlp(t)
        h = x.unsqueeze(1)  # (B, 1, L)
        h = self.stem(h)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.head_act(self.head_norm(h))
        h = self.head_conv(h)
        return h.squeeze(1)


def build_score_network(model_config: dict) -> nn.Module:
    """
    Construct a score network from a checkpoint's (or TrainingConfig's) config dict.

    Dispatches on model_config['arch']: an absent key defaults to 'mlp', so every
    checkpoint saved before this factory existed keeps loading as ScoreNetwork
    unchanged. Pass 'conv1d' to build Conv1dScoreNetwork instead.
    """
    arch = model_config.get('arch', 'mlp')
    target_length = model_config['target_length']

    if arch == 'mlp':
        return ScoreNetwork(
            input_dim=target_length + 1,
            output_dim=target_length,
            hidden_dims=model_config['hidden_dims'],
            use_layer_norm=model_config.get('use_layer_norm', False),
            use_residual=model_config.get('use_residual', False),
            use_time_embedding=model_config.get('use_time_embedding', False),
            time_embed_dim=model_config.get('time_embed_dim', 128),
        )
    elif arch == 'conv1d':
        dilations = model_config.get('dilations')
        return Conv1dScoreNetwork(
            target_length=target_length,
            channels=model_config.get('channels', 128),
            n_blocks=model_config.get('n_blocks'),
            kernel_size=model_config.get('kernel_size', 5),
            dilations=tuple(dilations) if dilations else None,
        )
    else:
        raise ValueError(f"Unknown score network arch: {arch!r}")
