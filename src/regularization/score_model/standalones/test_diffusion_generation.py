"""
Reverse Diffusion Generation Test

Tests whether trained score models can unconditionally generate plausible SELE
profiles by running reverse VP-SDE (Euler-Maruyama) from Gaussian noise.

For each of the 3 models (Alon's d32, my d32, my d500):
  1. Start from 5 independent Gaussian noise vectors
  2. Integrate the reverse SDE from t≈1 → t≈0
  3. Display all 5 generated curves in a figure

Run from repo root:
    python src/regularization/score_model/standalones/test_diffusion_generation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.regularization.score_model.score_model_grad import load_score_model

_REPO_ROOT = Path(__file__).resolve().parents[4]

MODELS: dict[str, str] = {
    "Alon's d32": str(_REPO_ROOT / "Data" / "score_model" / "models" / "alon_sele_score_net_d32.pt"),
    "My d32":     str(_REPO_ROOT / "Data" / "score_model" / "models" / "sele_score_net_d32.pt"),
    "My d500":    str(_REPO_ROOT / "Data" / "score_model" / "models" / "sele_score_net_d500.pt"),
}

N_SAMPLES = 5
N_STEPS = 1000
DEVICE = torch.device("cpu")

# Diffusion schedule defaults (same as training defaults in TrainingConfig)
_DEFAULT_BETA_MIN = 0.1
_DEFAULT_BETA_MAX = 20.0
_DEFAULT_TIME_EPS = 1e-4


class _AlonModelWrapper(nn.Module):
    """Wraps Alon's raw Sequential so it accepts (x, t) separately, like ScoreNetwork."""

    def __init__(self, seq_model: nn.Sequential) -> None:
        super().__init__()
        self.model = seq_model

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (batch, 32), t: (batch, 1)  →  concat to (batch, 33)
        return self.model(torch.cat([x, t], dim=-1))


def reverse_diffusion_sample(
    model: torch.nn.Module,
    d_min: float,
    d_max: float,
    target_length: int,
    beta_min: float,
    beta_max: float,
    time_eps: float,
    n_samples: int = N_SAMPLES,
    n_steps: int = N_STEPS,
) -> np.ndarray:
    """
    Generate samples via Euler-Maruyama on the reverse VP-SDE.

    Integrates from t = 1-time_eps → time_eps (high noise → low noise).

    Reverse SDE:
        dx = [-0.5 * beta(t) * x - beta(t) * score(x, t)] * dt + sqrt(beta(t)) * dW
    where dt < 0 (time going backward), so with positive step size h:
        x_prev = x + [0.5 * beta(t) * x + beta(t) * score(x, t)] * h + sqrt(beta(t) * h) * z

    Returns array of shape (n_samples, target_length) in physical SELE units.
    """
    norm_scale = 2.0 / (d_max - d_min)

    # Start from unit Gaussian — approximate x at t≈1 which is nearly pure noise
    x = torch.randn(n_samples, target_length, device=DEVICE)

    time_grid = np.linspace(1.0 - time_eps, time_eps, n_steps + 1)
    h = float(time_grid[0] - time_grid[1])  # positive step size

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(time_grid[:-1]):
            t_tensor = torch.full((n_samples, 1), t_val, dtype=torch.float32, device=DEVICE)

            beta_t = beta_min + (beta_max - beta_min) * t_val
            score = model(x, t_tensor)  # (n_samples, target_length)

            # Reverse drift (negating the forward drift sign)
            drift = 0.5 * beta_t * x + beta_t * score
            diffusion_coef = (beta_t * h) ** 0.5
            noise = torch.randn_like(x)

            x = x + drift * h + diffusion_coef * noise

    # Denormalize: S_norm ∈ [-1, 1] → physical units
    x_np = x.numpy()
    S_phys = (x_np + 1.0) / norm_scale + d_min
    return S_phys


def plot_generated_samples(model_name: str, samples: np.ndarray) -> None:
    """Plot 5 generated SELE profiles on a single figure."""
    n_pts = samples.shape[1]
    W_cm = 30e-4  # wafer width in cm
    z_um = np.linspace(0, W_cm * 1e4, n_pts)  # µm

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, N_SAMPLES))
    for i, (curve, color) in enumerate(zip(samples, colors)):
        ax.plot(z_um, curve, color=color, alpha=0.85, linewidth=1.5, label=f"Sample {i+1}")

    ax.set_xlabel("Depth z [µm]")
    ax.set_ylabel("SELE")
    ax.set_title(f"Reverse-diffusion generated SELE — {model_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()


def _load_model(model_name: str, model_path: str):
    """Load any of the 3 checkpoints, returning (model, d_min, d_max, target_length, beta_min, beta_max, time_eps)."""
    raw = torch.load(model_path, map_location=DEVICE, weights_only=False)

    if isinstance(raw, nn.Sequential):
        # Alon's legacy format: just a bare Sequential, no config dict
        model = _AlonModelWrapper(raw)
        model.eval()
        # d_min/d_max hardcoded from test-score-models.py (same values used there)
        d_min = 2.7389012e-21
        d_max = 0.03475773
        target_length = 32
        beta_min, beta_max, time_eps = _DEFAULT_BETA_MIN, _DEFAULT_BETA_MAX, _DEFAULT_TIME_EPS
    else:
        # Standard checkpoint dict format
        score_network, d_min, d_max, target_length = load_score_model(model_path)
        cfg = raw.get("config", {})
        beta_min = cfg.get("beta_min", _DEFAULT_BETA_MIN)
        beta_max = cfg.get("beta_max", _DEFAULT_BETA_MAX)
        time_eps = cfg.get("time_eps", _DEFAULT_TIME_EPS)
        model = score_network

    return model, d_min, d_max, target_length, beta_min, beta_max, time_eps


def main() -> None:
    for model_name, model_path in MODELS.items():
        print(f"\n--- Loading: {model_name} ---")
        print(f"    {model_path}")

        try:
            model, d_min, d_max, target_length, beta_min, beta_max, time_eps = _load_model(model_name, model_path)
        except Exception as e:
            print(f"    ERROR loading model: {e}")
            continue

        print(f"    target_length={target_length}, beta=[{beta_min}, {beta_max}], time_eps={time_eps}")
        print(f"    data range: [{d_min:.4f}, {d_max:.4f}]")
        print(f"    Generating {N_SAMPLES} samples with {N_STEPS} reverse steps...")

        samples = reverse_diffusion_sample(
            model=model,
            d_min=d_min,
            d_max=d_max,
            target_length=target_length,
            beta_min=beta_min,
            beta_max=beta_max,
            time_eps=time_eps,
            n_samples=N_SAMPLES,
            n_steps=N_STEPS,
        )

        print(f"    Generated sample range: [{samples.min():.4f}, {samples.max():.4f}]")
        plot_generated_samples(model_name, samples)

    plt.show(block=True)


if __name__ == "__main__":
    main()
