"""
Reverse Diffusion Generation Test

Tests whether trained score models can unconditionally generate plausible SELE
profiles by running reverse VP-SDE (Euler-Maruyama) from Gaussian noise.

For each of the 3 models (Alon's d32, my d32, my d500):
  1. Start from 5 independent Gaussian noise vectors
  2. Integrate the reverse SDE from t≈1 down to T_STOP
  3. Read out the clean curve from the final state via Tweedie's formula
  4. Display all 5 generated curves in a figure

Run from repo root:
    python -m src.regularization.score_model.standalones.new_model_toolkit.test_diffusion_generation
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.regularization.score_model.score_model_grad import load_score_model

_REPO_ROOT = Path(__file__).resolve().parents[5]

MODELS: dict[str, str] = {
    "Alon's d32": str(_REPO_ROOT / "Data" / "score_model" / "models" / "alon_sele_score_net_d32.pt"),
    "My d32":     str(_REPO_ROOT / "Data" / "score_model" / "models" / "sele_score_net_d32.pt"),
    "My d500":    str(_REPO_ROOT / "Data" / "score_model" / "models" / "sele_score_net_d500.pt"),
}

N_SAMPLES = 5
N_STEPS = 1000
DEVICE = torch.device("cpu")

# Where to stop the reverse SDE. The true score grows like 1/sigma_t as t -> 0 and
# every trained net saturates well before time_eps, so integrating further degrades
# the sample rather than refining it.
T_STOP = 0.02

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


def _vp_coefficients(t: float, beta_min: float, beta_max: float) -> tuple[float, float]:
    """VP-SDE marginal coefficients (a_t, sigma_t) for x_t = a_t * x_0 + sigma_t * z."""
    int_beta = (beta_min + 0.5 * (beta_max - beta_min) * t) * t
    return math.exp(-0.5 * int_beta), math.sqrt(-math.expm1(-int_beta))


def _power_law_time_grid(t_stop: float, t_start: float, n_steps: int, power: float = 3.0) -> np.ndarray:
    """
    Reverse-time grid concentrated near t -> t_stop.

    All shape information in this VP-SDE schedule lives at sigma_t < 0.2, i.e.
    t < ~0.06 (see plans/score-network-conv1d-explanation.md); a linear grid spends
    ~94% of its steps where the state is indistinguishable from noise. u is uniform
    on [0, 1]; t(u) = t_stop + (t_start - t_stop) * (1 - u)^power has |dt/du| -> 0 as
    u -> 1 for power > 1, so grid points cluster near t = t_stop.
    """
    u = np.linspace(0.0, 1.0, n_steps + 1)
    return t_stop + (t_start - t_stop) * (1.0 - u) ** power


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
    t_stop: float = T_STOP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate samples via Euler-Maruyama on the reverse VP-SDE.

    Integrates from t = 1-time_eps -> t_stop (high noise -> low noise) on a
    power-law grid concentrated near t -> t_stop.

    Reverse SDE:
        dx = [-0.5 * beta(t) * x - beta(t) * score(x, t)] * dt + sqrt(beta(t)) * dW
    where dt < 0 (time going backward), so with positive step size h:
        x_prev = x + [0.5 * beta(t) * x + beta(t) * score(x, t)] * h + sqrt(beta(t) * h) * z

    Returns (denoised, raw), each of shape (n_samples, target_length) in physical
    SELE units. `raw` is the SDE state itself, which by construction still carries
    sigma_t of white noise; `denoised` is the Tweedie posterior mean of the clean
    curve given that state, and is the one worth looking at.
    """
    norm_scale = 2.0 / (d_max - d_min)

    # Start from unit Gaussian -- approximate x at t~1 which is nearly pure noise
    x = torch.randn(n_samples, target_length, device=DEVICE)

    time_grid = _power_law_time_grid(t_stop, 1.0 - time_eps, n_steps)

    model.eval()
    with torch.no_grad():
        for t_val, t_next in zip(time_grid[:-1], time_grid[1:]):
            h = float(t_val - t_next)  # positive step size, varies across the grid
            t_tensor = torch.full((n_samples, 1), t_val, dtype=torch.float32, device=DEVICE)

            beta_t = beta_min + (beta_max - beta_min) * t_val
            score = model(x, t_tensor)  # (n_samples, target_length)

            # Reverse drift (negating the forward drift sign)
            drift = 0.5 * beta_t * x + beta_t * score
            diffusion_coef = (beta_t * h) ** 0.5
            noise = torch.randn_like(x)

            x = x + drift * h + diffusion_coef * noise

        # Tweedie readout: E[x_0 | x_t] = (x_t + sigma_t^2 * score) / a_t. Without
        # this the returned state still holds the schedule's own residual noise, so
        # even a perfect score model would look ~900x rougher than the training data.
        t_tensor = torch.full((n_samples, 1), t_stop, dtype=torch.float32, device=DEVICE)
        a_t, sigma_t = _vp_coefficients(t_stop, beta_min, beta_max)
        x0 = (x + sigma_t ** 2 * model(x, t_tensor)) / a_t

    # Denormalize: S_norm in [-1, 1] -> physical units
    to_phys = lambda v: (v.numpy() + 1.0) / norm_scale + d_min
    return to_phys(x0), to_phys(x)
    return S_phys


def compute_roughness(curves: np.ndarray) -> np.ndarray:
    """
    Per-curve roughness: rms(second difference) / std.

    Scale-free measure of point-to-point jitter relative to the curve's own size
    (see plans/score-network-conv1d-explanation.md). ~0.0003 for real SELE curves;
    O(1) for near-white-noise curves.
    """
    d2 = np.diff(curves, n=2, axis=1)
    rms = np.sqrt(np.mean(d2 ** 2, axis=1))
    std = curves.std(axis=1)
    return rms / std


_REFERENCE_CURVES_PATH = _REPO_ROOT / "Data" / "score_model" / "datasets" / "sele_simulated_1000_curves_500_long.csv"


def reference_roughness(target_length: int) -> float:
    """Median roughness of real training curves, downsampled to target_length to match generated samples."""
    real = np.loadtxt(_REFERENCE_CURVES_PATH, delimiter=",")
    if real.shape[1] != target_length:
        indices = np.round(np.linspace(0, real.shape[1] - 1, target_length)).astype(int)
        real = real[:, indices]
    return float(np.median(compute_roughness(real)))


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
    ax.set_title(f"Reverse-diffusion generated SELE — {model_name} (Tweedie readout @ t={T_STOP})")
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


def run(model_path: str, model_name: str = "New model") -> None:
    """Generate and plot reverse-diffusion samples for a single checkpoint."""
    print(f"\n--- Loading: {model_name} ---")
    print(f"    {model_path}")

    model, d_min, d_max, target_length, beta_min, beta_max, time_eps = _load_model(model_name, model_path)

    print(f"    target_length={target_length}, beta=[{beta_min}, {beta_max}], time_eps={time_eps}")
    print(f"    data range: [{d_min:.4f}, {d_max:.4f}]")
    print(f"    Generating {N_SAMPLES} samples with {N_STEPS} reverse steps...")

    samples, samples_raw = reverse_diffusion_sample(
        model=model,
        d_min=d_min,
        d_max=d_max,
        target_length=target_length,
        beta_min=beta_min,
        beta_max=beta_max,
        time_eps=time_eps,
        n_samples=N_SAMPLES,
        n_steps=N_STEPS,
        t_stop=T_STOP,
    )

    print(f"    Generated sample range: [{samples.min():.4f}, {samples.max():.4f}]")

    gen_roughness = float(np.median(compute_roughness(samples)))
    raw_roughness = float(np.median(compute_roughness(samples_raw)))
    ref_roughness = reference_roughness(target_length)
    print(
        f"    Roughness (median rms(d2)/std): denoised={gen_roughness:.4g}, "
        f"raw SDE state={raw_roughness:.4g}, real={ref_roughness:.4g}, "
        f"ratio={gen_roughness / ref_roughness:.1f}x"
    )

    plot_generated_samples(model_name, samples)


def main() -> None:
    for model_name, model_path in MODELS.items():
        try:
            run(model_path, model_name)
        except Exception as e:
            print(f"    ERROR loading model: {e}")
            continue

    plt.show(block=True)


if __name__ == "__main__":
    main()
