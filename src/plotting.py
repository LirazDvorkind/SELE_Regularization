"""Plotting helpers (matplotlib)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Sequence


def _ensure_results_dir():
    os.makedirs('results', exist_ok=True)


def plot_lcurve(seminorms: Sequence[float], residuals: Sequence[float], kappa_vals,
                knee_idx: int, *, save: bool = False):
    fig, ax = plt.subplots()
    ax.loglog(residuals, seminorms, '-o', markersize=3)
    ax.scatter(residuals[knee_idx], seminorms[knee_idx], marker='x', s=60,
               label=f'κ_knee = {kappa_vals[knee_idx]:.2e}')
    ax.set_xlabel(r'$\varepsilon = ||\,G S - \eta_{\mathrm{ext}}\,||_2$')
    ax.set_ylabel(r'$||\,L S\,||_2$')
    plt.title("Regularization Loss Curve")
    ax.legend()
    if save:
        _ensure_results_dir()
        fig.savefig('results/lcurve.png', dpi=300)
    plt.show(block=False)


def plot_sele(z: Sequence[float], S_mean: Sequence[float], S_std: Sequence[float],
              *, save: bool = False):
    fig, ax = plt.subplots()
    ax.plot(z * 1e4, S_mean, label='SELE')
    ax.fill_between(z, np.asarray(S_mean) - S_std, np.asarray(S_mean) + S_std,
                    alpha=0.3, label=r'$\pm 1\,\sigma$')
    ax.set_xlabel('z $[\\mu m]$')
    ax.set_ylabel('SELE')
    plt.title("SELE Plot")
    ax.legend()
    if save:
        _ensure_results_dir()
        fig.savefig('results/sele_profile.png', dpi=300)
    plt.show(block=False)


def plot_eta(lambda_vals, eta_meas, eta_fit, *, save: bool = False):
    fig, ax = plt.subplots()
    ax.plot(lambda_vals, eta_meas, label='Measured')
    ax.plot(lambda_vals, eta_fit, '--', label='Reconstructed')
    ax.set_xlabel('Wavelength (arb. index)')
    ax.set_ylabel(r'$\eta_{ext}$')
    ax.legend()
    plt.title("Reconstructed ELE")
    if save:
        _ensure_results_dir()
        fig.savefig('results/eta_fit.png', dpi=300)
    plt.show(block=False)


def plot_interpolation_check(
        z_old: np.ndarray,
        z_new: np.ndarray,
        G_old: np.ndarray,
        G_new: np.ndarray,
        wav_idx: int = 100,
        *,
        save: bool = False
) -> None:
    # plot against left edges (no centre shift)
    fig = plt.figure()
    plt.plot(z_old * 1e4, G_old[wav_idx], "o-", label="original G")
    plt.plot(z_new * 1e4, G_new[wav_idx], ".-", label="interpolated G")
    plt.xlabel("depth z $[\\mu m]$")
    plt.ylabel("integrated ΔG per bin")
    plt.title(f"G interpolation check – λ index {wav_idx}")
    plt.legend()
    if save:
        _ensure_results_dir()
        fig.savefig('results/interpolated_G.png', dpi=300)
    plt.show(block=True)


# Like S3.2 figure in SI paper
def plot_mesh_elements_position_and_size(z: np.ndarray, z_turn: float, *, save: bool = False) -> None:
    """
    z: new mesh in [cm]
    z_turn: where we turn from lin to exp
    """
    mesh_sizes = np.diff(z)
    mesh_positions = z[:-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Left subplot (full range)
    ax1.plot(mesh_positions * 1e4, mesh_sizes * 1e4, 'k.', markersize=6)
    ax1.set_xlabel(r'Position [$\mu$m]')
    ax1.set_ylabel(r'Element Size ($\mu$m)')

    # Right subplot (zoom at start)
    ax2.plot(mesh_positions * 1e4, mesh_sizes * 1e4, 'k.', markersize=6)
    ax2.set_xlabel(r'Position [$\mu$m]')
    ax2.set_ylabel(r'Element Size ($\mu$m)')
    ax2.set_xlim(0, z_turn * 1e4 * 2)  # zoom in horizontally

    fig.suptitle("Mesh elements size and position.")
    ax1.set_title("High level view")
    ax2.set_title("Linear zoomed-in view")
    if save:
        _ensure_results_dir()
        fig.savefig('results/mesh.png', dpi=300)
    plt.show(block=False)
