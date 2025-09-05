"""Plotting helpers (matplotlib)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Sequence
import mplcursors


def _ensure_results_dir():
    os.makedirs('results', exist_ok=True)


def plot_lcurve(seminorms: Sequence[float], residuals: Sequence[float], kappa_vals,
                knee_idx: int, mask: Sequence[bool], *, save: bool = False):
    seminorms = np.asarray(seminorms)
    residuals = np.asarray(residuals)
    mask = np.asarray(mask, dtype=bool)

    fig, ax = plt.subplots()

    # Main curve (Line2D)
    line, = ax.loglog(residuals, seminorms, '-o', markersize=3, color="C0")

    # Highlight masked points (PathCollection)
    mask = np.asarray(mask, dtype=bool)
    idx_mask = np.flatnonzero(mask)
    sc_mask = ax.scatter(residuals[mask], seminorms[mask],
                         c="red", s=20, label="Conf window")

    # Knee point (single PathCollection)
    idx_knee = np.array([knee_idx])
    sc_knee = ax.scatter(residuals[knee_idx], seminorms[knee_idx],
                         marker='x', s=60, color="black",
                         label=f'κ_knee = {kappa_vals[knee_idx]:.2e}')

    # One cursor for all artists; show κ on hover
    cursor = mplcursors.cursor([line, sc_mask, sc_knee], hover=True)

    # Build per-artist index mapping back to kappa indices
    index_map = {
        line: np.arange(len(kappa_vals), dtype=int),
        sc_mask: idx_mask.astype(int),
        sc_knee: np.array([knee_idx], dtype=int),
    }

    @cursor.connect("add")
    def _(sel):
        artist = sel.artist
        mapping = index_map.get(artist)
        if mapping is None or len(mapping) == 0:
            sel.annotation.set_text("κ = n/a")
            return
        i_local = sel.index
        i_local = 0 if i_local is None else int(i_local)
        i_global = int(mapping[i_local])
        sel.annotation.set_text(f"κ = {kappa_vals[i_global]:.2e}")

    ax.set_xlabel(r'$\varepsilon = ||\,G S - \eta_{\mathrm{ext}}\,||_2$')
    ax.set_ylabel(r'$||\,L S\,||_2$')
    plt.title("Regularization Loss Curve")
    ax.legend()

    if save:
        _ensure_results_dir()
        fig.savefig('results/lcurve.png', dpi=300)

    plt.show(block=False)


def plot_sele(z_centres, S_mean, S_std, sele_gt, z_gt, *, save: bool = False):
    mask = z_gt <= np.max(z_centres)
    sele_gt = sele_gt[mask]
    z_gt = z_gt[mask]
    fig, ax = plt.subplots()
    ax.plot(z_centres * 1e4, S_mean, label='SELE (reconstructed)')
    ax.fill_between(z_centres * 1e4,
                    np.asarray(S_mean) - S_std,
                    np.asarray(S_mean) + S_std,
                    alpha=0.3, label=r'$\pm 1\,\sigma$')

    ax.plot(z_gt * 1e4, sele_gt, 'k--', label='SELE ground truth')

    ax.set_xlabel('z $[\\mu m]$')
    ax.set_ylabel('SELE')
    plt.title("SELE vs Ground Truth")
    ax.legend()
    if save:
        _ensure_results_dir()
        fig.savefig('results/sele_profile.png', dpi=300)
    plt.show(block=False)


def plot_eta(lambda_vals, eta_meas, eta_fit, *, save: bool = False):
    fig, ax = plt.subplots()
    ax.plot(lambda_vals, eta_meas, label='Measured')
    ax.plot(lambda_vals, eta_fit, '--', label='Reconstructed')
    ax.set_xlabel('Wavelength [nm]')
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
