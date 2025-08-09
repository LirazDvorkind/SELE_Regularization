"""High‑level orchestration of the SELE extraction workflow (non‑uniform mesh version)."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.io import load_eta, load_G, load_z, save_csv
from src.operators import build_L
from src.tikhonov import sweep_kappa, find_knee
from src.plotting import plot_lcurve, plot_sele, plot_interpolation_check, plot_eta
from src.mesh import remesh_G

def _edges_from_centres(centres: np.ndarray) -> np.ndarray:
    """Return *edges* given a strictly increasing list of *centres*."""
    centres = np.asarray(centres, float).ravel()
    dz = np.diff(centres)
    edges = np.empty(centres.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * dz[0]
    edges[-1] = centres[-1] + 0.5 * dz[-1]
    return edges


def run_regularization(
    *,
    eta_path: str,
    G_path: str,
    z_path: str,
    L_flag: str = "L0",
    kappa_max: float = 1e-2,
    kappa_min: float = 1e-15,
    n_kappa: int = 200,
    conf_fact: float = 10.0,
    is_save_plots: bool = True,
    e_charge: float = 1.60217657e-19,
    photon_flux: float = 1e14,
):
    """Full regularisation pipeline supporting arbitrary 1‑D meshes."""
    # 1. Load data ---------------------------------------------------------
    eta_ext = load_eta(eta_path)
    G = load_G(G_path)
    z = load_z(z_path).ravel()

    # Create non uniform z and interpolate new G
    z_new, G_new = remesh_G(z, G)
    plot_interpolation_check(z, z_new, G, G_new, save=True)
    # use G_new and z_new from here onward
    G, z = G_new, z_new

    # 2. Unit normalisation (A and B must have same units)
    G = G * photon_flux * e_charge
    B = eta_ext * photon_flux * e_charge

    if G.shape[0] != eta_ext.size:
        raise ValueError("Row mismatch between G and η_ext")

    # 3. Regularisation operator
    L = build_L(L_flag, len(z))

    # 4. κ‑sweep
    kappa_vals = np.logspace(np.log10(kappa_max), np.log10(kappa_min), n_kappa)
    residuals, seminorms, S_list = sweep_kappa(G, B, L, kappa_vals)

    # 5. Knee detection
    kappa_knee, knee_idx = find_knee(residuals, seminorms, kappa_vals)
    # kappa_knee /= 7
    print(f"kappa_knee = {kappa_knee:.3e}")

    # 6. Confidence window
    mask = (kappa_vals >= kappa_knee / conf_fact) & (kappa_vals <= kappa_knee * conf_fact)
    S_stack = np.stack([S_list[i] for i, m in enumerate(mask) if m], axis=1)
    S_mean = S_stack.mean(axis=1)
    S_std = S_stack.std(axis=1)

    # 7. Reconstruction @ kappa_knee
    S_knee = S_list[knee_idx]
    eta_fit = G @ S_knee / (photon_flux * e_charge)

    # 8. Persist results
    Path("results").mkdir(exist_ok=True)
    save_csv("results/S_mean.csv", np.column_stack([z, S_mean]), header="z_cm,S_mean")
    save_csv("results/S_std.csv", np.column_stack([z, S_std]), header="z_cm,S_std")
    save_csv("results/eta_fit.csv", eta_fit, header="eta_fit")

    # 9. Plotting
    plot_lcurve(seminorms, residuals, kappa_vals, knee_idx, save=is_save_plots)
    plot_sele(z, S_mean, S_std, save=is_save_plots)
    lambda_vals = np.arange(eta_ext.size)
    plot_eta(lambda_vals, eta_ext, eta_fit, save=is_save_plots)
    plt.show(block=True)
