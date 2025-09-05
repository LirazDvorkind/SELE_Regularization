"""High‑level orchestration of the SELE extraction workflow (non‑uniform mesh version)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.io import load_eta, load_z, save_csv
from src.operators import build_L
from src.tikhonov import sweep_kappa, find_knee
from src.plotting import plot_lcurve, plot_sele, plot_eta
from src.mesh import non_uniform_mesh
from src.types.enums import LFlag, RegularizationMethod
from src.types.config import Config


def run_regularization(config: Config):
    """Full regularisation pipeline supporting arbitrary 1‑D meshes."""

    # 0. Load configuration values:
    eta_path: str = config.data_paths.eta_ext
    z_path: str = config.data_paths.z
    k_path: str = config.data_paths.k
    lambda_for_alpha_path: str = config.data_paths.lambda_for_alpha
    wavelengths_path: str = config.data_paths.wavelengths
    z_gt_path: str = config.data_paths.z_gt
    sele_gt_path: str = config.data_paths.sele_gt
    L_score_network_path: str = config.data_paths.L_score_network
    L_flag: LFlag = config.L_flag
    regularization_method: RegularizationMethod = config.regularization_method
    kappa_max, kappa_min = config.kappa_range
    n_kappa: int = config.n_kappa
    conf_fact: float = config.conf_window
    is_save_plots: bool = config.is_save_plots
    e_charge: float = config.e_charge
    photon_flux: float = config.photon_flux
    z_turn: float = config.z_turn
    lin_mesh_size: float = config.lin_mesh_size
    exp_base: float = config.exp_base
    W: float = config.W

    # 1. Load data ---------------------------------------------------------
    z_gt = load_eta(z_gt_path)
    sele_gt = load_eta(sele_gt_path)
    eta_ext = load_eta(eta_path)
    z = load_z(z_path).ravel()
    # Load optical inputs for recomputing G on the new mesh
    k = load_z(k_path).ravel()  # extinction coefficient k(λ) [unitless]
    lambda_for_alpha = load_z(lambda_for_alpha_path).ravel()  # wavelengths for alpha [nm]
    wavelengths = load_z(wavelengths_path).ravel()  # wavelengths of G [nm]

    # Recompute G on a non-uniform mesh directly from Beer–Lambert optics
    z_new, G_new = non_uniform_mesh(
        z_min=float(z[0]),
        z_max=W,
        wavelengths=wavelengths,
        k=k,
        lambda_for_alpha=lambda_for_alpha,
        z_turn=z_turn,
        lin_mesh_size=lin_mesh_size,
        exp_base=exp_base
    )
    # Persist the newly created values, including mesh element sizes
    save_csv("results/raw/z_new.csv", z_new)
    save_csv("results/raw/dz_new.csv", np.diff(z_new))
    save_csv("results/raw/G_new.csv", G_new)
    # Use the recomputed quantities from here onward
    G, z = G_new, z_new

    # 2. Unit normalisation (A and B must have same units)
    G = G * photon_flux * e_charge
    B = eta_ext * photon_flux * e_charge

    if G.shape[0] != eta_ext.size:
        raise ValueError(f"Row mismatch between G and η_ext: G[0] is {G.shape[0]} but n_ext is {eta_ext.size}")

    # 3. Regularisation operator
    L = build_L(L_flag, len(z) - 1)

    # 4. κ‑sweep
    kappa_vals = np.logspace(np.log10(kappa_max), np.log10(kappa_min), n_kappa)
    residuals, seminorms, S_list = sweep_kappa(G, B, L, kappa_vals)

    # 5. Knee detection
    kappa_knee, knee_idx = find_knee(residuals, seminorms, kappa_vals)
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
    z_centres = 0.5 * (z[:-1] + z[1:])  # length M-1
    save_csv("results/raw/S_mean.csv", np.column_stack([z_centres, S_mean]), header="z_cm,S_mean")
    save_csv("results/raw/S_std.csv", np.column_stack([z_centres, S_std]), header="z_cm,S_std")
    save_csv("results/raw/eta_fit.csv", eta_fit, header="eta_fit")

    # 9. Plotting
    plot_lcurve(seminorms, residuals, kappa_vals, knee_idx, mask, save=is_save_plots)
    plot_sele(z_centres, S_mean, S_std, sele_gt, z_gt, save=is_save_plots)
    plot_eta(wavelengths, eta_ext, eta_fit, save=is_save_plots)
    plt.show(block=True)
