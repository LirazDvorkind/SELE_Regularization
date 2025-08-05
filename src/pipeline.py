"""High‑level orchestration of the SELE extraction workflow."""
from __future__ import annotations
import os

import matplotlib.pyplot as plt
import numpy as np

from src.io import load_eta, load_G, load_z, save_csv
from src.operators import build_L
from src.tikhonov import sweep_kappa, find_knee
from src.plotting import plot_lcurve, plot_sele, plot_eta

def run_regularization(
    *,
    eta_path: str,
    G_path: str,
    z_path: str,
    L_flag: str = 'L0',
    kappa_max: float = 1e-2,
    kappa_min: float = 1e-15,
    n_kappa: int = 200,
    conf_fact: float = 10.0,
    is_save_plots: bool = True,
    e_charge: float = 1.60217657e-19,
    photon_flux: float = 1e14,
):
    # 1. Load data
    eta_ext = load_eta(eta_path)
    G = load_G(G_path)
    z = load_z(z_path)
    # Apply common photon-flux scaling so A and B share units
    G *= photon_flux * e_charge  # G is already scaled by delta_z
    B = eta_ext * photon_flux * e_charge

    if G.shape[0] != eta_ext.shape[0]:
        raise ValueError('Dimension mismatch: G rows must match length of eta_ext.')

    # 2. Build regularization operator
    L = build_L(L_flag, G.shape[1])

    # 3. κ sweep
    kappa_vals = np.logspace(np.log10(kappa_max), np.log10(kappa_min), n_kappa)
    residuals, seminorms, S_list = sweep_kappa(G, B, L, kappa_vals)

    # 4. Knee detection
    kappa_knee, knee_idx = find_knee(residuals, seminorms, kappa_vals)
    print(f'κ_knee = {kappa_knee:.3e}')

    # 5. Confidence window
    conf_mask = (kappa_vals >= kappa_knee / conf_fact) & (kappa_vals <= kappa_knee * conf_fact)
    S_stack = np.stack([S_list[i] for i, m in enumerate(conf_mask) if m], axis=1)
    S_mean = S_stack.mean(axis=1)
    S_std = S_stack.std(axis=1)

    # 6. Reconstruction using κ_knee
    S_knee = S_list[knee_idx]
    eta_fit = G @ S_knee / (photon_flux * e_charge)

    # 7. Save numeric outputs
    os.makedirs('results', exist_ok=True)
    save_csv('results/S_mean.csv', S_mean, header='S_mean')
    save_csv('results/S_std.csv', S_std, header='S_std')
    save_csv('results/eta_fit.csv', eta_fit, header='eta_fit')

    # 8. Plotting
    plot_lcurve(seminorms, residuals, kappa_vals, knee_idx, save=is_save_plots)
    plot_sele(z, S_mean, S_std, save=is_save_plots)
    lambda_vals = np.arange(len(eta_ext))
    plot_eta(lambda_vals, eta_ext, eta_fit, save=is_save_plots)
    plt.show(block=True)
