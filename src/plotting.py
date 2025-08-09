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
    ax.legend()
    if save:
        _ensure_results_dir()
        fig.savefig('results/lcurve.png', dpi=300)
    plt.title("Regularization Loss Curve")
    plt.show(block=False)

def plot_sele(z: Sequence[float], S_mean: Sequence[float], S_std: Sequence[float],
              *, save: bool = False):
    fig, ax = plt.subplots()
    ax.plot(z, S_mean, label='SELE')
    ax.fill_between(z, np.asarray(S_mean) - S_std, np.asarray(S_mean) + S_std,
                    alpha=0.3, label=r'$\pm 1\,\sigma$')
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('SELE')
    ax.legend()
    if save:
        _ensure_results_dir()
        fig.savefig('results/sele_profile.png', dpi=300)
    plt.title("SELE Plot")
    plt.show(block=False)

def plot_eta(lambda_vals, eta_meas, eta_fit, *, save: bool = False):
    fig, ax = plt.subplots()
    ax.plot(lambda_vals, eta_meas, label='Measured')
    ax.plot(lambda_vals, eta_fit, '--', label='Reconstructed')
    ax.set_xlabel('Wavelength (arb. index)')
    ax.set_ylabel(r'$\eta_{ext}$')
    ax.legend()
    if save:
        _ensure_results_dir()
        fig.savefig('results/eta_fit.png', dpi=300)
    plt.title("Reconstructed ELE")
    plt.show(block=False)
