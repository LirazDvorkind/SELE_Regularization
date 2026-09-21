"""Figures for a parametric reconstruction: the SELE band, and the ELE consistency check.

The second figure is not decoration. A SELE profile is only an answer if it reproduces the
measurement through ``ELE = G @ SELE``, and that plot is where a posterior that drifted off
the data becomes visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.regularization.parametric_model.inference import SeleEnsemble

# Okabe-Ito, per this repo's plotting conventions.
_MEDIAN = "#0072B2"
_TRUTH = "#D55E00"
_MEASURED = "#000000"
_UM_PER_CM = 1e4


def plot_sele_band(
        ensemble: SeleEnsemble,
        ground_truth_z_cm: Optional[NDArray[np.float64]] = None,
        ground_truth_sele: Optional[NDArray[np.float64]] = None,
        title: str = "",
        ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.6))

    z_um = ensemble.z_cm * _UM_PER_CM
    low95, low68, median, high68, high95 = (
        ensemble.percentile(q) for q in (2.275, 15.865, 50.0, 84.135, 97.725))

    ax.fill_between(z_um, low95, high95, color=_MEDIAN, alpha=0.18,
                    linewidth=0, label="95% band")
    ax.fill_between(z_um, low68, high68, color=_MEDIAN, alpha=0.35,
                    linewidth=0, label="68% band")
    ax.plot(z_um, median, color=_MEDIAN, linewidth=2.0, label="posterior median")

    if ground_truth_sele is not None:
        truth_z = (ground_truth_z_cm if ground_truth_z_cm is not None
                   else ensemble.z_cm) * _UM_PER_CM
        ax.plot(truth_z, ground_truth_sele, color=_TRUTH, linewidth=1.6,
                linestyle="--", label="ground truth")

    ax.set_xlabel("depth z [um]")
    ax.set_ylabel("SELE [fraction]")
    ax.set_xlim(0, z_um.max())
    ax.set_title(title or "SELE posterior")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return ax


def plot_ele_consistency(
        ensemble: SeleEnsemble,
        wavelength_nm: NDArray[np.float64],
        measured_ele: NDArray[np.float64],
        title: str = "",
        ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.6))

    low95, low68, median, high68, high95 = (
        ensemble.ele_percentile(q) for q in (2.275, 15.865, 50.0, 84.135, 97.725))

    ax.fill_between(wavelength_nm, low95, high95, color=_MEDIAN, alpha=0.18,
                    linewidth=0, label="95% band")
    ax.fill_between(wavelength_nm, low68, high68, color=_MEDIAN, alpha=0.35,
                    linewidth=0, label="68% band")
    ax.plot(wavelength_nm, median, color=_MEDIAN, linewidth=2.0, label="refit median")
    ax.plot(wavelength_nm, np.asarray(measured_ele).ravel(), color=_MEASURED,
            linestyle="none", marker="o", markersize=4, label="measured ELE")

    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("ELE")
    ax.set_title(title or "Measurement refit through G")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return ax


def save_figures(
        ensemble: SeleEnsemble,
        wavelength_nm: NDArray[np.float64],
        measured_ele: NDArray[np.float64],
        output_dir: Path,
        stem: str,
        ground_truth_z_cm: Optional[NDArray[np.float64]] = None,
        ground_truth_sele: Optional[NDArray[np.float64]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
    plot_sele_band(ensemble, ground_truth_z_cm, ground_truth_sele,
                   title=f"SELE posterior -- {stem}", ax=axes[0])
    plot_ele_consistency(ensemble, wavelength_nm, measured_ele,
                         title=f"ELE refit -- {stem}", ax=axes[1])
    figure.tight_layout()

    path = output_dir / f"{stem}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  saved {path}")
