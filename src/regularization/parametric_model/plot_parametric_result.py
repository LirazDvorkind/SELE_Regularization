"""Figures for a parametric reconstruction: the SELE band, and the ELE consistency check.

The SELE band is the envelope of the draws that best reproduce the measurement, not
percentiles of every draw. Ranking by data fit is what makes it an uncertainty set rather
than a picture of the network's own spread; ``uncertainty.py`` explains the levels and
``standalones/check_uncertainty.py`` is where their coverage was measured.

The second panel is not decoration. A SELE profile is only an answer if it reproduces the
measurement through ``ELE = G @ SELE``, and that plot is where a posterior that drifted off
the data becomes visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.regularization.parametric_model import uncertainty as unc
from src.regularization.parametric_model.inference import SeleEnsemble

# Okabe-Ito, per this repo's plotting conventions.
_MEDIAN = "#0072B2"
_TRUTH = "#D55E00"
_MEASURED = "#000000"
_UM_PER_CM = 1e4
_PERCENT = 100.0


def plot_sele_band(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
        ground_truth_z_cm: Optional[NDArray[np.float64]] = None,
        ground_truth_sele: Optional[NDArray[np.float64]] = None,
        title: str = "",
        ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.6))

    z_um = ensemble.z_cm * _UM_PER_CM
    wide = unc.misfit_band(ensemble, measured_ele, max(unc.DEFAULT_LEVELS))
    tight = unc.misfit_band(ensemble, measured_ele, unc.REPORTED_LEVEL)

    ax.fill_between(z_um, wide.low * _PERCENT, wide.high * _PERCENT, color=_MEDIAN,
                    alpha=0.18, linewidth=0,
                    label=f"best-fitting {max(unc.DEFAULT_LEVELS):.0%} of draws")
    ax.fill_between(z_um, tight.low * _PERCENT, tight.high * _PERCENT, color=_MEDIAN,
                    alpha=0.35, linewidth=0,
                    label=f"best-fitting {unc.REPORTED_LEVEL:.0%} of draws")
    ax.plot(z_um, tight.median * _PERCENT, color=_MEDIAN, linewidth=2.0,
            label="median of those")

    if ground_truth_sele is not None:
        truth_z = (ground_truth_z_cm if ground_truth_z_cm is not None
                   else ensemble.z_cm) * _UM_PER_CM
        ax.plot(truth_z, np.asarray(ground_truth_sele) * _PERCENT, color=_TRUTH,
                linewidth=1.6, linestyle="--", label="ground truth")

    ax.set_xlabel("depth z [um]")
    ax.set_ylabel("SELE [%]")
    ax.set_xlim(0, z_um.max())
    ax.set_title(title or "SELE reconstruction")
    ax.legend(frameon=False, fontsize=8.5)
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

    # The same draws the SELE panel shows, so the two are one statement rather than two.
    wide = unc.select_by_misfit(ensemble, measured_ele, max(unc.DEFAULT_LEVELS))
    tight = unc.select_by_misfit(ensemble, measured_ele, unc.REPORTED_LEVEL)

    ax.fill_between(wavelength_nm, wide.ele.min(axis=0) * _PERCENT,
                    wide.ele.max(axis=0) * _PERCENT, color=_MEDIAN, alpha=0.18, linewidth=0,
                    label=f"best-fitting {max(unc.DEFAULT_LEVELS):.0%} of draws")
    ax.fill_between(wavelength_nm, tight.ele.min(axis=0) * _PERCENT,
                    tight.ele.max(axis=0) * _PERCENT, color=_MEDIAN, alpha=0.35, linewidth=0,
                    label=f"best-fitting {unc.REPORTED_LEVEL:.0%} of draws")
    ax.plot(wavelength_nm, np.median(tight.ele, axis=0) * _PERCENT, color=_MEDIAN,
            linewidth=2.0, label="median of those")
    ax.plot(wavelength_nm, np.asarray(measured_ele).ravel() * _PERCENT, color=_MEASURED,
            linestyle="none", marker="o", markersize=4, label="measured ELE")

    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("ELE [%]")
    ax.set_title(title or "Measurement refit through G")
    ax.legend(frameon=False, fontsize=8.5)
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
    plot_sele_band(ensemble, measured_ele, ground_truth_z_cm, ground_truth_sele,
                   title=f"SELE reconstruction -- {stem}", ax=axes[0])
    plot_ele_consistency(ensemble, wavelength_nm, measured_ele,
                         title=f"ELE refit -- {stem}", ax=axes[1])
    figure.tight_layout()

    path = output_dir / f"{stem}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  saved {path}")
