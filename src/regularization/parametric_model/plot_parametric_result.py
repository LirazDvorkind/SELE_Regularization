"""Figures for a parametric reconstruction: the SELE profile, and the ELE consistency check.

The bold curve is the best-fitting draw, a real simulator curve with parameters behind it.
The light lines are every other draw within ``uncertainty.TOLERANCE`` of the measurement, with
their envelope shaded: where they fan out, the measurement no longer pins the profile down.

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
_BEST = "#0072B2"
_KEPT = "#56B4E9"
_TRUTH = "#D55E00"
_MEASURED = "#000000"
_UM_PER_CM = 1e4
_PERCENT = 100.0


def _draw_selection(ax, x, best, kept, selection: unc.Selection) -> None:
    """Kept draws as thin lines under their envelope, the best draw bold on top."""
    if selection.met:
        label = f"{selection.n_kept} draws within {selection.tolerance:.0%} in ELE"
        ax.fill_between(x, kept.min(axis=0) * _PERCENT, kept.max(axis=0) * _PERCENT,
                        color=_KEPT, alpha=0.15, linewidth=0)
        # Line opacity falls with the count so a few hundred lines still read as a spread.
        alpha = float(np.clip(8.0 / kept.shape[0], 0.03, 0.4))
        for curve in kept:
            ax.plot(x, curve * _PERCENT, color=_KEPT, linewidth=0.6, alpha=alpha)
        # A proxy, so the legend swatch is not as faint as the individual lines.
        ax.plot([], [], color=_KEPT, linewidth=2.0, label=label)
        best_label = f"best fit ({selection.best_misfit:.2%} ELE misfit)"
    else:
        best_label = (f"best fit -- misses ELE by {selection.best_misfit:.2%}, "
                      f"none within {selection.tolerance:.0%}")
    ax.plot(x, best * _PERCENT, color=_BEST, linewidth=2.0, label=best_label)


def plot_sele_selection(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
        ground_truth_z_cm: Optional[NDArray[np.float64]] = None,
        ground_truth_sele: Optional[NDArray[np.float64]] = None,
        title: str = "",
        ax: Optional[plt.Axes] = None,
        selection: Optional[unc.Selection] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.6))

    selection = selection or unc.select(ensemble, measured_ele)
    z_um = ensemble.z_cm * _UM_PER_CM
    _draw_selection(ax, z_um, selection.best_sele, selection.kept.sele, selection)

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
        selection: Optional[unc.Selection] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.6))

    # The same draws the SELE panel shows, so the two are one statement rather than two.
    selection = selection or unc.select(ensemble, measured_ele)
    _draw_selection(ax, wavelength_nm, selection.best_ele, selection.kept.ele, selection)
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
        selection: Optional[unc.Selection] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = selection or unc.select(ensemble, measured_ele)

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
    plot_sele_selection(ensemble, measured_ele, ground_truth_z_cm, ground_truth_sele,
                        title=f"SELE reconstruction -- {stem}", ax=axes[0], selection=selection)
    plot_ele_consistency(ensemble, wavelength_nm, measured_ele,
                         title=f"ELE refit -- {stem}", ax=axes[1], selection=selection)
    figure.tight_layout()

    path = output_dir / f"{stem}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  saved {path}")
