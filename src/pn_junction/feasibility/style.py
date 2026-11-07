"""Shared look of the junction feasibility figures (Okabe-Ito colours)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.pn_junction.feasibility import measurement

OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "pn_junction" / "feasibility"
UM = 1e4
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.92", "grid.linewidth": 0.6,
    "font.size": 9, "axes.titlesize": 9, "legend.frameon": False,
})

DEPLETION_LABEL = "depletion region (field > 10% of peak)"
BLIND_LABEL = "blur width ℓ(z) > 2z: data no longer place features"


def shade(ax, z_cm, ell_cm, xmax_um, depletion_cm=None):
    if depletion_cm is not None:
        ax.axvspan(depletion_cm[0] * UM, depletion_cm[1] * UM, color=ORANGE, alpha=0.18, lw=0, zorder=0)
    blind = measurement.blind_from(z_cm, ell_cm) * UM
    if blind < xmax_um:
        ax.axvspan(blind, xmax_um, facecolor="none", edgecolor="0.6", hatch="////", lw=0, zorder=0)


def shading_handles():
    return ([Patch(facecolor=ORANGE, alpha=0.18), Patch(facecolor="none", edgecolor="0.6", hatch="////")],
            [DEPLETION_LABEL, BLIND_LABEL])


def ylim(*curves):
    top = max(np.max(cv) for cv in curves)
    bottom = min(np.min(cv) for cv in curves)
    pad = 0.08 * (top - bottom)
    return bottom - pad, top + pad
