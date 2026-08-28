"""
Overview figure for the ground-truth test set -- a visual check that the extracted profiles
behave the way the physics says they should before they are used to score reconstructions.

Run:  python -m src.test_set.plot_test_set
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.test_set.loader import load_test_set

_OUT_PATH = Path("results") / "test_set_overview.png"
_CM_TO_UM = 1e4


def plot_test_set(save: bool = True) -> None:
    curves = load_test_set()
    srv = [c for c in curves if c.param_name == "SRV"]
    tau = [c for c in curves if c.param_name == "tau_SRH"]
    reference = [c for c in curves if c.param_name == "reference"]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    _plot_sweep(axes[0], srv, "SRV [cm/s]", log_param=True)
    _plot_sweep(axes[1], tau, r"$\tau_{SRH}$ [ns]", log_param=True)

    for axis in axes:
        for curve in reference:
            axis.plot(curve.z_cm * _CM_TO_UM, curve.sele, color="black",
                      linestyle="--", linewidth=1.5, label="existing GT", zorder=5)
        axis.set_xlabel(r"Depth [$\mu$m]")
        axis.set_ylabel("SELE [fraction]")
        axis.set_xscale("log")
        axis.set_xlim(5e-3, 3.5e2)
        axis.legend(fontsize=8, loc="upper left")

    axes[0].set_title("Surface recombination velocity sweep")
    axes[1].set_title("SRH lifetime sweep")

    if save:
        _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(_OUT_PATH, dpi=300)
        print(f"Saved {_OUT_PATH}")
    else:
        plt.show()


def _plot_sweep(axis, curves, colorbar_label: str, log_param: bool) -> None:
    """Draw one parameter sweep, colouring by the swept value on a perceptual colormap."""
    values = np.array([c.param_value for c in curves], dtype=float)
    scaled = np.log10(values) if log_param else values
    normalized = (scaled - scaled.min()) / np.ptp(scaled)
    colormap = plt.get_cmap("viridis")

    # Dashes vary alongside hue so the sweep stays readable without colour.
    dash_styles = ["-", "--", "-.", ":"]
    for index, (curve, shade) in enumerate(zip(curves, normalized)):
        axis.plot(curve.z_cm * _CM_TO_UM, curve.sele,
                  color=colormap(shade), linestyle=dash_styles[index % len(dash_styles)],
                  linewidth=1.6, label=curve.label)

    mappable = plt.cm.ScalarMappable(cmap=colormap,
                                     norm=plt.Normalize(scaled.min(), scaled.max()))
    colorbar = axis.figure.colorbar(mappable, ax=axis)
    colorbar.set_label(f"log10({colorbar_label})" if log_param else colorbar_label)


if __name__ == "__main__":
    plot_test_set()
