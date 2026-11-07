"""Band diagrams and carrier/current profiles for ``DeviceState``s."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.drift_diffusion.solver import DeviceState

# Okabe-Ito
BLUE, VERMILION, SKY, ORANGE, GREEN, PURPLE, GREY = (
    "#0072B2", "#D55E00", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#999999")


def _um(x_cm):
    return np.asarray(x_cm) * 1e4


def _mark_junction(ax: Axes, state: DeviceState) -> None:
    ax.axvline(_um(state.device.junction_cm), color=GREY, ls=":", lw=1)


def plot_band_diagram(state: DeviceState, ax: Optional[Axes] = None,
                      title: Optional[str] = None, show_intrinsic: bool = False) -> Axes:
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    x = _um(state.x_cm)
    ax.plot(x, state.ec_ev, color=BLUE, ls="-", lw=1.8, label=r"$E_c$")
    ax.plot(x, state.ev_ev, color=VERMILION, ls="-", lw=1.8, label=r"$E_v$")
    ax.plot(x, state.efn_ev, color=SKY, ls="--", lw=1.6, label=r"$E_{Fn}$")
    ax.plot(x, state.efp_ev, color=ORANGE, ls="-.", lw=1.6, label=r"$E_{Fp}$")
    if show_intrinsic:
        ax.plot(x, state.ei_ev, color=GREY, ls=":", lw=1, label=r"$E_i$")
    _mark_junction(ax, state)
    ax.set_xlabel(r"Depth $x$ ($\mu$m)")
    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(x[0], x[-1])
    ax.set_title(title or _describe(state))
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_band_diagrams(states: Sequence[DeviceState],
                       titles: Optional[Sequence[str]] = None) -> Figure:
    cols = min(len(states), 2)
    rows = int(np.ceil(len(states) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows),
                             sharey=True, squeeze=False)
    for k, state in enumerate(states):
        plot_band_diagram(state, axes.flat[k], None if titles is None else titles[k])
    for ax in axes.flat[len(states):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_carriers(state: DeviceState, ax: Optional[Axes] = None) -> Axes:
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    x = _um(state.x_cm)
    ax.semilogy(x, state.n, color=BLUE, ls="-", label=r"$n$")
    ax.semilogy(x, state.p, color=VERMILION, ls="--", label=r"$p$")
    _mark_junction(ax, state)
    ax.set_xlabel(r"Depth $x$ ($\mu$m)")
    ax.set_ylabel(r"Density (cm$^{-3}$)")
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=8)
    return ax


def plot_rates(state: DeviceState, ax: Optional[Axes] = None) -> Axes:
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    x = _um(state.x_cm)
    if np.any(state.generation > 0):
        ax.semilogy(x, state.generation, color=GREEN, ls="-", label=r"$G$")
    ax.semilogy(x, np.abs(state.r_srh), color=PURPLE, ls="--", label=r"$|R_{SRH}|$")
    ax.semilogy(x, np.abs(state.r_rad), color=ORANGE, ls="-.", label=r"$|R_{rad}|$")
    _mark_junction(ax, state)
    ax.set_xlabel(r"Depth $x$ ($\mu$m)")
    ax.set_ylabel(r"Rate (cm$^{-3}$ s$^{-1}$)")
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=8)
    return ax


def plot_currents(state: DeviceState, ax: Optional[Axes] = None) -> Axes:
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    x = _um(state.x_edge_cm)
    ax.plot(x, state.jn * 1e3, color=BLUE, ls="-", label=r"$J_n$")
    ax.plot(x, state.jp * 1e3, color=VERMILION, ls="--", label=r"$J_p$")
    ax.plot(x, state.j_total * 1e3, color="black", ls=":", lw=2, label=r"$J = J_n + J_p$")
    _mark_junction(ax, state)
    ax.set_xlabel(r"Depth $x$ ($\mu$m)")
    ax.set_ylabel(r"Current density along $+x$ (mA/cm$^2$)")
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=8)
    return ax


def plot_overview(state: DeviceState) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_band_diagram(state, axes[0, 0])
    plot_carriers(state, axes[0, 1])
    plot_rates(state, axes[1, 0])
    plot_currents(state, axes[1, 1])
    fig.tight_layout()
    return fig


def _describe(state: DeviceState) -> str:
    light = "illuminated" if np.any(state.generation > 0) else "dark"
    return (f"{light}, $V_a$ = {state.applied_voltage:.2f} V, "
            f"$J$ = {state.current_density * 1e3:.3g} mA/cm$^2$")
