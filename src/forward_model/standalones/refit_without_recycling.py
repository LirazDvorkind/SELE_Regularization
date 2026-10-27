"""Fit the repo's recycling-free simulator to the ELE of the full-physics SELE (V3), and compare SELEs.

    python -m src.forward_model.standalones.refit_without_recycling

Tests how far ignoring recycling and back reflection moves the retrieved SELE once the five
simulator parameters are free to compensate. Targets are V3 profiles from
``estimate_pr_reflection`` (noise-free, same G for target and candidate, so any gap is model
mismatch). Method and assumptions: ``docs/pr_reflection_estimate.md``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from src.forward_model.parameters import PARAMETER_SPEC
from src.forward_model.sele_simulator import B0, simulate_sele
from src.forward_model.standalones.estimate_pr_reflection import (
    S51, Optics, ele_from, run, wafer_edges)

_WINDOWS = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 10), (10, 30)]
_TARGETS = [("p0=1e18, tau=60 ns", 1e18, 60e-9), ("p0=1e18, tau=0.4 ns", 1e18, 0.4e-9),
            ("p0=1e19 (S5.1), tau=0.4 ns", 1e19, 0.4e-9)]


def fit(target_ele, centres, edges, starts=40, seed=0):
    rng = np.random.default_rng(seed)
    best = None

    def residual(u):
        sele = simulate_sele(PARAMETER_SPEC.from_normalized(np.clip(u, -1, 1))[None, :], centres)[0]
        return ele_from(edges, sele) / target_ele - 1.0

    for u0 in rng.uniform(-1, 1, (starts, 5)):
        result = least_squares(residual, u0, bounds=(-1, 1))
        if best is None or result.cost < best.cost:
            best = result
    return PARAMETER_SPEC.from_normalized(best.x), np.sqrt(2 * best.cost / target_ele.size)


def window_ratio(z_um, a, b, lo, hi):
    m = (z_um >= lo) & (z_um < hi)
    return np.trapezoid(a[m], z_um[m]) / np.trapezoid(b[m], z_um[m])


def main() -> None:
    edges = wafer_edges()
    centres = 0.5 * (edges[:-1] + edges[1:])
    z_um = centres * 1e4
    for name, p0, tau in _TARGETS:
        optics = Optics(edges, p0, 1.0, 8)
        kernels = {"esc_simple": optics.escape_front("simple"), "esc_direct": optics.escape_front("direct"),
                   "esc_full": optics.escape_front("full"),
                   "K_direct": optics.recycling_matrix(False), "K_full": optics.recycling_matrix(True)}
        for label, scale in (("spectrum as emitted", 1.0), ("scaled to B0", B0 / optics.b_spectral)):
            profiles, _, gain = run({**S51, "p0": p0, "tau": tau}, optics, edges, kernels, scale)
            truth = profiles["V3"]
            params, rms = fit(ele_from(edges, truth), centres, edges)
            refit = simulate_sele(params[None, :], centres)[0]
            print(f"\n{name}, {label}: loop gain {gain:.2f}; ELE rms misfit of the refit {100*rms:.2f}%")
            print("  fitted:", PARAMETER_SPEC.describe(params))
            print("  refit/V3 SELE integral, windows [um]: "
                  + "  ".join(f"{lo}-{hi}: {window_ratio(z_um, refit, truth, lo, hi):.2f}" for lo, hi in _WINDOWS))
            print(f"  peak: refit {100*refit.max():.3f}% at {z_um[refit.argmax()]:.1f} um, "
                  f"V3 {100*truth.max():.3f}% at {z_um[truth.argmax()]:.1f} um", flush=True)


if __name__ == "__main__":
    main()
