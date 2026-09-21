"""Does the degenerate set differ where the measurement can see, or only deeper?

G is blind past a few micrometres, so parameter sets that agree near the surface and differ
only in the deep tail are expected and harmless. The question this script answers is the
other one: among the parameter vectors that reproduce a measurement equally well, how much do
their SELE profiles disagree *inside* the first few micrometres, where SRV is read off.

It also reports whether that set is straight or curved in normalised parameter space, which
is what decides whether a single Gaussian could describe it at all.

    python -m src.forward_model.standalones.probe_ridge --curve srv_1e5
    python -m src.forward_model.standalones.probe_ridge --test-set
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from src.forward_model.analytic_ele import simulate_ele
from src.forward_model.ele import mesh_and_operator
from src.forward_model.parameters import PARAMETER_NAMES, ParameterSpec
from src.forward_model.sele_simulator import simulate_sele
from src.test_set.loader import load_curve, load_test_set

_DRAWS = 400_000
_RESOLUTION = 500
# A member of the degenerate set is anything the measurement cannot tell apart from the best
# fit available. Relative to the best residual rather than absolute, because the reachable
# residual differs by two orders of magnitude between the SRV and lifetime sweeps.
_SLACK = 2.0
_MIN_MEMBERS = 50
_ZONES = ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 30.0))


def _relative_residuals(params, measured) -> NDArray[np.float64]:
    predicted = simulate_ele(params)
    return np.linalg.norm(predicted - measured, axis=1) / np.linalg.norm(measured)


def _band_width(sele: NDArray[np.float64]) -> NDArray[np.float64]:
    """Depth-wise 68% band width relative to the median."""
    low, median, high = (np.percentile(sele, q, axis=0) for q in (15.865, 50.0, 84.135))
    return np.divide(high - low, np.abs(median),
                     out=np.full_like(median, np.nan), where=median != 0)


def _straightness(normalized: NDArray[np.float64]) -> str:
    """How much of the set a straight line, then a plane, accounts for.

    A Gaussian is an ellipsoid, so it can follow a straight elongated set exactly. If one
    principal direction already explains nearly everything the set is straight and the
    Gaussian head was the right choice; if it takes several, the set is bent and no single
    Gaussian can sit on it without also covering the space it curves around.
    """
    centred = normalized - normalized.mean(axis=0)
    variance = np.linalg.svd(centred, compute_uv=False) ** 2
    share = np.cumsum(variance / variance.sum())
    return "  ".join(f"{k}D {share[k - 1] * 100:5.1f}%" for k in range(1, 4))


def probe(curve_id: str, spec: ParameterSpec, seed: int = 11) -> None:
    curve = load_curve(curve_id)
    measured = np.asarray(curve.ele, dtype=np.float64).ravel()

    rng = np.random.default_rng(seed)
    params = spec.sample(_DRAWS, rng)
    residuals = _relative_residuals(params, measured)

    best = residuals.min()
    members = residuals <= best * _SLACK
    if members.sum() < _MIN_MEMBERS:
        members = residuals <= np.partition(residuals, _MIN_MEMBERS)[_MIN_MEMBERS]

    selected = params[members]
    print(f"\n{curve_id}")
    print(f"  best residual {best * 100:.3f}%, "
          f"{int(members.sum())} draws within {_SLACK:g}x of it "
          f"(worst of them {residuals[members].max() * 100:.3f}%)")

    normalized = spec.to_normalized(selected)
    print(f"  parameter spread (normalised, min..max of a [-1,1] box):")
    for i, name in enumerate(PARAMETER_NAMES):
        column = normalized[:, i]
        print(f"    {name:<12} {column.min():+.2f} .. {column.max():+.2f}"
              f"   width {np.ptp(column):.2f}")
    print(f"  variance explained by   {_straightness(normalized)}")

    z_cm, _ = mesh_and_operator(_RESOLUTION)
    sele = simulate_sele(selected, z_cm)
    finite = np.isfinite(sele).all(axis=1)
    sele = sele[finite]

    relative = _band_width(sele)
    z_um = z_cm / 1e-4
    print("  68% SELE band width relative to the median, by depth:")
    for lo, hi in _ZONES:
        zone = (z_um >= lo) & (z_um < hi)
        print(f"    {lo:5.1f}-{hi:4.1f} um   {np.nanmedian(relative[zone]):.3f}")
    print(f"  at the shallowest mesh point ({z_um[0]:.3f} um)   {relative[0]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=str, default=None)
    parser.add_argument("--test-set", action="store_true")
    args = parser.parse_args()

    spec = ParameterSpec()
    ids = ([c.curve_id for c in load_test_set(include_reference=False)]
           if args.test_set else [args.curve])
    if not ids or ids == [None]:
        parser.error("pick one of --curve or --test-set")

    for curve_id in ids:
        probe(curve_id, spec)


if __name__ == "__main__":
    main()
