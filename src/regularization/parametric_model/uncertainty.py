"""Uncertainty on an unknown SELE, ranked by how well each draw explains the measurement.

``inference.py`` reports percentiles of the posterior as drawn: every sample counts the same,
whether it reproduces the measurement or not. That is the network's own spread, and taken at
the 68% level it does not hold the truth -- on the lifetime curves of the test set only
12-15% of depths fall inside it.

Here the ordering is the data. Every draw is pushed back through the forward model, scored by
its mean squared error against the measured ELE, and the band is the envelope of the
best-fitting fraction. The gain is calibration rather than width: at the same nominal level
the two bands are nearly the same size, but the misfit band's coverage was measured, and the
level that holds the truth everywhere (0.68) is about 1.7x tighter than the percentile band
that also holds it everywhere (95%).

It does not reach the ambiguity the physics allows -- the parameters that genuinely fit span
roughly 0.007 near the surface where this reports a few tenths. That gap is not closed by
cutting harder: below 0.68 the truth starts falling out, because a better ELE fit is not
reliably a closer profile. What remains is an honest upper bound, not a tight estimate.

Nothing here needs the truth, so it all applies to a real measurement. What it cannot do is
prove the truth is inside; ``standalones/check_uncertainty.py`` measures that against the
ground-truth test set, which is what licenses believing it elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from src.regularization.parametric_model.inference import SeleEnsemble

# 0.68 is the tightest level that still holds the truth at every depth of the first 3 um on
# every curve of the ground-truth test set; at 0.25 coverage falls to a median of 72% and
# keeps dropping. Over the full 0-30 um even 0.68 lets the truth out on the longest-lifetime
# curves, which is the deep tail no measurement constrains rather than a bad level. Tighter
# does not work because a better ELE fit is not reliably a closer profile -- that is the
# degeneracy, not a tuning failure. 0.95 is kept alongside for comparison with the plain
# percentile band it replaces. See standalones/check_uncertainty.py.
DEFAULT_LEVELS: Tuple[float, ...] = (0.68, 0.95)
REPORTED_LEVEL = 0.68


@dataclass(frozen=True)
class MisfitBand:
    """Envelope of the best-fitting fraction of draws, with the cut that produced it."""

    level: float                      # fraction of draws kept
    z_cm: NDArray[np.float64]
    low: NDArray[np.float64]          # per-depth minimum over the kept draws
    high: NDArray[np.float64]         # per-depth maximum
    median: NDArray[np.float64]
    n_kept: int
    mse_cut: float                    # worst MSE admitted
    relative_cut: float               # the same cut as a relative L2, for reading


def ele_mse(ensemble: SeleEnsemble, measured_ele: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean squared error in ELE space for every draw, shape ``(n_samples,)``."""
    measured = np.asarray(measured_ele, dtype=np.float64).ravel()
    return np.mean((ensemble.ele - measured) ** 2, axis=1)


def ele_relative(ensemble: SeleEnsemble, measured_ele: NDArray[np.float64]) -> NDArray[np.float64]:
    """The same misfit as a relative L2 norm. Same ordering as ``ele_mse``, readable units."""
    measured = np.asarray(measured_ele, dtype=np.float64).ravel()
    return np.linalg.norm(ensemble.ele - measured, axis=1) / np.linalg.norm(measured)


def misfit_quantiles(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
        quantiles: Sequence[float] = (0, 5, 25, 50, 75, 95, 100),
) -> Dict[float, Tuple[float, float]]:
    """Distribution of the misfit itself, as ``{quantile: (mse, relative)}``.

    Read before choosing a level. If even the best draw misfits badly the posterior does not
    contain an answer and no choice of cut rescues it -- the band would be an envelope of
    wrong curves.
    """
    mse = ele_mse(ensemble, measured_ele)
    relative = ele_relative(ensemble, measured_ele)
    return {q: (float(np.percentile(mse, q)), float(np.percentile(relative, q)))
            for q in quantiles}


def select_by_misfit(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
        level: float,
) -> SeleEnsemble:
    """The ``level`` fraction of draws with the smallest ELE misfit, as an ensemble."""
    if not 0.0 < level <= 1.0:
        raise ValueError(f"level must be in (0, 1], got {level}")

    mse = ele_mse(ensemble, measured_ele)
    keep = max(1, int(round(level * mse.size)))
    order = np.argsort(mse)[:keep]

    return replace(ensemble, sele=ensemble.sele[order], ele=ensemble.ele[order],
                   params=ensemble.params[order])


def misfit_band(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
        level: float,
) -> MisfitBand:
    """Envelope of the draws that fit the measurement best.

    The band is the per-depth minimum and maximum over the kept set rather than percentiles
    of it: the set has already been cut by how well it explains the data, so every member is
    admissible and the honest report is the full extent of what survived.
    """
    kept = select_by_misfit(ensemble, measured_ele, level)
    return MisfitBand(
        level=level,
        z_cm=ensemble.z_cm,
        low=kept.sele.min(axis=0),
        high=kept.sele.max(axis=0),
        median=np.median(kept.sele, axis=0),
        n_kept=kept.sele.shape[0],
        mse_cut=float(ele_mse(kept, measured_ele).max()),
        relative_cut=float(ele_relative(kept, measured_ele).max()),
    )


def scalar_intervals(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
        level: float = REPORTED_LEVEL,
) -> Dict[str, Tuple[float, float, float]]:
    """``{name: (low, median, high)}`` for the quantities actually worth quoting.

    Propagating to the scalar beats reading it off a band. A band's edges are the extremes
    at each depth separately, so it loses the correlation between depths -- it cannot express
    that a draw with a high peak also tends to put that peak deeper -- and peak depth cannot
    be read off a band at all.
    """
    kept = select_by_misfit(ensemble, measured_ele, level)
    z_um = kept.z_cm / 1e-4

    quantities = {
        "surface SELE": kept.sele[:, 0],
        "peak SELE": kept.sele.max(axis=1),
        "peak depth [um]": z_um[kept.sele.argmax(axis=1)],
        "mean SELE 0-3um": kept.sele[:, z_um < 3.0].mean(axis=1),
    }
    return {name: (float(v.min()), float(np.median(v)), float(v.max()))
            for name, v in quantities.items()}


def report(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
        levels: Sequence[float] = DEFAULT_LEVELS,
) -> str:
    lines = ["  ELE misfit across the draws (MSE, and the same as relative L2):"]
    for q, (mse, rel) in misfit_quantiles(ensemble, measured_ele).items():
        lines.append(f"    {q:3.0f}th   {mse:.3e}   {rel * 100:7.3f}%")

    z_um = ensemble.z_cm / 1e-4
    lines.append("  band width relative to the median, by depth zone:")
    header = "    level   kept   worst fit" + "".join(
        f"{f'{lo:g}-{hi:g}um':>11}" for lo, hi in ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0)))
    lines.append(header)

    for level in levels:
        band = misfit_band(ensemble, measured_ele, level)
        relative = np.divide(band.high - band.low, np.abs(band.median),
                             out=np.full_like(band.median, np.nan), where=band.median != 0)
        row = "".join(f"{np.nanmedian(relative[(z_um >= lo) & (z_um < hi)]):10.3f} "
                      for lo, hi in ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0)))
        lines.append(f"    {level:5.2f} {band.n_kept:6d} {band.relative_cut * 100:9.3f}%  {row}")

    lines.append(f"  quantities worth quoting (best-fitting {REPORTED_LEVEL:.0%} of draws):")
    for name, (low, median, high) in scalar_intervals(ensemble, measured_ele).items():
        lines.append(f"    {name:<18} {median:11.4g}   [{low:.4g}, {high:.4g}]")
    return "\n".join(lines)


__all__ = [
    "MisfitBand", "DEFAULT_LEVELS", "ele_mse", "ele_relative", "misfit_quantiles",
    "select_by_misfit", "misfit_band", "scalar_intervals", "report",
]
