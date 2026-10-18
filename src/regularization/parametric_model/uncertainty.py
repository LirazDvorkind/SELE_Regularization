"""Which draws answer the measurement, and which one to report.

``inference.py`` reports percentiles of the posterior as drawn: every sample counts the same,
whether it reproduces the measurement or not. That is the network's own spread, not an answer.

Here the data decides. Every draw is pushed back through the forward model and scored by its
relative L2 misfit against the measured ELE. Two things come out:

- **the reported curve** is the single best-fitting draw. It is a real simulator curve with a
  parameter set behind it, which a depth-wise median of many draws is not -- that stitches
  different draws together at different depths and corresponds to no parameters at all;
- **the close-enough set** is every draw within an absolute ``TOLERANCE`` of the measurement.
  They all explain the data, so where they spread apart in depth the measurement has stopped
  constraining the profile. Their envelope is drawn alongside them.

The cut is an absolute misfit rather than a fraction of draws. A fraction says nothing about
fit: the previous best-fitting-68% rule admitted draws missing the measurement by 13-23%, and
its band held the truth largely because of them. A tolerance means the same thing on every
measurement. If no draw meets it, the best draw is reported alone with a warning: the
posterior then holds nothing that explains the data, and an envelope of misfitting curves
would only dress that up.

Nothing here needs the truth, so it all applies to a real measurement. How often the
close-enough envelope contains the truth is measured on the ground-truth test set by
``standalones/check_uncertainty.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from src.regularization.parametric_model.inference import SeleEnsemble

# Relative L2 ELE misfit a draw must meet to count as explaining the measurement.
TOLERANCE = 0.01


@dataclass(frozen=True)
class Selection:
    """The best-fitting draw and the draws within ``tolerance`` of the measurement."""

    tolerance: float
    z_cm: NDArray[np.float64]
    best_sele: NDArray[np.float64]
    best_ele: NDArray[np.float64]
    best_params: NDArray[np.float64]  # physical units
    best_misfit: float                # relative L2
    kept: SeleEnsemble                # the close-enough draws, best first; just the best if none
    met: bool                         # whether any draw met the tolerance

    @property
    def n_kept(self) -> int:
        return self.kept.sele.shape[0]

    @property
    def low(self) -> NDArray[np.float64]:
        return self.kept.sele.min(axis=0)

    @property
    def high(self) -> NDArray[np.float64]:
        return self.kept.sele.max(axis=0)


def ele_mse(ensemble: SeleEnsemble, measured_ele: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean squared error in ELE space for every draw, shape ``(n_samples,)``."""
    measured = np.asarray(measured_ele, dtype=np.float64).ravel()
    return np.mean((ensemble.ele - measured) ** 2, axis=1)


def ele_relative(ensemble: SeleEnsemble, measured_ele: NDArray[np.float64]) -> NDArray[np.float64]:
    """Relative L2 misfit of every draw against the measurement, shape ``(n_samples,)``."""
    measured = np.asarray(measured_ele, dtype=np.float64).ravel()
    return np.linalg.norm(ensemble.ele - measured, axis=1) / np.linalg.norm(measured)


def misfit_quantiles(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
        quantiles: Sequence[float] = (0, 5, 25, 50, 75, 95, 100),
) -> Dict[float, Tuple[float, float]]:
    """Distribution of the misfit itself, as ``{quantile: (mse, relative)}``."""
    mse = ele_mse(ensemble, measured_ele)
    relative = ele_relative(ensemble, measured_ele)
    return {q: (float(np.percentile(mse, q)), float(np.percentile(relative, q)))
            for q in quantiles}


def select(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
        tolerance: float = TOLERANCE,
) -> Selection:
    """The best draw, and every draw whose relative ELE misfit is within ``tolerance``."""
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    misfit = ele_relative(ensemble, measured_ele)
    order = np.argsort(misfit)
    keep = order[misfit[order] <= tolerance]
    met = keep.size > 0
    if not met:
        warnings.warn(
            f"no draw reproduces the measured ELE within {tolerance:.1%}; the best misses by "
            f"{misfit[order[0]]:.2%}. Reporting that draw alone, with no close-enough set.",
            stacklevel=2)
        keep = order[:1]

    best = order[0]
    return Selection(
        tolerance=tolerance,
        z_cm=ensemble.z_cm,
        best_sele=ensemble.sele[best],
        best_ele=ensemble.ele[best],
        best_params=ensemble.params[best],
        best_misfit=float(misfit[best]),
        kept=replace(ensemble, sele=ensemble.sele[keep], ele=ensemble.ele[keep],
                     params=ensemble.params[keep]),
        met=met,
    )


def scalar_intervals(selection: Selection) -> Dict[str, Tuple[float, float, float]]:
    """``{name: (low, best, high)}``: the best draw's value and its range over the kept set.

    Propagating to the scalar beats reading it off the envelope. The envelope's edges are the
    extremes at each depth separately, so it loses the correlation between depths -- it cannot
    express that a draw with a high peak also tends to put that peak deeper -- and peak depth
    cannot be read off it at all.
    """
    z_um = selection.z_cm / 1e-4

    def quantities(sele):
        sele = np.atleast_2d(sele)
        return {
            "surface SELE": sele[:, 0],
            "peak SELE": sele.max(axis=1),
            "peak depth [um]": z_um[sele.argmax(axis=1)],
            "mean SELE 0-3um": sele[:, z_um < 3.0].mean(axis=1),
        }

    best = quantities(selection.best_sele)
    return {name: (float(values.min()), float(best[name][0]), float(values.max()))
            for name, values in quantities(selection.kept.sele).items()}


def report(selection: Selection, ensemble: SeleEnsemble,
           measured_ele: NDArray[np.float64]) -> str:
    lines = ["  ELE misfit across the draws (MSE, and the same as relative L2):"]
    for q, (mse, rel) in misfit_quantiles(ensemble, measured_ele).items():
        lines.append(f"    {q:3.0f}th   {mse:.3e}   {rel * 100:7.3f}%")

    if not selection.met:
        lines.append(f"  WARNING: no draw within {selection.tolerance:.1%}; the best misses "
                     f"by {selection.best_misfit:.2%} and is reported alone.")
    else:
        lines.append(f"  {selection.n_kept} of {ensemble.sele.shape[0]} draws within "
                     f"{selection.tolerance:.1%}; best draw {selection.best_misfit:.3%}")

    z_um = selection.z_cm / 1e-4
    relative = np.divide(selection.high - selection.low, np.abs(selection.best_sele),
                         out=np.full_like(selection.best_sele, np.nan),
                         where=selection.best_sele != 0)
    lines.append("  envelope width relative to the best draw, by depth zone:")
    lines.append("    " + "".join(
        f"{f'{lo:g}-{hi:g}um':>11}" for lo, hi in ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0))))
    lines.append("    " + "".join(
        f"{np.nanmedian(relative[(z_um >= lo) & (z_um < hi)]):10.3f} "
        for lo, hi in ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0))))

    lines.append("  quantities worth quoting (best draw, [range over the close-enough set]):")
    for name, (low, best, high) in scalar_intervals(selection).items():
        lines.append(f"    {name:<18} {best:11.4g}   [{low:.4g}, {high:.4g}]")
    return "\n".join(lines)


__all__ = [
    "TOLERANCE", "Selection", "ele_mse", "ele_relative", "misfit_quantiles", "select",
    "scalar_intervals", "report",
]
