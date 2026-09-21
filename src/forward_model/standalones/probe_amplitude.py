"""Why the lifetime sweep will not fit: shape or brightness?

The residual on those curves bottoms out around 6-9 percent while every shape in the test set
is reproducible to a fraction of a percent. That points at overall scale rather than at the
wavelength dependence, and scale is worth separating because the two have different cures --
a shape the family cannot make is a missing physical term, a brightness it cannot reach may
only be a bound in the wrong place.

For each curve this reports the best fit attainable when the prediction is allowed a free
multiplier, and what that multiplier has to be. A multiplier far from one with a near-zero
shape residual means the family makes the right curve at the wrong brightness.

It then asks whether the brightness is reachable at all, by scanning the box for the
brightest and dimmest ELE it can produce.

    python -m src.forward_model.standalones.probe_amplitude
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from src.forward_model.analytic_ele import simulate_ele
from src.forward_model.parameters import PARAMETER_NAMES, ParameterSpec
from src.test_set.loader import load_test_set

_DRAWS = 200_000
_SEED = 11


def _scaled_residual(predicted: NDArray[np.float64], measured: NDArray[np.float64]):
    """Best residual per curve once the prediction may be rescaled freely, and that scale.

    The optimal multiplier is the least-squares projection of the measurement onto each
    prediction, so this is the residual of the shape alone.
    """
    scale = (predicted @ measured) / np.einsum("ij,ij->i", predicted, predicted)
    residual = (np.linalg.norm(scale[:, None] * predicted - measured, axis=1)
                / np.linalg.norm(measured))
    return residual, scale


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    spec = ParameterSpec()
    rng = np.random.default_rng(_SEED)
    params = spec.sample(_DRAWS, rng)
    predicted = simulate_ele(params)
    brightness = predicted.mean(axis=1)

    print(f"  ELE brightness reachable in the box (mean over wavelength):")
    print(f"    dimmest {brightness.min():.3e}   brightest {brightness.max():.3e}")
    brightest = params[np.argmax(brightness)]
    print(f"    at the brightest: {spec.describe(brightest)}")

    print(f"\n  curve         free fit   needed scale   direct fit   measured / brightest")
    for curve in load_test_set(include_reference=False):
        measured = np.asarray(curve.ele, dtype=np.float64).ravel()
        residual, scale = _scaled_residual(predicted, measured)
        best = int(np.argmin(residual))

        direct = (np.linalg.norm(predicted - measured, axis=1) / np.linalg.norm(measured)).min()
        headroom = measured.mean() / brightness.max()
        print(f"  {curve.curve_id:<12} {residual[best] * 100:7.3f}% {scale[best]:14.3f} "
              f"{direct * 100:11.3f}% {headroom:21.3f}")


if __name__ == "__main__":
    main()
