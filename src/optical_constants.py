"""
GaAs optical constants, read from the paper's own figure -- the single source of optics for
the whole project: the solver's G matrices and the ground-truth test set both come from here.

The paper carries *two* absorption coefficients (main text section 2.2, SI section S2), and
the distinction matters for the generation matrix:

* ``alpha`` -- from the ellipsometry model *including* the Drude oscillator. Free-carrier
  absorption attenuates the beam, so this is what sets how fast light dies away with depth.
* ``alpha_b`` -- from the same model with the Drude term removed. FCA is an intra-band
  transition and produces no mobile charge carriers, so only this fraction of the absorbed
  photons contributes to generation (and hence to luminescence).

Optical generation therefore carries the prefactor ``alpha_b / alpha``, exactly as in SI
eq. S4.1. Over the 400-670 nm measurement band the factor runs 0.976-0.998, a small but
wavelength-dependent tilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.matlab_fig import read_fig

_FIG_PATH = (Path(__file__).resolve().parents[1] / "Data" / "Tamir_paper_SELE_figs"
             / "Optical constants GaAs.fig")

# DisplayName of each series in the figure. The figure draws the constants twice (a full
# view plus a zoom on the absorption edge), so the first match of each name is taken.
_SERIES = {
    "n": "n with drude",
    "n_bulk": "n no drude",
    "k": "k with drude",
    "k_bulk": "k no drude",
}


@dataclass(frozen=True)
class OpticalConstants:
    """Ellipsometry-derived constants on the figure's own 601-point wavelength grid."""

    wavelength_nm: NDArray[np.float64]
    n: NDArray[np.float64]
    n_bulk: NDArray[np.float64]
    k: NDArray[np.float64]
    k_bulk: NDArray[np.float64]


def load_optical_constants() -> OpticalConstants:
    lines = [line for axes in read_fig(str(_FIG_PATH)) for line in axes.lines]

    found = {}
    for field, display_name in _SERIES.items():
        for line in lines:
            if line.display_name.strip().lower() == display_name:
                found[field] = line
                break
        else:
            raise ValueError(f"{_FIG_PATH.name} has no series named {display_name!r}")

    wavelength_nm = found["n"].x
    for field, line in found.items():
        if not np.array_equal(line.x, wavelength_nm):
            raise ValueError(f"series {field!r} is on a different wavelength grid")

    return OpticalConstants(
        wavelength_nm=wavelength_nm,
        n=found["n"].y,
        n_bulk=found["n_bulk"].y,
        k=found["k"].y,
        k_bulk=found["k_bulk"].y,
    )


def extinction_at(wavelength_nm: NDArray[np.float64]) \
        -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate ``(k, k_bulk)`` onto the requested wavelengths."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
    constants = load_optical_constants()

    outside = (wavelength_nm < constants.wavelength_nm.min()) | \
              (wavelength_nm > constants.wavelength_nm.max())
    if np.any(outside):
        raise ValueError(
            f"{wavelength_nm[outside]} nm lies outside the measured range "
            f"[{constants.wavelength_nm.min():g}, {constants.wavelength_nm.max():g}] nm."
        )

    return (np.interp(wavelength_nm, constants.wavelength_nm, constants.k),
            np.interp(wavelength_nm, constants.wavelength_nm, constants.k_bulk))


def absorption_coefficients(wavelength_nm: NDArray[np.float64]) \
        -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(alpha, alpha_b)`` in cm^-1: total attenuation, and the carrier-generating part."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
    k, k_bulk = extinction_at(wavelength_nm)
    wavelength_cm = wavelength_nm * 1e-7
    return 4.0 * np.pi * k / wavelength_cm, 4.0 * np.pi * k_bulk / wavelength_cm
