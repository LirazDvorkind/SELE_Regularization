"""Optical generation profiles G(x) in cm^-3 s^-1, as callables of depth in cm.

The solver accepts any callable ``G(x_cm)``; these are conveniences for the common cases.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.optical_constants import absorption_coefficients, load_optical_constants

GenerationProfile = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def uniform(rate_cm3_s: float) -> GenerationProfile:
    return lambda x: np.full_like(np.asarray(x, dtype=np.float64), rate_cm3_s)


def tabulated(x_cm: ArrayLike, rate_cm3_s: ArrayLike) -> GenerationProfile:
    """Arbitrary profile sampled on its own depths, interpolated by depth (zero outside)."""
    x_cm = np.asarray(x_cm, dtype=np.float64)
    rate_cm3_s = np.asarray(rate_cm3_s, dtype=np.float64)
    return lambda x: np.interp(x, x_cm, rate_cm3_s, left=0.0, right=0.0)


def beer_lambert(wavelength_nm: float, photon_flux_cm2_s: float,
                 include_reflection: bool = True) -> GenerationProfile:
    """Monochromatic illumination at normal incidence, from the project's GaAs constants.

    Attenuation follows ``alpha`` (with free-carrier absorption) while generation carries
    ``alpha_b`` (without it), as everywhere else in the project.
    """
    alpha, alpha_b = (float(a[0]) for a in absorption_coefficients(np.array([wavelength_nm])))
    transmitted = photon_flux_cm2_s
    if include_reflection:
        constants = load_optical_constants()
        n = np.interp(wavelength_nm, constants.wavelength_nm, constants.n)
        k = np.interp(wavelength_nm, constants.wavelength_nm, constants.k)
        reflectance = ((n - 1.0) ** 2 + k ** 2) / ((n + 1.0) ** 2 + k ** 2)
        transmitted *= 1.0 - reflectance
    return lambda x: transmitted * alpha_b * np.exp(-alpha * np.asarray(x, dtype=np.float64))
