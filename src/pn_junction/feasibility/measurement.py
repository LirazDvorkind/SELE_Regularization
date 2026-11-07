"""The synthetic ELE measurement of a junction, and what that measurement can resolve.

G is the project's Beer-Lambert generation matrix with the GaAs ellipsometry constants, on the
existing measurement wavelengths. It ignores the thesis cell's window, back-surface field,
mirror and interference (the repo has no transfer-matrix optics).

The data are generated on a finer mesh than any method inverts on, so no method is handed its
own discretisation of the forward model.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.__init__ import CONFIG
from src.io import load_csv
from src.mesh import _compute_front_generation
from src.optical_constants import load_optical_constants

TRUTH_SPACING_CM = 2e-7
INVERSION_SPACING_CM = 1e-6


class Operator(NamedTuple):
    edges_cm: NDArray[np.float64]
    centres_cm: NDArray[np.float64]
    G: NDArray[np.float64]           # (wavelengths, elements); ELE = G @ SELE
    wavelength_nm: NDArray[np.float64]


def measurement_wavelengths() -> NDArray[np.float64]:
    return load_csv(CONFIG.data_paths.wavelengths).ravel()


def operator(depth_cm: float, spacing_cm: float, k_scale: float = 1.0,
             k_tilt: float = 0.0) -> Operator:
    """G on a linear mesh over ``[0, depth_cm]``.

    ``k_scale`` and ``k_tilt`` perturb the extinction coefficient used to build G -- a common
    factor, and a relative slope across the wavelength band -- to stand in for an imperfect
    optical model.
    """
    wavelengths = measurement_wavelengths()
    constants = load_optical_constants()
    k = np.interp(wavelengths, constants.wavelength_nm, constants.k)
    k_bulk = np.interp(wavelengths, constants.wavelength_nm, constants.k_bulk)
    span = (wavelengths - wavelengths.mean()) / np.ptp(wavelengths)
    factor = k_scale * (1.0 + k_tilt * span)
    n = int(round(depth_cm / spacing_cm))
    edges = np.linspace(0.0, depth_cm, n + 1)
    G = _compute_front_generation(k=k * factor, wavelength_nm=wavelengths, z_cm=edges,
                                  k_bulk=k_bulk * factor)
    return Operator(edges, 0.5 * (edges[:-1] + edges[1:]), G, wavelengths)


def add_noise(ele: NDArray[np.float64], relative_level: float, n_draws: int,
              rng: np.random.Generator) -> NDArray[np.float64]:
    """``(n_draws, wavelengths)`` copies of ``ele`` with independent relative white noise."""
    return ele * (1.0 + relative_level * rng.standard_normal((n_draws, ele.size)))


def blur_width(op: Operator, relative_noise: float) -> NDArray[np.float64]:
    """Backus-Gilbert blur width at each element: how wide a feature the data can see there.

    Keep the singular directions of G that stand above the noise, form the projector onto
    them, and read ``dz / P_ii``. Where the data carry no information the width diverges.
    """
    dz = np.diff(op.edges_cm)
    weighted = op.G / np.sqrt(dz)[None, :]
    _, s, vt = np.linalg.svd(weighted, full_matrices=False)
    kept = vt[s / s[0] > relative_noise]
    diag = np.sum(kept ** 2, axis=0)
    with np.errstate(divide="ignore"):
        return dz / diag


def blind_from(z_cm: NDArray[np.float64], ell_cm: NDArray[np.float64]) -> float:
    """Depth beyond which the blur width stays above twice the depth: the data no longer place
    features there."""
    resolved = np.flatnonzero(ell_cm < 2.0 * z_cm)
    return float(z_cm[resolved[-1] + 1]) if resolved.size and resolved[-1] + 1 < z_cm.size else 0.0
