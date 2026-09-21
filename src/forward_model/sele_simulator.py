"""Vectorised NumPy port of ``calc_Sp2.m`` / ``compute_sele_curve.m``.

``Sp(E_emit, x)`` is the probability that a carrier generated at depth ``x`` leaves the wafer
as a photon of emission energy ``E_emit``. SELE is its integral over the emission band.

Both terms of ``Sp`` are exponentials in ``x``, so nothing here needs a solver. The cost is
entirely the ``(curves x emission wavelengths x depths)`` tensor, accumulated in blocks over
the emission axis to keep memory bounded.

That depth axis is why this module is for *curves you want to look at*, on a mesh of a few
hundred points. Anything that only needs the measurement should go through ``analytic_ele``,
which sums the same exponentials in closed form and never allocates a depth axis at all.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import scipy.io
from numpy.typing import NDArray

from src.forward_model._parallel import map_curve_blocks
from src.optical_constants import load_optical_constants

# calc_Sp2.m lines 22-25.
KB_EV_PER_K = 8.6173e-5
HBAR_EV_S = 6.582119569e-16
TK = 299.0
C0_CM_PER_S = 2.998e10

# create_training_set.m lines 26-31.
NC = 8.63e13 * TK ** 1.5
NV = 1.83e15 * TK ** 1.5
C_AUGER = 15e-30
B0 = 2.5e-10

_PL_MAT = (Path(__file__).resolve().parents[2] / "MATLAB SELE Simulation"
           / "Incident_wavelength_dependent_PL.mat")


class EmissionBand(NamedTuple):
    """Everything on the emission grid that does not depend on the sampled parameters."""

    wavelength_nm: NDArray[np.float64]      # (E,), the MATLAB ``wavelength_PL``
    photon_energy_ev: NDArray[np.float64]   # (E,)
    trapz_weight: NDArray[np.float64]       # (E,), integrates over energy ascending
    n: NDArray[np.float64]                  # (E,), refractive index with the Drude term
    transmission: NDArray[np.float64]       # (E,), Fresnel T at normal incidence
    escape_cone: NDArray[np.float64]        # (E,), ``1 - cos(theta_c)``
    alpha_b: NDArray[np.float64]            # (E,), 1/cm, Drude term removed
    k: NDArray[np.float64]                  # (E,), extinction with the Drude term
    k_bulk: NDArray[np.float64]             # (E,), extinction without it


def fresnel_transmission(n_incident: NDArray[np.float64]) -> NDArray[np.float64]:
    """``Fresnel.m`` at normal incidence into vacuum: both polarisations collapse to one term."""
    n_incident = np.asarray(n_incident, dtype=np.float64)
    reflectance = ((n_incident - 1.0) / (n_incident + 1.0)) ** 2
    return 1.0 - reflectance


def _trapezoid_weights(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Weights ``w`` such that ``w @ y`` equals the trapezoidal integral of ``y`` over ``grid``."""
    spacing = np.diff(grid)
    weights = np.zeros_like(grid)
    weights[:-1] += spacing / 2.0
    weights[1:] += spacing / 2.0
    return weights


@lru_cache(maxsize=1)
def emission_band() -> EmissionBand:
    wavelength_nm = np.asarray(
        scipy.io.loadmat(str(_PL_MAT))["wavelength_PL"], dtype=np.float64).ravel()

    constants = load_optical_constants()
    n = np.interp(wavelength_nm, constants.wavelength_nm, constants.n)
    k = np.interp(wavelength_nm, constants.wavelength_nm, constants.k)
    k_bulk = np.interp(wavelength_nm, constants.wavelength_nm, constants.k_bulk)

    photon_energy_ev = 1240.0 / wavelength_nm
    order = np.argsort(photon_energy_ev)

    wavelength_cm = wavelength_nm * 1e-7
    band = EmissionBand(
        wavelength_nm=wavelength_nm,
        photon_energy_ev=photon_energy_ev,
        # The integral runs over ascending energy; scatter the weights back to the original
        # wavelength order so every array here shares one indexing.
        trapz_weight=np.zeros_like(photon_energy_ev),
        n=n,
        transmission=fresnel_transmission(n),
        escape_cone=1.0 - np.cos(np.arcsin(1.0 / n)),
        alpha_b=4.0 * np.pi * k_bulk / wavelength_cm,
        k=k,
        k_bulk=k_bulk,
    )
    weights = np.zeros_like(photon_energy_ev)
    weights[order] = _trapezoid_weights(photon_energy_ev[order])
    return band._replace(trapz_weight=weights)


def effective_lifetime(p0: NDArray[np.float64], tau: NDArray[np.float64]) -> NDArray[np.float64]:
    tau_auger = 1.0 / (p0 ** 2 * C_AUGER)
    tau_rad = 1.0 / (p0 * B0)
    return 1.0 / (1.0 / tau + 1.0 / tau_rad + 1.0 / tau_auger)


def intrinsic_concentration(p0: NDArray[np.float64]) -> NDArray[np.float64]:
    """``ni`` after heavy-doping bandgap narrowing (``compute_sele_curve.m`` lines 5-14)."""
    reduced = p0 / 1e18
    d_eg = (9.71 * reduced ** (1 / 3)
            + 12.19 * reduced ** (1 / 4)
            + 3.88 * reduced ** 0.5) / 1000.0
    e_gap = 1.519 - 5.405e-4 * TK ** 2 / (TK + 204.0) - 0.8 * d_eg
    return np.sqrt(NC * NV) * np.exp(-e_gap / (2.0 * KB_EV_PER_K * TK))


def simulate_sele(
        params: NDArray[np.float64],
        x_cm: NDArray[np.float64],
        curve_chunk: int = 256,
        emission_chunk: int = 16,
        workers: Optional[int] = None,
) -> NDArray[np.float64]:
    """SELE(x) for a batch of parameter vectors.

    ``params`` is ``(n_curves, 5)`` ordered as ``src.forward_model.parameters.PARAMETER_NAMES``.
    Returns ``(n_curves, len(x_cm))``. The result does not depend on ``workers``.
    """
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    if params.shape[1] != 5:
        raise ValueError(f"expected (n, 5) parameters, got {params.shape}")
    x_cm = np.asarray(x_cm, dtype=np.float64).ravel()

    return map_curve_blocks(
        params,
        lambda block: _simulate_block(block, x_cm, emission_chunk),
        n_columns=x_cm.size,
        curve_chunk=curve_chunk,
        workers=workers,
    )


class CurveTerms(NamedTuple):
    """The per-curve coefficients of ``Sp``, before any depth axis is introduced.

    ``Sp(E, x) = k1 * (2 exp(-alpha x) + a2 exp(-x / Ln))``, so these four arrays describe a
    curve completely. Keeping them separate from the depth evaluation is what lets the ELE
    integral below be done in closed form instead of on a mesh.
    """

    weighted_k1: NDArray[np.float64]       # (B, E), k1 folded with the energy-integral weight
    a2: NDArray[np.float64]                # (B, E)
    alpha: NDArray[np.float64]             # (B, E), 1/cm
    diffusion_length: NDArray[np.float64]  # (B, 1), cm


def curve_terms(params: NDArray[np.float64]) -> CurveTerms:
    band = emission_band()
    p0, diffusivity, srv, tau, alpha_scale = (params[:, i][:, None] for i in range(5))

    tau_eff = effective_lifetime(p0, tau)
    ni = intrinsic_concentration(p0)
    diffusion_length = np.sqrt(diffusivity * tau_eff)          # (B, 1)

    # Free-carrier absorption fades out as doping drops. interp1 is linear in its samples, so
    # blending the two interpolated curves is identical to interpolating the blended one.
    k_eff = band.k_bulk + (band.k - band.k_bulk) * (p0 / 1e19)  # (B, E)
    wavelength_cm = band.wavelength_nm * 1e-7
    alpha = 4.0 * np.pi * k_eff / wavelength_cm * alpha_scale   # (B, E)

    energy = band.photon_energy_ev
    emission_rate = (1.0 / (np.pi ** 2 * HBAR_EV_S ** 3 * C0_CM_PER_S ** 2 * ni ** 2)
                     * band.alpha_b * band.n ** 2 * energy ** 2
                     * np.exp(-energy / (KB_EV_PER_K * TK)))
    k2 = band.transmission * emission_rate * p0 * tau_eff * band.escape_cone / 4.0
    k1 = k2 / (1.0 - (alpha * diffusion_length) ** 2)           # (B, E)

    a0 = (diffusivity - srv * diffusion_length) / (srv * diffusion_length + diffusivity)
    a2 = a0 - 1.0 - diffusion_length * alpha * (a0 + 1.0)       # (B, E)

    return CurveTerms(k1 * band.trapz_weight, a2, alpha, diffusion_length)


def _simulate_block(
        params: NDArray[np.float64],
        x_cm: NDArray[np.float64],
        emission_chunk: int,
) -> NDArray[np.float64]:
    terms = curve_terms(params)

    # The exp(-x/Ln) term's depth dependence carries no emission-wavelength index, so its
    # energy integral collapses to one coefficient per curve before touching the depth axis.
    surface_coefficient = np.sum(terms.weighted_k1 * terms.a2, axis=1)[:, None]
    surface_term = surface_coefficient * np.exp(-x_cm[None, :] / terms.diffusion_length)

    generation_term = np.zeros((params.shape[0], x_cm.size), dtype=np.float64)
    for start in range(0, terms.alpha.shape[1], emission_chunk):
        sl = slice(start, start + emission_chunk)
        decay = np.exp(-terms.alpha[:, sl, None] * x_cm[None, None, :])
        generation_term += np.einsum("be,bex->bx", 2.0 * terms.weighted_k1[:, sl], decay)

    return generation_term + surface_term
