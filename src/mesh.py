from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plotting import plot_mesh_elements_position_and_size


def non_uniform_mesh(
        z_min: float,
        z_max: float,
        wavelengths: NDArray[np.float64],
        k: NDArray[np.float64],
        lambda_for_alpha: NDArray[np.float64],
        *,
        z_turn: float = 1e-4,
        lin_mesh_size: float = 3e-6,
        exp_base: float = 10.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Create a non-uniform mesh and (re)compute the front-illumination optical generation G
    directly from Beer–Lambert optics using k(λ) and λ.

    Parameters
    ----------
    z_min, z_max
        New mesh bounds [cm]
    wavelengths
        The G wavelengths [nm]
    k
        Extinction coefficient spectrum (unitless), shape (L,).
    lambda_for_alpha
        Wavelengths corresponding to k (nm), shape (L,), strictly positive.
    z_turn
        Depth [cm] where the mesh transitions from linear to exponentially stretched spacing.
    lin_mesh_size
        Linear-part element size [cm].
    exp_base
        Base of the logarithmic spacing; larger values → fewer points in the exponential part.

    Returns
    -------
    z_new, G_new
        z_new : (M,) new mesh edges [cm], strictly increasing.
        G_new : (L, M-1) absorbed photon flux per element [photons·cm^-2·s^-1].

    Notes
    -----
    - Computes **front-illumination** G only.
    - For volumetric generation later, divide each column of G_new by Δz.
    """
    lambda_for_alpha = np.asarray(lambda_for_alpha, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    k = np.interp(wavelengths, lambda_for_alpha, np.asarray(k, dtype=np.float64))

    if k.ndim != 1 or wavelengths.ndim != 1 or k.shape != wavelengths.shape:
        raise ValueError("k and wavelength_nm must be 1D arrays of the same shape.")
    if np.any(wavelengths <= 0.0):
        raise ValueError("wavelength_nm must be strictly positive (nm).")
    if lin_mesh_size <= 0.0:
        raise ValueError("lin_mesh_size must be positive.")
    if exp_base <= 1.0:
        raise ValueError("exp_base must be > 1.0.")
    if not (z_min < z_turn < z_max):
        raise ValueError("z_turn must lie strictly within [z_old[0], z_old[-1]].")

    # ---------- construct new mesh ----------
    n_lin = max(1, math.floor((z_turn - z_min) / lin_mesh_size))
    z_lin = np.linspace(z_min, z_turn, n_lin + 1)  # includes both ends
    lin_mesh_size = float(z_lin[1] - z_lin[0])  # exact value used below

    # Determine number of exponentially stretched points to span [z_turn, z_max]
    # such that the first exp element size ≈ lin_mesh_size.
    span = z_max - z_turn
    ratio = 1.0 + ((exp_base - 1.0) * lin_mesh_size) / span
    if ratio <= 1.0:
        n_exp = 1
    else:
        n_exp = max(1, math.floor(1.0 / np.log10(ratio)))

    factor = np.logspace(0.0, 1.0, n_exp, base=exp_base) - 1.0
    z_exp = z_turn + span * (factor / (exp_base - 1.0))
    z_new = np.hstack([z_lin, z_exp[1:]])  # drop duplicate z_turn

    # ---------- compute G on the new mesh using the optical method ----------
    G_new = compute_front_generation(
        k=k,
        wavelength_nm=wavelengths,
        z_cm=z_new,
        volumetric=False,
    )

    plot_mesh_elements_position_and_size(z_new, z_turn, save=True)

    return z_new, G_new


def compute_front_generation(
        k: ArrayLike,
        wavelength_nm: ArrayLike,
        z_cm: ArrayLike,
        volumetric: bool = False,
) -> NDArray[np.float64]:
    """
    Compute the front-illumination optical generation matrix G for an arbitrary mesh.

    Why:
        SELE extraction uses, for each incident wavelength, the fraction of the incident
        photon flux absorbed in every mesh element. Under Beer–Lambert attenuation,
        integrating the local absorption rate α e^{-α z} over each element [z_i, z_{i+1}]
        yields G_λ,i = (e^{-α z_i} - e^{-α z_{i+1}}) * Φ_ph. Returning this quantity on an
        area basis (and optionally per volume) matches the formulation used to relate G to
        the measured ELE.

    Parameters
    ----------
    k
        Extinction coefficient array (unitless). Shape (L,).
    wavelength_nm
        Wavelengths (nm) corresponding to k. Shape (L,). Values must be > 0.
    z_cm
        Strictly increasing mesh edges (cm). Shape (M,), M >= 2.
    volumetric
        If True, divides each element’s absorbed flux by its thickness Δz to return
        volumetric generation [photons·cm^-3·s^-1]. If False, returns area-based absorbed
        flux per element [photons·cm^-2·s^-1].

    Returns
    -------
    G : ndarray, shape (L, M-1)
        Optical generation matrix for front illumination. Row λ_j, column element i.

    Raises
    ------
    ValueError
        On shape mismatch, non-positive wavelengths, or non-monotonic mesh.
    """
    k = np.asarray(k, dtype=np.float64)
    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float64)
    z_cm = np.asarray(z_cm, dtype=np.float64)

    if k.ndim != 1 or wavelength_nm.ndim != 1:
        raise ValueError("k and wavelength_nm must be 1D arrays.")
    if k.shape != wavelength_nm.shape:
        raise ValueError("k and wavelength_nm must have the same length.")
    if np.any(wavelength_nm <= 0.0):
        raise ValueError("wavelength_nm must be strictly positive.")
    if z_cm.ndim != 1 or z_cm.size < 2:
        raise ValueError("z_cm must be 1D with at least two points (edges).")
    if not np.all(np.diff(z_cm) > 0.0):
        raise ValueError("z_cm must be strictly increasing (monotonic mesh).")

    # α(λ) in cm^-1 from k and λ (convert nm -> cm with 1e-7)
    lambda_cm = wavelength_nm * 1e-7
    alpha_cm_inv = 4.0 * np.pi * k / lambda_cm  # (L,)

    # Compute exponentials at edges for all wavelengths: shape (L, M)
    exp_edges = np.exp(-alpha_cm_inv[:, None] * z_cm[None, :])

    # Integrated absorption over element i: e^{-α z_i} - e^{-α z_{i+1}} -> (L, M-1)
    delta_exp = exp_edges[:, :-1] - exp_edges[:, 1:]
    G = delta_exp

    if volumetric:
        dz = np.diff(z_cm)  # (M-1,)
        if np.any(dz <= 0.0):
            raise ValueError("Non-positive element width encountered in z_cm.")
        G = G / dz[None, :]

    return G
