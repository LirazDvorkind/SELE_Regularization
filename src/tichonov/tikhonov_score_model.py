"""Core Tikhonov solver and κ‑sweep utilities."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

from src.io import load_score_model_S
from src.__init__ import CONFIG
from src.types.enums import RegularizationMethod


def solve_tikhonov(G: NDArray, B: NDArray, L: NDArray, kappa1: float, kappa2: float) -> NDArray:
    """Solve the Tikhonov‑regularized least squares problem. Find S that minimizes:
    min_S { ||G S − B|| + κ₁||L S|| + κ₂||S - S_model|| }

    Returns
    -------
    S : ndarray, shape (N,)
    """
    L_force_last_zero = np.zeros(L.shape)
    L_force_last_zero[-1, -1] = 1 if CONFIG.force_SELE_last_zero else 0
    S_score = load_score_model_S()  # 32-long vector
    if G.shape[1] != S_score.size:
        raise ValueError(
            f"Row mismatch between G and L_score: G[1] is {G.shape[1]} but L_score is {S_score.size}")
    K = np.vstack((G, kappa1 * L, kappa2*np.eye(S_score.size), L_force_last_zero))
    rhs = np.concatenate((B, np.zeros(L.shape[0]), S_score, np.zeros(L.shape[0])))
    # Solves with regular least squares
    S, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    # Solves with non negative least squares
    # S, _ = nnls(K, rhs)  # from scipy.optimize import nnls
    return S


def sweep_kappa(G: NDArray, B: NDArray, L: NDArray, kappa_vals1: NDArray, kappa_vals2: NDArray
                ) -> Tuple[NDArray, NDArray, NDArray, List[List[NDArray]]]:
    """Compute residual, seminorm and model residual across a range of κ1, κ2 values.
    We solve for minS {||GS-B|| + κ1||LS|| + κ2||S - S_model||}
    The size of each term are the residual, seminorm and model-residual respectively.
    """
    residuals = np.zeros((kappa_vals1.size, kappa_vals2.size))
    seminorms = np.zeros((kappa_vals1.size, kappa_vals2.size))
    model_residuals = np.zeros((kappa_vals1.size, kappa_vals2.size))
    S_score = load_score_model_S()  # 32-long vector
    S_list: List[List[NDArray]] = []
    for i, kappa1 in enumerate(kappa_vals1):
        S_list.append([])
        for j, kappa2 in enumerate(kappa_vals2):
            S = solve_tikhonov(G, B, L, kappa1, kappa2)
            residuals[i, j] = np.linalg.norm(G @ S - B)
            seminorms[i, j] = np.linalg.norm(L @ S)
            model_residuals[i, j] = np.linalg.norm(S - S_score)
            S_list[i].append(S)
    return residuals, seminorms, model_residuals, S_list


def find_knee(residuals: NDArray, seminorms: NDArray, model_residuals: NDArray, kappa1_vals: NDArray, kappa2_vals: NDArray,
              *, normalize: bool = True) -> Tuple[float, float, int, int]:
    """
    Locate κ_knee by the minimum Euclidean distance to the origin in log–log space.
    Optionally normalizes log-axes to [0,1] to balance scales.

    Returns (kappa_at_knee, knee_index).
    """
    residuals = np.asarray(residuals, dtype=float)
    seminorms = np.asarray(seminorms, dtype=float)
    model_residuals = np.asarray(model_residuals, dtype=float)
    kappa1_vals = np.asarray(kappa1_vals)
    kappa2_vals = np.asarray(kappa2_vals)

    if residuals.shape != seminorms.shape:
        raise ValueError("residuals and seminorms must have the same shape")
    if residuals.shape != model_residuals.shape:
        raise ValueError("residuals and model_residuals must have the same shape")
    if residuals.shape[0] != kappa1_vals.shape[0]:
        raise ValueError("kappa_vals1 length must match residuals/seminorms/model_residuals height")
    if residuals.shape[1] != kappa2_vals.shape[0]:
        raise ValueError("kappa_vals2 length must match residuals/seminorms/model_residuals width")

    # Guard against non-positive values before log
    eps = np.finfo(float).tiny
    x = np.log10(np.clip(residuals, eps, None))
    y = np.log10(np.clip(seminorms, eps, None))
    z = np.log10(np.clip(model_residuals, eps, None))

    if normalize:
        def _norm(v: NDArray) -> NDArray:
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            if not np.isfinite(vmin + vmax) or vmax == vmin:
                return np.zeros_like(v)
            return (v - vmin) / (vmax - vmin)

        x, y, z = _norm(x), _norm(y), _norm(z)

    d2 = x * x + y * y + z * z # distance^2 to (0,0) in log–log space
    flat_min_index = np.nanargmin(d2)
    # Convert the flattened (1D) index back to 2D indices (i, j)
    i, j = np.unravel_index(flat_min_index, d2.shape)
    i, j = int(i), int(j) # just convert to int type (from signed int)
    return float(kappa1_vals[i]), float(kappa2_vals[i]), i, j


def set_kappa_knee(kappa_vals: NDArray, *, desired_kappa_value: float):
    """
    Find the closest kappa value to the desired one.
    Use for debugging when you want to set a specific kappa

    Returns
    -------
    closest, idx
        closest : Closest kappa to the desired value.
        idx : Closest kappa index.
    """
    idx = np.abs(kappa_vals - desired_kappa_value).argmin()
    closest = kappa_vals[idx]
    return closest, idx
