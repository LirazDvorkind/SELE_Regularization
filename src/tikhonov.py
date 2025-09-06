"""Core Tikhonov solver and κ‑sweep utilities."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from scipy.optimize import nnls

from src.io import load_score_model_S
from src.__init__ import CONFIG
from src.types.enums import RegularizationMethod


def solve_tikhonov(regularization_method: RegularizationMethod, G: NDArray, J: NDArray, L: NDArray,
                   kappa: float) -> NDArray:
    """Solve the Tikhonov‑regularized least squares problem.

    min_S ||G S − J||² + κ² ||L S||²
    using the normal‑equations form.

    Returns
    -------
    S : ndarray, shape (N,)
    """
    L_force_last_zero = np.zeros(L.shape)
    L_force_last_zero[-1, -1] = 1 if CONFIG.force_SELE_last_zero else 0
    if regularization_method is RegularizationMethod.NON_UNIFORM_MESH:
        K = np.vstack((G, kappa * L, L_force_last_zero))
        rhs = np.concatenate((J, np.zeros(L.shape[0]), np.zeros(L.shape[0])))
    elif regularization_method is RegularizationMethod.MODEL_SCORING:
        S_score = load_score_model_S()  # 32-long vector
        if G.shape[1] != S_score.size - 1:
            raise ValueError(
                f"Row mismatch between G and L_score: G[1] is {G.shape[1]} but L_score - 1 is {S_score.size - 1}")
        K = np.vstack((G, kappa * L, np.eye(S_score.size-1), L_force_last_zero))
        rhs = np.concatenate((J, np.zeros(L.shape[0]), S_score[:31], np.zeros(L.shape[0])))
    else:
        raise NotImplementedError(
            f"The regularization method {regularization_method} is unsupported by {solve_tikhonov.__name__}")
    # Solves with regular least squares
    S, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    # Solves with non negative least squares
    # S, _ = nnls(K, rhs)  # from scipy.optimize import nnls
    return S


def sweep_kappa(regularization_method: RegularizationMethod, A: NDArray, B: NDArray, L: NDArray, kappa_vals: NDArray
                ) -> Tuple[NDArray, NDArray, List[NDArray]]:
    """Compute residual and seminorm across a range of κ values."""
    residuals = np.empty_like(kappa_vals)
    seminorms = np.empty_like(kappa_vals)
    S_list: List[NDArray] = []
    for i, kappa in enumerate(kappa_vals):
        S = solve_tikhonov(regularization_method, A, B, L, kappa)
        residuals[i] = np.linalg.norm(A @ S - B)
        seminorms[i] = np.linalg.norm(L @ S)
        S_list.append(S)
    return residuals, seminorms, S_list


def find_knee(residuals: NDArray, seminorms: NDArray, kappa_vals: NDArray,
              *, normalize: bool = True) -> Tuple[float, int]:
    """
    Locate κ_knee by the minimum Euclidean distance to the origin in log–log space.
    Optionally normalizes log-axes to [0,1] to balance scales.

    Returns (kappa_at_knee, knee_index).
    """
    residuals = np.asarray(residuals, dtype=float)
    seminorms = np.asarray(seminorms, dtype=float)
    kappa_vals = np.asarray(kappa_vals)

    if residuals.shape != seminorms.shape:
        raise ValueError("residuals and seminorms must have the same shape")
    if residuals.shape[0] != kappa_vals.shape[0]:
        raise ValueError("kappa_vals length must match residuals/seminorms")

    # Guard against nonpositive values before log
    eps = np.finfo(float).tiny
    x = np.log10(np.clip(residuals, eps, None))
    y = np.log10(np.clip(seminorms, eps, None))

    if normalize:
        def _norm(v: NDArray) -> NDArray:
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            if not np.isfinite(vmin + vmax) or vmax == vmin:
                return np.zeros_like(v)
            return (v - vmin) / (vmax - vmin)

        x, y = _norm(x), _norm(y)

    d2 = x * x + y * y  # distance^2 to (0,0) in log–log space
    idx = int(np.nanargmin(d2))
    return float(kappa_vals[idx]), idx


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
