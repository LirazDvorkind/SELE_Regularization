"""Core Tikhonov solver and κ‑sweep utilities."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from scipy.optimize import nnls


def solve_tikhonov(G: NDArray, J: NDArray, L: NDArray, kappa: float, force_zero: bool = True) -> NDArray:
    """Solve the Tikhonov‑regularized least squares problem.

    min_S ||G S − J||² + κ² ||L S||²
    using the normal‑equations form.

    Returns
    -------
    S : ndarray, shape (N,)
    """
    if force_zero:
        L_force_last_zero = np.zeros(L.shape)
        L_force_last_zero[-1, -1] = 1
        K = np.vstack((G, kappa * L, L_force_last_zero))
        rhs = np.concatenate((J, np.zeros(L.shape[0]), np.zeros(L.shape[0])))
    else:
        K = np.vstack((G, kappa * L))
        rhs = np.concatenate((J, np.zeros(L.shape[0])))
    # Solves with regular least squares
    S, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    # Solves with non negative least squares
    # S, _ = nnls(K, rhs)  # from scipy.optimize import nnls
    return S


def sweep_kappa(A: NDArray, B: NDArray, L: NDArray, kappa_vals: NDArray
                ) -> Tuple[NDArray, NDArray, List[NDArray]]:
    """Compute residual and seminorm across a range of κ values."""
    residuals = np.empty_like(kappa_vals)
    seminorms = np.empty_like(kappa_vals)
    S_list: List[NDArray] = []
    for i, kappa in enumerate(kappa_vals):
        S = solve_tikhonov(A, B, L, kappa)
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
