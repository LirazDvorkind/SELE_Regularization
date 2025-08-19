"""Core Tikhonov solver and κ‑sweep utilities."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from scipy.optimize import nnls

def solve_tikhonov(G: NDArray, J: NDArray, L: NDArray, kappa: float) -> NDArray:
    """Solve the Tikhonov‑regularized least squares problem.

    min_S ||G S − J||² + κ² ||L S||²
    using the normal‑equations form.

    Returns
    -------
    S : ndarray, shape (N,)
    """
    L_force_last_zero = np.zeros(L.shape)
    L_force_last_zero[-1, -1] = 1
    K = np.vstack((G, kappa * L, L_force_last_zero))
    rhs = np.concatenate((J, np.zeros(L.shape[0]), np.zeros(L.shape[0])))
    S, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    #S, _ = nnls(K, rhs)
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

def _curvature(x: NDArray, y: NDArray) -> NDArray:
    """Numerical curvature of parametric curve (x(t), y(t))."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    return np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-15)

# TODO: 1. check if there are built in knee finding algorithms for tikhonov
#  2. try the min distance to [0,0] in log scale trick
def find_knee(residuals: NDArray, seminorms: NDArray, kappa_vals: NDArray
             ) -> Tuple[float, int]:
    """Locate κ_knee using maximum curvature of the L‑curve."""
    log_r = np.log10(residuals)
    log_s = np.log10(seminorms)
    kappa_vals = np.asarray(kappa_vals)
    curv = _curvature(log_s, log_r)
    idx = int(np.argmax(curv))
    return float(kappa_vals[idx]), idx
