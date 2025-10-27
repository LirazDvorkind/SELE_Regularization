"""Core Tikhonov solver and κ-sweep utilities."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

from src.__init__ import CONFIG
from src.types.enums import RegularizationMethod


def solve_tikhonov(G: NDArray, B: NDArray, L: NDArray, kappa: float) -> NDArray:
    """Solve the Tikhonov-regularized least squares problem.

    min_S ||G S − B||² + κ² ||L S||²
    using the normal-equations form.
    """
    # ---------- Whitening by median row 2-norm (stabilizes κ tradeoffs) ----------
    def _median_row_norm(A: np.ndarray) -> float:
        if A.ndim != 2 or A.size == 0:
            return 1.0
        rn = np.linalg.norm(A, axis=1)
        rn = rn[np.isfinite(rn)]
        if rn.size == 0:
            return 1.0
        m = float(np.median(rn))
        return m if m > 0 else 1.0

    g_scale = _median_row_norm(G)
    l_scale = _median_row_norm(L)

    # Scale logging (helps pick sensible κ-ranges)
    print(f"[scales] G_medRow2={g_scale:.3e}  L_medRow2={l_scale:.3e}")

    # Build stacked least-squares system
    K_parts = [ G / g_scale, (kappa * L) / l_scale ]
    rhs_parts = [ B / g_scale, np.zeros(L.shape[0]) ]

    # Clean 1×N constraint row to force the last SELE element to zero (optional)
    if CONFIG.force_SELE_last_zero:
        N = G.shape[1]
        C = np.zeros((1, N), dtype=G.dtype)
        C[0, -1] = 1.0  # enforce S[-1] = 0
        K_parts.append(C)
        rhs_parts.append(np.zeros(1, dtype=B.dtype))

    K = np.vstack(K_parts)
    rhs = np.concatenate(rhs_parts)

    # Regular least squares
    S, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    return S



def sweep_kappa(A: NDArray, B: NDArray, L: NDArray, kappa_vals: NDArray) -> Tuple[NDArray, NDArray, List[NDArray]]:
    """Compute residual and seminorm across a range of κ values."""
    residuals = np.empty_like(kappa_vals, dtype=float)
    seminorms = np.empty_like(kappa_vals, dtype=float)
    S_list: List[NDArray] = []
    for i, kappa in enumerate(kappa_vals):
        S = solve_tikhonov(A, B, L, kappa)
        residuals[i] = np.linalg.norm(A @ S - B)
        seminorms[i] = np.linalg.norm(L @ S)
        S_list.append(S)
    return residuals, seminorms, S_list


def find_knee(residuals: NDArray, seminorms: NDArray, kappa_vals: NDArray,
              *, normalize: bool = True) -> Tuple[float, int]:
    """Locate κ_knee by the minimum Euclidean distance to the origin in log–log space."""
    residuals = np.asarray(residuals, dtype=float)
    seminorms = np.asarray(seminorms, dtype=float)
    kappa_vals = np.asarray(kappa_vals)

    eps = np.finfo(float).tiny
    x = np.log10(np.clip(residuals, eps, None))
    y = np.log10(np.clip(seminorms, eps, None))

    if normalize:
        def _norm(v):
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            if vmax == vmin:
                return np.zeros_like(v)
            return (v - vmin) / (vmax - vmin)
        x, y = _norm(x), _norm(y)

    d2 = x * x + y * y
    idx = int(np.nanargmin(d2))
    return float(kappa_vals[idx]), idx


def set_kappa_knee(kappa_vals: NDArray, *, desired_kappa_value: float):
    """Find the closest kappa value to the desired one."""
    idx = np.abs(kappa_vals - desired_kappa_value).argmin()
    return kappa_vals[idx], idx
