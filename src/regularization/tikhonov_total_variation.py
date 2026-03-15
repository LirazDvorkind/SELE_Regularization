"""Core Tikhonov solver and κ-sweep utilities for the 2-parameter (κ₁, κ₂) case."""
from __future__ import annotations
import numpy as np
import cvxpy as cp
from numpy.typing import NDArray
from typing import List, Tuple

from src.__init__ import CONFIG


def solve_tikhonov(G: NDArray, B: NDArray, L1: NDArray, L2: NDArray,
                   kappa1: float, kappa2: float) -> NDArray:
    """Solve:  min_S ||G S − B||² + κ₁||L₂S||₂ + κ₂||L₁S||₁
    """
    N = G.shape[1]
    S = cp.Variable(N)

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
    l1_scale = _median_row_norm(L1)
    l2_scale = _median_row_norm(L2)

    # Build objective function terms
    obj_data = cp.sum_squares(G @ S - B) / (g_scale**2)
    obj_l1 = cp.norm1(L1 @ S) / l1_scale
    obj_l2 = cp.norm(L2 @ S, 2) / l2_scale

    objective = cp.Minimize(obj_data + kappa1 * obj_l2 + kappa2 * obj_l1)

    # Optional constraint: last SELE element = 0
    constraints = []
    if CONFIG.force_SELE_last_zero:
        constraints.append(S[-1] == 0)

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return S.value




def sweep_kappa(G: NDArray, B: NDArray, L1: NDArray, L2: NDArray, kappa_vals1: NDArray, kappa_vals2: NDArray
                ) -> Tuple[NDArray, NDArray, NDArray, List[List[NDArray]]]:
    """Compute residual, seminorm and model residual across (κ₁, κ₂)."""
    n1, n2 = kappa_vals1.size, kappa_vals2.size
    residuals = np.zeros((n1, n2))
    seminorms = np.zeros((n1, n2))
    tv_norms = np.zeros((n1, n2))
    S_list: List[List[NDArray]] = []
    for i, kappa1 in enumerate(kappa_vals1):
        S_list.append([])
        for j, kappa2 in enumerate(kappa_vals2):
            S = solve_tikhonov(G, B, L1, L2, kappa1, kappa2)
            residuals[i, j] = np.linalg.norm(G @ S - B)
            seminorms[i, j] = np.linalg.norm(L2 @ S)
            tv_norms[i, j] = np.linalg.norm(L1 @ S, ord=1)
            S_list[i].append(S)
    return residuals, seminorms, tv_norms, S_list


def find_knee(residuals: NDArray, seminorms: NDArray, tv_norms: NDArray) -> Tuple[int, int]:
    """Find the knee of the L-surface."""
    # Normalize the data
    res_norm = (residuals - np.min(residuals)) / (np.max(residuals) - np.min(residuals))
    sem_norm = (seminorms - np.min(seminorms)) / (np.max(seminorms) - np.min(seminorms))
    tv_norm = (tv_norms - np.min(tv_norms)) / (np.max(tv_norms) - np.min(tv_norms))

    # Find the point closest to the origin
    distances = np.sqrt(res_norm**2 + sem_norm**2 + tv_norm**2)
    i_star, j_star = np.unravel_index(np.argmin(distances), distances.shape)
    return i_star, j_star
