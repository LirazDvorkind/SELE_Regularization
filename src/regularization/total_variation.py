"""Core Tikhonov solver and κ-sweep utilities for the 2-parameter (κ₁, κ₂) case and pure TV."""
from __future__ import annotations
import numpy as np
import cvxpy as cp
from numpy.typing import NDArray
from typing import List, Tuple

from src.__init__ import CONFIG


def _median_row_norm(A: np.ndarray) -> float:
    if A.ndim != 2 or A.size == 0:
        return 1.0
    rn = np.linalg.norm(A, axis=1)
    rn = rn[np.isfinite(rn)]
    if rn.size == 0:
        return 1.0
    m = float(np.median(rn))
    return m if m > 0 else 1.0


def solve_tv(G: NDArray, B: NDArray, kappa: float) -> NDArray:
    """Solve:  min_S ||GS - B||² + κ·||L₁S||₁,  S ≥ 0  (pure TV, single-parameter)."""
    from src.operators import build_L
    from src.types.enums import LFlag
    N = G.shape[1]
    L1 = build_L(LFlag.L1, N)
    S = cp.Variable(N)
    b_scale = float(np.linalg.norm(B))
    Gs = G / b_scale
    Bs = B / b_scale
    objective = cp.Minimize(
        cp.sum_squares(Gs @ S - Bs)
        + kappa * cp.norm1(L1 @ S)
    )
    constraints = [S >= 0]
    if CONFIG.force_SELE_last_zero:
        constraints.append(S[-1] == 0)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return S.value, problem


def sweep_kappa_tv(G: NDArray, B: NDArray, kappa_vals: NDArray
                   ) -> Tuple[NDArray, NDArray, List[NDArray]]:
    """Sweep κ for pure TV regularization; return (residuals, tv_norms, S_list)."""
    from src.operators import build_L
    from src.types.enums import LFlag
    L1 = build_L(LFlag.L1, G.shape[1])

    # Diagnostics: print problem structure once
    G_rank = np.linalg.matrix_rank(G)
    print(f"\n[TV Sweep Diagnostics] G.shape={G.shape}, rank(G)={G_rank}, underdetermined={(G.shape[1] > G_rank)}")

    residuals = np.empty_like(kappa_vals, dtype=float)
    tv_norms = np.empty_like(kappa_vals, dtype=float)
    S_list: List[NDArray] = []
    for i, kappa in enumerate(kappa_vals):
        S, problem = solve_tv(G, B, kappa)
        residuals[i] = np.linalg.norm(G @ S - B)
        tv_norms[i] = np.linalg.norm(L1 @ S, ord=1)
        S_list.append(S)

        # Print per-iteration diagnostics
        if i % max(1, len(kappa_vals) // 5) == 0:  # print ~5 times across sweep
            print(f"  κ={kappa:.2e}: status={problem.status:12s}, obj={problem.value:.6e}, res={residuals[i]:.2e}, TV={tv_norms[i]:.2e}")

    return residuals, tv_norms, S_list


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
