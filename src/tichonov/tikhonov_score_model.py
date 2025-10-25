"""Core Tikhonov solver and κ-sweep utilities for the 2-parameter (κ₁, κ₂) case."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

from src.io import load_score_model_S
from src.__init__ import CONFIG
from src.tichonov import tikhonov_non_uniform


def solve_tikhonov(G: NDArray, B: NDArray, L: NDArray, kappa1: float, kappa2: float) -> NDArray:
    """Solve:  min_S ||G S − B||² + κ₁²||L S||² + κ₂²||S − S_model||²"""
    S_model = load_score_model_S()
    N = G.shape[1]
    if N != S_model.size:
        raise ValueError(f"Column mismatch: G has N={N} but S_model size={S_model.size}")

    K_parts = [G, kappa1 * L, kappa2 * np.eye(N)]
    rhs_parts = [B, np.zeros(L.shape[0]), S_model]

    if CONFIG.force_SELE_last_zero:
        C = np.zeros((1, N), dtype=G.dtype)
        C[0, -1] = 1.0
        K_parts.append(C)
        rhs_parts.append(np.zeros(1, dtype=B.dtype))

    K = np.vstack(K_parts)
    rhs = np.concatenate(rhs_parts)

    S, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    return S


def sweep_kappa(G: NDArray, B: NDArray, L: NDArray, kappa_vals1: NDArray, kappa_vals2: NDArray
                ) -> Tuple[NDArray, NDArray, NDArray, List[List[NDArray]]]:
    """Compute residual, seminorm and model residual across (κ₁, κ₂)."""
    n1, n2 = kappa_vals1.size, kappa_vals2.size
    residuals = np.zeros((n1, n2))
    seminorms = np.zeros((n1, n2))
    model_residuals = np.zeros((n1, n2))
    S_model = load_score_model_S()
    S_list: List[List[NDArray]] = []
    for i, kappa1 in enumerate(kappa_vals1):
        S_list.append([])
        for j, kappa2 in enumerate(kappa_vals2):
            S = solve_tikhonov(G, B, L, kappa1, kappa2)
            residuals[i, j] = np.linalg.norm(G @ S - B)
            seminorms[i, j] = np.linalg.norm(L @ S)
            model_residuals[i, j] = np.linalg.norm(S - S_model)
            S_list[i].append(S)
    return residuals, seminorms, model_residuals, S_list


def slice_knees(residuals: NDArray, seminorms: NDArray, kappa1_vals: NDArray) -> Tuple[NDArray, NDArray]:
    """Find κ₁ knees per κ₂ slice (column)."""
    n1, n2 = residuals.shape
    i_knee = np.zeros(n2, dtype=int)
    k1_knees = np.zeros(n2)
    for j in range(n2):
        k1_knee, i = tikhonov_non_uniform.find_knee(residuals[:, j], seminorms[:, j], kappa1_vals)
        i_knee[j], k1_knees[j] = int(i), float(k1_knee)
    return i_knee, k1_knees


def _minmax_norm(arr: NDArray) -> NDArray:
    arr = np.asarray(arr, dtype=float)
    amin, amax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(amin + amax) or amax == amin:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)
