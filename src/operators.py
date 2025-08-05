"""Discrete regularisation operators for arbitrary 1‑D meshes."""

from __future__ import annotations

import numpy as np


def _check_mesh(z_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_edges = np.asarray(z_edges, float).ravel()
    if z_edges.ndim != 1 or z_edges.size < 2:
        raise ValueError("z_edges must be 1‑D with at least two points")
    if np.any(np.diff(z_edges) <= 0):
        raise ValueError("z_edges must be strictly increasing")
    dz = np.diff(z_edges)
    return z_edges, dz


def build_L(flag: str, z_edges: np.ndarray) -> np.ndarray:
    """Return a discrete derivative operator for **non‑uniform** meshes.

    Parameters
    ----------
    flag
        One of ``'L0'`` (identity), ``'L1'`` (first derivative),
        or ``'L2'`` (second derivative).
    z_edges
        1‑D array of element **edges** *(length = N + 1)*.

    Returns
    -------
    numpy.ndarray
        Regularisation matrix with shape *(M, N)* where
        *N = len(z_edges) ‑ 1* and

        * ``M = N`` for *L0*
        * ``M = N‑1`` for *L1*
        * ``M = N‑2`` for *L2*
    """
    flag = flag.upper()
    z_edges, dz = _check_mesh(z_edges)
    N = dz.size  # number of unknowns

    if flag == "L0":
        return np.eye(N)

    if flag == "L1":
        rows = []
        for i in range(N - 1):
            row = np.zeros(N)
            row[i] = -1.0 / dz[i]
            row[i + 1] = 1.0 / dz[i]
            row *= np.sqrt(0.5 * (dz[i] + dz[i]))  # simple scaling
            rows.append(row)
        return np.vstack(rows)

    if flag == "L2":
        rows = []
        for i in range(1, N - 1):
            dzm, dzp = dz[i - 1], dz[i]
            denom = dzp + dzm
            row = np.zeros(N)
            row[i - 1] = 2.0 / (dzm * denom)
            row[i] = -2.0 * (1.0 / dzp + 1.0 / dzm) / denom
            row[i + 1] = 2.0 / (dzp * denom)
            row *= np.sqrt(0.5 * denom)
            rows.append(row)
        return np.vstack(rows)

    raise ValueError(f"Unknown L flag '{flag}'.")
