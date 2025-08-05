"""Debug helpers for mesh interpolation."""

from __future__ import annotations

import numpy as np


def validate_interpolation(
    G_old: np.ndarray,
    z_old_edges: np.ndarray,
    G_new: np.ndarray,
    z_new_edges: np.ndarray,
    *,
    rtol: float = 1e-3,
) -> np.ndarray:
    """Validate that remeshing approximately conserves total *G*.

    Returns an array of relative errors (per wavelength row).
    """
    sum_old = G_old.sum(axis=1)
    sum_new = G_new.sum(axis=1)
    rel_err = np.abs(sum_new - sum_old) / np.maximum(sum_old, 1e-30)

    print(f"Max rel. error: {rel_err.max():.2e}")
    if (rel_err > rtol).any():
        print("⚠️  Warning: remesh error exceeds tolerance in some rows.")
    else:
        print("✅  Remesh validation passed.")
    return rel_err
