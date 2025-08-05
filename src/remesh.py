"""Mesh‑aware utilities."""

from __future__ import annotations

import numpy as np


def conservative_remesh_G(
    G_old: np.ndarray,
    z_old_edges: np.ndarray,
    z_new_edges: np.ndarray,
) -> np.ndarray:
    """Conservatively interpolate *G* onto a new non‑uniform mesh.

    *G* is assumed to already include a multiplication by :math:`\Delta z`
    of the **old** mesh. The routine:

    1. Recovers a *density* :math:`g = G/\Delta z_{old}`.
    2. Linearly interpolates *g* to the centres of the new mesh.
    3. Integrates by multiplying with :math:`\Delta z_{new}`.

    This procedure approximately conserves the total generation in every
    wavelength row.

    Parameters
    ----------
    G_old
        Array with shape *(n_\lambda, N_old)*.
    z_old_edges, z_new_edges
        Edge positions (*cm*) of the old and new meshes.

    Returns
    -------
    numpy.ndarray
        *G_new* with shape *(n_\lambda, N_new)*.
    """
    z_old_edges = np.asarray(z_old_edges, float)
    z_new_edges = np.asarray(z_new_edges, float)

    dz_old = np.diff(z_old_edges)
    dz_new = np.diff(z_new_edges)

    centres_old = 0.5 * (z_old_edges[:-1] + z_old_edges[1:])
    centres_new = 0.5 * (z_new_edges[:-1] + z_new_edges[1:])

    density_old = G_old / dz_old  # broadcast along rows

    G_new = np.empty((G_old.shape[0], centres_new.size), dtype=G_old.dtype)
    for idx, row in enumerate(density_old):
        interp = np.interp(centres_new, centres_old, row)
        G_new[idx] = np.clip(interp, 0, None) * dz_new

    return G_new
