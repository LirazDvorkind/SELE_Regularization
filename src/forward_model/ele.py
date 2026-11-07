"""SELE -> ELE on an explicit mesh, using the project's one forward model.

G comes from ``src.test_set.build_test_set``, which is the same operator
``src.mesh._linear_mesh`` builds and the same one the ground-truth test set was measured
with. Reimplementing Beer-Lambert here would make any benchmark measure the gap between two
G's rather than reconstruction error.

For generating data this module is the *reference*, not the workhorse. ``analytic_ele``
computes the same sum in closed form and is around two orders of magnitude faster, because
the depth axis it needs turns out to be summable. The two are checked against each other by
``standalones/check_analytic_ele.py``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src.test_set.build_test_set import (
    measurement_wavelengths,
    photogeneration_matrix_on_edges,
    solver_mesh_edges,
)


def element_centres(z_edges_cm: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (z_edges_cm[:-1] + z_edges_cm[1:])


@lru_cache(maxsize=8)
def mesh_and_operator(n_elements: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(element centres, G)`` for a linear mesh of ``n_elements`` spanning the wafer depth W."""
    edges = solver_mesh_edges(n_elements)
    return element_centres(edges), photogeneration_matrix_on_edges(edges)


def sele_to_ele(sele: NDArray[np.float64], operator: NDArray[np.float64]) -> NDArray[np.float64]:
    """``ELE = G @ SELE``, batched over the leading axis of ``sele``.

    No reflectance factor: ``phi_abs = phi_0 * A`` already normalises by the absorbed
    fraction, so applying it to G again would double-count.
    """
    return np.asarray(sele, dtype=np.float64) @ np.asarray(operator, dtype=np.float64).T


__all__ = [
    "element_centres",
    "mesh_and_operator",
    "measurement_wavelengths",
    "sele_to_ele",
    "solver_mesh_edges",
]
