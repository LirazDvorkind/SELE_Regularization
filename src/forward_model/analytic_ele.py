"""ELE straight from the parameters, with no depth axis.

``ELE[l] = sum_i G[l,i] SELE(x_i)`` looks like it needs SELE sampled on a fine mesh, and
computing it that way is what makes dataset generation slow: the intermediate tensor is
``curves x emission wavelengths x depths``, and the depth axis has to be fine enough for the
quadrature to converge.

It is avoidable. On a uniform mesh both factors are geometric in the element index:

    G[l,i]    = c_l (1 - e^{-a_l h}) e^{-a_l i h}
    SELE(x_i) = sum_e w_e k1_e (2 e^{-alpha_e x_i} + a2_e e^{-x_i / Ln}),   x_i = (i + 1/2) h

so the sum over elements is a geometric series with a closed form, and the depth axis
disappears before it is ever allocated:

    S_l(beta) = sum_i G[l,i] e^{-beta x_i}
              = c_l (1 - e^{-a_l h}) e^{-beta h/2} (1 - e^{-a_l W} e^{-beta W})
                                                 / (1 - e^{-a_l h} e^{-beta h})

Every exponential factorises across the two indices, so nothing transcendental is evaluated
on the full three-index tensor -- only multiplications.

This is exact rather than an approximation: it is the same sum the mesh version computes, so
mesh resolution costs nothing and stops being a tradeoff. ``check_analytic_ele.py`` asserts
the two paths agree, which is what keeps this second route honest.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from src.forward_model._parallel import map_curve_blocks
from src.forward_model.ele import measurement_wavelengths, solver_mesh_edges
from src.forward_model.sele_simulator import curve_terms
from src.optical_constants import extinction_at

# The cost of this routine does not depend on the mesh at all, so there is no reason to
# economise: a million elements sits within 1e-7 of the continuum limit, where 8000 is off by
# 2e-3 and the 500-element solver mesh by about 50%.
DEFAULT_ELEMENTS = 1_000_000


class IncidentOptics(NamedTuple):
    """The wavelength-axis half of the geometric series. Arrays are shape ``(L,)``."""

    prefactor: NDArray[np.float64]          # alpha_b / alpha, the carrier-generating fraction
    decay_per_element: NDArray[np.float64]  # exp(-alpha h)
    decay_per_wafer: NDArray[np.float64]    # exp(-alpha W)
    spacing_cm: float                       # h
    width_cm: float                         # W


@lru_cache(maxsize=8)
def incident_optics(n_elements: int) -> IncidentOptics:
    wavelengths = measurement_wavelengths()
    k, k_bulk = extinction_at(wavelengths)
    alpha = 4.0 * np.pi * k / (wavelengths * 1e-7)

    # Only the wafer depth is needed, so take it from the same helper without materialising
    # a million edges.
    width = float(solver_mesh_edges(1)[-1])
    spacing = width / n_elements
    return IncidentOptics(
        prefactor=k_bulk / k,
        decay_per_element=np.exp(-alpha * spacing),
        decay_per_wafer=np.exp(-alpha * width),
        spacing_cm=spacing,
        width_cm=width,
    )


def _series(beta: NDArray[np.float64], optics: IncidentOptics) -> NDArray[np.float64]:
    """``S(beta)``: the operator's response to a single decaying exponential in depth.

    ``beta`` carries a trailing singleton axis to broadcast against the wavelength axis, so
    its three exponentials stay on the curve/emission axes and never reach the product.
    """
    return (optics.prefactor
            * (1.0 - optics.decay_per_element)
            * np.exp(-beta * (optics.spacing_cm / 2.0))
            * (1.0 - optics.decay_per_wafer * np.exp(-beta * optics.width_cm))
            / (1.0 - optics.decay_per_element * np.exp(-beta * optics.spacing_cm)))


def _ele_block(params: NDArray[np.float64], optics: IncidentOptics) -> NDArray[np.float64]:
    terms = curve_terms(params)

    # The generating term: one exponential per (curve, emission wavelength), summed against
    # the operator's response to each.
    generation = np.einsum(
        "be,bel->bl",
        2.0 * terms.weighted_k1,
        _series(terms.alpha[:, :, np.newaxis], optics),
    )

    # The diffusion term shares one decay constant across the whole emission band, so its
    # energy integral collapses to a single weight per curve.
    surface_weight = np.sum(terms.weighted_k1 * terms.a2, axis=1)[:, np.newaxis]
    surface_series = _series(1.0 / terms.diffusion_length, optics)

    return generation + surface_weight * surface_series


def simulate_ele(
        params: NDArray[np.float64],
        n_elements: int = DEFAULT_ELEMENTS,
        curve_chunk: int = 2048,
        workers: Optional[int] = None,
) -> NDArray[np.float64]:
    """``(n_curves, n_wavelengths)`` ELE for a batch of parameter vectors.

    The result does not depend on ``workers``.
    """
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    if params.shape[1] != 5:
        raise ValueError(f"expected (n, 5) parameters, got {params.shape}")

    optics = incident_optics(n_elements)
    return map_curve_blocks(
        params,
        lambda block: _ele_block(block, optics),
        n_columns=optics.prefactor.size,
        curve_chunk=curve_chunk,
        workers=workers,
    )


__all__ = ["DEFAULT_ELEMENTS", "IncidentOptics", "incident_optics", "simulate_ele"]
