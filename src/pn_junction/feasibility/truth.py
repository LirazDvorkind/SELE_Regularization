"""The ground-truth SELE of a junction, from the drift-diffusion solver.

The device follows the thesis' thin-film GaAs cell (section 3.4, Table 3.3): a 0.12 um n-emitter
at 2e18 on a 3.5 um p-base at 9e16. The thesis' AlGaAs back-surface field is replaced by a
slow back surface (a homojunction solver has no band offset), the front is an ideal ohmic
contact as in the thesis, and there is no photon recycling or back mirror. This makes the truth
a physically consistent junction profile, not a reproduction of thesis Fig. 3.11.

None of the reconstruction methods may use anything from here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.pn_junction.drift_diffusion import generation, sele, solver
from src.pn_junction.drift_diffusion.device import PNJunction

THESIS_DEVICE = PNJunction(length_cm=3.62e-4, junction_cm=0.12e-4, donor_cm3=2e18,
                           acceptor_cm3=9e16, back_srv_cm_s=1e2)

# Roughly 1 sun of absorbed photons. SELE at short circuit is linear response in low
# injection, so it does not depend on this level (checked over two decades).
LIGHT_BIAS = generation.beer_lambert(600.0, photon_flux_cm2_s=2e17)


class Truth(NamedTuple):
    x_cm: NDArray[np.float64]
    sele: NDArray[np.float64]
    device: PNJunction
    applied_voltage: float
    depletion_cm: tuple[float, float]


def junction_truth(device: PNJunction = THESIS_DEVICE, applied_voltage: float = 0.0,
                   **device_changes) -> Truth:
    if device_changes:
        device = replace(device, **device_changes)
    state = solver.solve(device, applied_voltage, LIGHT_BIAS)
    strong = state.x_edge_cm[np.abs(state.field) > 0.1 * np.abs(state.field).max()]
    # Sub-roundoff negatives inside the depletion region are noise around a true zero.
    return Truth(state.x_cm, np.clip(sele.compute_sele(state), 0.0, None), device,
                 applied_voltage, (float(strong.min()), float(strong.max())))


def resample(truth: Truth, z_cm: NDArray[np.float64]) -> NDArray[np.float64]:
    """The truth at depths ``z_cm``, by depth (zero beyond the device)."""
    return np.interp(z_cm, truth.x_cm, truth.sele, right=0.0)


def filled_dip(truth: Truth, z_cm: NDArray[np.float64], sele: NDArray[np.float64],
               past_depletion_cm: float = 0.1e-4) -> NDArray[np.float64]:
    """``sele`` with the junction dip bridged by a straight line from the emitter peak to just
    past the depletion region: the same profile with no junction in it."""
    emitter = z_cm < truth.device.junction_cm
    start = int(np.argmax(np.where(emitter, sele, -np.inf)))
    end = int(np.argmin(np.abs(z_cm - (truth.depletion_cm[1] + past_depletion_cm))))
    out = sele.copy()
    out[start:end] = np.interp(z_cm[start:end], z_cm[[start, end]], sele[[start, end]])
    return out


# A plain p-type wafer from the wafer simulator: low at the surface, a broad peak near 1.5 um,
# no junction. Parameters in src.forward_model.parameters order (p0, D, S, tau, alpha_scale).
WAFER_PARAMETERS = (1e18, 5.0, 1e5, 5e-9, 1.0)


def wafer_truth(z_cm: NDArray[np.float64]) -> NDArray[np.float64]:
    from src.forward_model.sele_simulator import simulate_sele
    return simulate_sele(np.array([WAFER_PARAMETERS]), z_cm)[0]
