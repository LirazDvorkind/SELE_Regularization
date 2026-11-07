"""SELE and SCE of a solved junction, by linear response of its steady state.

SELE(x) is the extra photoluminescence leaving the front per extra electron-hole pair generated
at x; SCE(x) is the extra collected current per pair. Each is the derivative of one output with
respect to the generation at every node, so one adjoint solve with the Newton Jacobian gives the
whole profile. That is the same answer as perturbing each depth in turn and re-solving (paper
eq. 5), without the cancellation of differencing two nearly equal PL totals.

The terminal voltage is held fixed while the generation changes, which is exact at short
circuit. Photon recycling and any back mirror are left out: a photon either escapes through the
front on its first pass or is lost.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import ArrayLike, NDArray

from src.forward_model.sele_simulator import emission_band
from src.pn_junction.drift_diffusion import solver
from src.pn_junction.drift_diffusion.solver import DeviceState


def escape_probability(x_cm: ArrayLike, thermal_voltage: float) -> NDArray[np.float64]:
    """Chance that a band-edge photon emitted at depth x leaves through the front.

    Emission follows the van Roosbroeck-Shockley spectrum of the project's GaAs constants; each
    photon is attenuated on a straight path to the surface (with the Drude term, as attenuation
    is everywhere in the project) and escapes if it falls inside the escape cone and is
    transmitted -- the same per-photon escape as the wafer simulator in ``src/forward_model``.
    """
    band = emission_band()
    energy = band.photon_energy_ev
    spectrum = band.alpha_b * band.n ** 2 * energy ** 2 * np.exp(-energy / thermal_voltage)
    weight = band.trapz_weight * spectrum
    weight /= weight.sum()
    alpha = 4.0 * np.pi * band.k / (band.wavelength_nm * 1e-7)
    per_photon = weight * band.transmission * band.escape_cone / 2.0
    x = np.asarray(x_cm, dtype=np.float64)
    return per_photon @ np.exp(-alpha[:, None] * x.ravel()[None, :])


def photoluminescence(state: DeviceState) -> float:
    """Photons leaving the front, cm^-2 s^-1."""
    disc = solver.discretise(state.device, state.x_cm)
    esc = escape_probability(state.x_cm, state.device.material.thermal_voltage)
    return float(np.sum(disc.volume * esc * state.r_rad))


def _equation_rows(state: DeviceState):
    """Masks of the nodes whose electron / hole rows are continuity equations, not pinned."""
    N = state.x_cm.size
    electron = np.zeros(N, dtype=bool)
    hole = np.zeros(N, dtype=bool)
    electron[1:-1] = hole[1:-1] = True
    hole[0] = state.device.front_srv_cm_s is not None
    electron[-1] = state.device.back_srv_cm_s is not None
    return electron, hole


def _response_per_pair(state: DeviceState, d_output_dz: NDArray[np.float64]) -> NDArray[np.float64]:
    """d(output) / d(pairs generated in node i's cell), for every node, by one adjoint solve."""
    device = state.device
    disc = solver.discretise(device, state.x_cm)
    _, J = solver.residual_and_jacobian(disc, device, solver._unknowns(state),
                                        state.generation, state.applied_voltage)
    # Continuity rows scale with carrier densities spanning ~25 decades; equilibrate first.
    row_scale = 1.0 / np.asarray(abs(J).max(axis=1).todense()).ravel()
    scaled = (sp.diags(row_scale) @ J).tocsc()
    lam = row_scale * spla.spsolve(scaled.T.tocsc(), d_output_dz)

    N = state.x_cm.size
    electron, hole = _equation_rows(state)
    # Generation enters the electron row as +vol*G and the hole row as -vol*G; per pair the
    # volume cancels.
    return -(lam[N:2 * N] * electron - lam[2 * N:] * hole)


def compute_sele(state: DeviceState) -> NDArray[np.float64]:
    """SELE at every mesh node of ``state``, as a fraction (not %)."""
    m = state.device.material
    disc = solver.discretise(state.device, state.x_cm)
    vt = m.thermal_voltage
    a, b = state.phi_n / vt, state.phi_p / vt
    d_rrad_db = m.b_rad * m.ni ** 2 * np.exp(b - a)
    weight = disc.volume * escape_probability(state.x_cm, vt) * d_rrad_db
    N = state.x_cm.size
    d_pl_dz = np.concatenate([np.zeros(N), -weight, weight])
    return _response_per_pair(state, d_pl_dz)


def compute_sce(state: DeviceState) -> NDArray[np.float64]:
    """SCE at every mesh node of ``state``: collected pairs per generated pair."""
    device = state.device
    disc = solver.discretise(device, state.x_cm)
    z = solver._unknowns(state)
    N = state.x_cm.size
    f = solver._evaluate(disc, device, z[:N], z[N:2 * N], z[2 * N:])
    # Total current is conserved edge to edge, so differentiate it on the edge with the least
    # roundoff, as DeviceState.current_density does.
    k = int(np.argmin(f.flux_scale))
    grad = np.zeros(3 * N)
    grad[[k, k + 1]] = [f.fn_ul[k] + f.fp_ul[k], f.fn_ur[k] + f.fp_ur[k]]
    grad[[N + k, N + k + 1]] = [f.fn_al[k], f.fn_ar[k]]
    grad[[2 * N + k, 2 * N + k + 1]] = [f.fp_bl[k], f.fp_br[k]]
    return _response_per_pair(state, grad)
