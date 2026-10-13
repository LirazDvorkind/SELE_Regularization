"""Steady-state 1-D drift-diffusion: Poisson + electron and hole continuity.

Box-integration finite volumes with Scharfetter-Gummel fluxes, solved by fully coupled
Newton in the scaled potentials ``(psi, phi_n, phi_p) / Vt``. Quasi-Fermi potentials rather
than densities are the unknowns because GaAs minority densities span ~25 decades
(ni ~ 2e6 cm^-3), which makes a Newton system in n and p hopelessly ill-conditioned.

Sign conventions: Va is applied to the p-side contact at x = L, the n-side contact at x = 0
is grounded, so Va > 0 is forward bias. Energies are in eV relative to the grounded
contact's Fermi level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from src.drift_diffusion.device import PNJunction
from src.drift_diffusion.generation import GenerationProfile
from src.drift_diffusion.material import Q

MAX_NEWTON_STEP = 2.0       # in units of Vt; larger updates are scaled down
NEWTON_TOLERANCE = 1e-10    # in units of Vt
MAX_NEWTON_ITERATIONS = 100


@dataclass(frozen=True)
class Discretisation:
    x: NDArray[np.float64]           # (N,) node positions, cm
    h: NDArray[np.float64]           # (N-1,) edge lengths
    volume: NDArray[np.float64]      # (N,) control-volume lengths
    doping: NDArray[np.float64]      # (N,) control-volume-averaged Nd - Na
    mu_n: NDArray[np.float64]        # (N-1,) on edges
    mu_p: NDArray[np.float64]


def discretise(device: PNJunction, x: NDArray[np.float64]) -> Discretisation:
    x = np.asarray(x, dtype=np.float64)
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("mesh must be strictly increasing")
    mid = 0.5 * (x[:-1] + x[1:])
    total_doping = device.donors(mid) + device.acceptors(mid)

    volume = np.zeros_like(x)
    volume[:-1] += h / 2
    volume[1:] += h / 2
    doping = np.zeros_like(x)
    doping[:-1] += h / 2 * device.net_doping(mid)
    doping[1:] += h / 2 * device.net_doping(mid)
    doping /= volume

    m = device.material
    return Discretisation(x, h, volume, doping,
                          m.electron_mobility(total_doping), m.hole_mobility(total_doping))


@dataclass(frozen=True)
class DeviceState:
    """A converged solution. Node arrays are (N,); current arrays live on the N-1 edges."""

    device: PNJunction
    x_cm: NDArray[np.float64]
    applied_voltage: float
    generation: NDArray[np.float64]  # cm^-3 s^-1
    psi: NDArray[np.float64]         # V, intrinsic-level potential
    phi_n: NDArray[np.float64]       # V
    phi_p: NDArray[np.float64]       # V
    n: NDArray[np.float64]           # cm^-3
    p: NDArray[np.float64]
    r_srh: NDArray[np.float64]       # cm^-3 s^-1
    r_rad: NDArray[np.float64]
    x_edge_cm: NDArray[np.float64]
    jn: NDArray[np.float64]          # A/cm^2, along +x
    jp: NDArray[np.float64]
    field: NDArray[np.float64]       # V/cm, along +x
    current_density: float           # A/cm^2, into the p contact (forward dark current > 0)

    @property
    def j_total(self) -> NDArray[np.float64]:
        return self.jn + self.jp

    @property
    def ei_ev(self) -> NDArray[np.float64]:
        return -self.psi

    @property
    def ec_ev(self) -> NDArray[np.float64]:
        return self.ei_ev + self.device.material.ec_minus_ei_ev

    @property
    def ev_ev(self) -> NDArray[np.float64]:
        return self.ec_ev - self.device.material.bandgap_ev

    @property
    def vacuum_level_ev(self) -> NDArray[np.float64]:
        return self.ec_ev + self.device.material.electron_affinity_ev

    @property
    def efn_ev(self) -> NDArray[np.float64]:
        return -self.phi_n

    @property
    def efp_ev(self) -> NDArray[np.float64]:
        return -self.phi_p


def _bernoulli(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """``B(x) = x / (exp(x) - 1)``, with a series where the closed form cancels."""
    out = np.empty_like(x)
    small = np.abs(x) < 1e-2
    xs = x[small]
    out[small] = 1.0 - xs / 2 + xs ** 2 / 12 - xs ** 4 / 720 + xs ** 6 / 30240
    xl = x[~small]
    with np.errstate(over="ignore"):
        out[~small] = xl / np.expm1(xl)
    return out


def _bernoulli_derivative(x: NDArray[np.float64], bx: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty_like(x)
    small = np.abs(x) < 1e-2
    xs = x[small]
    out[small] = -0.5 + xs / 6 - xs ** 3 / 180 + xs ** 5 / 5040
    xl, bl = x[~small], bx[~small]
    # B'(x) = B/x - B e^x/(e^x-1), rewritten without e^x so it cannot overflow.
    out[~small] = bl * (1.0 - bl) / xl - bl
    return out


@dataclass(frozen=True)
class _Fields:
    n: NDArray[np.float64]
    p: NDArray[np.float64]
    fn: NDArray[np.float64]          # Jn / q on edges
    fp: NDArray[np.float64]          # Jp / q on edges
    # flux partials w.r.t. (left, right) node unknowns
    fn_ul: NDArray[np.float64]
    fn_ur: NDArray[np.float64]
    fn_al: NDArray[np.float64]
    fn_ar: NDArray[np.float64]
    fp_ul: NDArray[np.float64]
    fp_ur: NDArray[np.float64]
    fp_bl: NDArray[np.float64]
    fp_br: NDArray[np.float64]
    r_srh: NDArray[np.float64]
    r_rad: NDArray[np.float64]
    r_u: NDArray[np.float64]
    r_a: NDArray[np.float64]
    r_b: NDArray[np.float64]
    flux_scale: NDArray[np.float64]  # magnitude of the cancelling terms in each edge flux


def _evaluate(disc: Discretisation, device: PNJunction, u, a, b) -> _Fields:
    m = device.material
    vt, ni = m.thermal_voltage, m.ni
    n = ni * np.exp(u - a)
    p = ni * np.exp(b - u)

    du = u[1:] - u[:-1]
    bp, bm = _bernoulli(du), _bernoulli(-du)
    dbp, dbm = _bernoulli_derivative(du, bp), _bernoulli_derivative(-du, bm)
    cn = disc.mu_n * vt / disc.h
    cp = disc.mu_p * vt / disc.h
    nl, nr, pl, pr = n[:-1], n[1:], p[:-1], p[1:]

    fn = cn * (nr * bp - nl * bm)
    fn_d = cn * (nr * dbp + nl * dbm)
    fn_nl, fn_nr = -cn * bm, cn * bp

    fp = cp * (pl * bp - pr * bm)
    fp_d = cp * (pl * dbp + pr * dbm)
    fp_pl, fp_pr = cp * bp, -cp * bm

    # n p - ni^2 straight from the potentials: computing it as a difference would lose
    # every digit near equilibrium.
    excess = ni ** 2 * np.expm1(b - a)
    denom = m.tau_p * (n + m.n1) + m.tau_n * (p + m.p1)
    r_srh = excess / denom
    r_rad = m.b_rad * excess
    r_n = (p * denom - excess * m.tau_p) / denom ** 2 + m.b_rad * p
    r_p = (n * denom - excess * m.tau_n) / denom ** 2 + m.b_rad * n

    return _Fields(
        n=n, p=p, fn=fn, fp=fp,
        fn_ul=-fn_d + fn_nl * nl, fn_ur=fn_d + fn_nr * nr,
        fn_al=-fn_nl * nl, fn_ar=-fn_nr * nr,
        fp_ul=-fp_d - fp_pl * pl, fp_ur=fp_d - fp_pr * pr,
        fp_bl=fp_pl * pl, fp_br=fp_pr * pr,
        r_srh=r_srh, r_rad=r_rad,
        r_u=r_n * n - r_p * p, r_a=-r_n * n, r_b=r_p * p,
        flux_scale=cn * (nl * np.abs(bm) + nr * np.abs(bp)) + cp * (pl * np.abs(bp) + pr * np.abs(bm)),
    )


def _contact_values(disc: Discretisation, device: PNJunction, applied_voltage: float):
    """Ohmic contacts: neutral and in equilibrium with their own Fermi level."""
    m = device.material
    vt = m.thermal_voltage
    phi = np.array([0.0, applied_voltage]) / vt
    doping = disc.doping[[0, -1]]
    u = np.arcsinh(doping / (2.0 * m.ni)) + phi
    return u, phi


def residual_and_jacobian(disc: Discretisation, device: PNJunction,
                          z: NDArray[np.float64], generation: NDArray[np.float64],
                          applied_voltage: float):
    """Residual F(z) and sparse Jacobian for ``z = [u, a, b]`` (each scaled by Vt)."""
    N = disc.x.size
    u, a, b = z[:N], z[N:2 * N], z[2 * N:]
    m = device.material
    f = _evaluate(disc, device, u, a, b)

    iu, ia, ib = np.arange(N), N + np.arange(N), 2 * N + np.arange(N)
    i = np.arange(1, N - 1)
    left, right = i - 1, i        # edges either side of node i
    vol = disc.volume[i]
    eps_q = m.permittivity * m.thermal_voltage / Q

    F = np.empty(3 * N)
    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # Poisson
    d_flux = eps_q * (u[1:] - u[:-1]) / disc.h
    F[iu[i]] = d_flux[right] - d_flux[left] + vol * (f.p[i] - f.n[i] + disc.doping[i])
    add(iu[i], iu[i + 1], eps_q / disc.h[right])
    add(iu[i], iu[i - 1], eps_q / disc.h[left])
    add(iu[i], iu[i], -eps_q / disc.h[right] - eps_q / disc.h[left] - vol * (f.p[i] + f.n[i]))
    add(iu[i], ia[i], vol * f.n[i])
    add(iu[i], ib[i], vol * f.p[i])

    net = vol * (f.r_srh[i] + f.r_rad[i] - generation[i])

    # Electron continuity: div(Jn)/q = R - G
    F[ia[i]] = f.fn[right] - f.fn[left] - net
    add(ia[i], iu[i], f.fn_ul[right] - f.fn_ur[left] - vol * f.r_u[i])
    add(ia[i], iu[i + 1], f.fn_ur[right])
    add(ia[i], iu[i - 1], -f.fn_ul[left])
    add(ia[i], ia[i], f.fn_al[right] - f.fn_ar[left] - vol * f.r_a[i])
    add(ia[i], ia[i + 1], f.fn_ar[right])
    add(ia[i], ia[i - 1], -f.fn_al[left])
    add(ia[i], ib[i], -vol * f.r_b[i])

    # Hole continuity: div(Jp)/q = G - R
    F[ib[i]] = f.fp[right] - f.fp[left] + net
    add(ib[i], iu[i], f.fp_ul[right] - f.fp_ur[left] + vol * f.r_u[i])
    add(ib[i], iu[i + 1], f.fp_ur[right])
    add(ib[i], iu[i - 1], -f.fp_ul[left])
    add(ib[i], ib[i], f.fp_bl[right] - f.fp_br[left] + vol * f.r_b[i])
    add(ib[i], ib[i + 1], f.fp_br[right])
    add(ib[i], ib[i - 1], -f.fp_bl[left])
    add(ib[i], ia[i], vol * f.r_a[i])

    u_c, phi_c = _contact_values(disc, device, applied_voltage)
    ends = np.array([0, N - 1])
    F[iu[ends]] = u[ends] - u_c
    F[ia[ends]] = a[ends] - phi_c
    F[ib[ends]] = b[ends] - phi_c
    for idx in (iu[ends], ia[ends], ib[ends]):
        add(idx, idx, np.ones(2))

    J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * N, 3 * N))
    return F, J


def _newton(disc, device, z, generation, applied_voltage, poisson_only=False):
    """Damped Newton. Returns ``(z, iterations)``, or ``(None, iterations)`` on failure."""
    N = disc.x.size
    z = z.copy()
    active = np.arange(N) if poisson_only else np.arange(3 * N)
    for iteration in range(1, MAX_NEWTON_ITERATIONS + 1):
        F, J = residual_and_jacobian(disc, device, z, generation, applied_voltage)
        F, J = F[active], J[active][:, active]
        # Row equilibration: continuity rows scale with the local carrier density, which
        # varies over ~25 decades.
        row_scale = 1.0 / np.asarray(abs(J).max(axis=1).todense()).ravel()
        D = sp.diags(row_scale)
        delta = spla.spsolve((D @ J).tocsc(), -row_scale * F)
        if not np.all(np.isfinite(delta)):
            return None, iteration
        step = np.max(np.abs(delta))
        z[active] += delta * min(1.0, MAX_NEWTON_STEP / step)
        if step < NEWTON_TOLERANCE:
            return z, iteration
    return None, MAX_NEWTON_ITERATIONS


def _generation_on_mesh(generation: Optional[GenerationProfile], x) -> NDArray[np.float64]:
    if generation is None:
        return np.zeros_like(x)
    g = np.asarray(generation(x), dtype=np.float64)
    if g.shape != x.shape:
        raise ValueError(f"generation returned shape {g.shape}, expected {x.shape}")
    return g


def _state(disc, device, z, generation, applied_voltage) -> DeviceState:
    N = disc.x.size
    vt = device.material.thermal_voltage
    u, a, b = z[:N], z[N:2 * N], z[2 * N:]
    f = _evaluate(disc, device, u, a, b)
    j_x = Q * (f.fn + f.fp)
    # J is constant in x, but each edge's value is a difference of terms as large as
    # flux_scale, so read it where that roundoff is smallest (inside the depletion region).
    terminal = -float(j_x[np.argmin(f.flux_scale)])
    return DeviceState(
        device=device, x_cm=disc.x, applied_voltage=applied_voltage, generation=generation,
        psi=u * vt, phi_n=a * vt, phi_p=b * vt, n=f.n, p=f.p,
        r_srh=f.r_srh, r_rad=f.r_rad,
        x_edge_cm=0.5 * (disc.x[:-1] + disc.x[1:]),
        jn=Q * f.fn, jp=Q * f.fp,
        field=-vt * np.diff(u) / disc.h,
        current_density=terminal,
    )


def _unknowns(state: DeviceState) -> NDArray[np.float64]:
    vt = state.device.material.thermal_voltage
    return np.concatenate([state.psi, state.phi_n, state.phi_p]) / vt


def solve_equilibrium(device: PNJunction,
                      mesh: Optional[NDArray[np.float64]] = None) -> DeviceState:
    """Dark, zero bias: nonlinear Poisson with flat quasi-Fermi levels."""
    x = device.build_mesh() if mesh is None else np.asarray(mesh, dtype=np.float64)
    disc = discretise(device, x)
    N = x.size
    u0 = np.arcsinh(disc.doping / (2.0 * device.material.ni))
    z0 = np.concatenate([u0, np.zeros(N), np.zeros(N)])
    g = np.zeros(N)
    z, iterations = _newton(disc, device, z0, g, 0.0, poisson_only=True)
    if z is None:
        raise RuntimeError(f"equilibrium Poisson solve did not converge ({iterations} iterations)")
    return _state(disc, device, z, g, 0.0)


def solve(device: PNJunction, applied_voltage: float = 0.0,
          generation: Optional[GenerationProfile] = None,
          initial: Optional[DeviceState] = None,
          mesh: Optional[NDArray[np.float64]] = None,
          max_voltage_step: float = 0.05) -> DeviceState:
    """Steady state at ``applied_voltage`` under ``generation`` (a callable of depth in cm).

    Continues from ``initial`` (default: equilibrium), ramping the voltage in steps no larger
    than ``max_voltage_step`` together with the generation, and halving the step whenever
    Newton fails.
    """
    if initial is None:
        initial = solve_equilibrium(device, mesh)
    elif mesh is not None and not np.array_equal(mesh, initial.x_cm):
        raise ValueError("initial state lives on a different mesh")
    disc = discretise(device, initial.x_cm)

    g_start, g_end = initial.generation, _generation_on_mesh(generation, initial.x_cm)
    v_start = initial.applied_voltage
    dv = applied_voltage - v_start
    max_ds = 1.0 if dv == 0 else min(1.0, max_voltage_step / abs(dv))

    z = _unknowns(initial)
    s, ds = 0.0, max_ds
    while s < 1.0:
        s_next = min(1.0, s + ds)
        trial, _ = _newton(disc, device, z, g_start + s_next * (g_end - g_start),
                           v_start + s_next * dv)
        if trial is None:
            ds /= 2.0
            if ds < 1e-4:
                raise RuntimeError(
                    f"no convergence ramping to Va = {applied_voltage:g} V "
                    f"(stuck at {v_start + s * dv:g} V)")
            continue
        z, s = trial, s_next
        ds = min(max_ds, 2.0 * ds)
    return _state(disc, device, z, g_end, applied_voltage)


def sweep_voltage(device: PNJunction, voltages: Sequence[float],
                  generation: Optional[GenerationProfile] = None,
                  mesh: Optional[NDArray[np.float64]] = None) -> list[DeviceState]:
    """Solve at each voltage in order, each warm-started from the previous one."""
    states = []
    previous = None
    for v in voltages:
        previous = solve(device, v, generation, initial=previous, mesh=mesh if previous is None else None)
        states.append(previous)
    return states
