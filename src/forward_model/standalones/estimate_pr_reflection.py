"""Size what the repo's forward model leaves out: back-surface reflection and photon recycling.

    python -m src.forward_model.standalones.estimate_pr_reflection

Method and assumptions: ``docs/pr_reflection_estimate.md``.

Four SELE variants on one finite-volume mesh of the 350 um wafer, all low-injection and linear:

  V0  the repo simulator's escape (normal-incidence T, one escape cone, e^{-alpha z}), no recycling
  V1  angle-resolved direct escape only
  V2  V1 + back/front reflection and multiple passes (SI S4.9), no recycling
  V3  V2 + photon recycling (SI S4.1-S4.6), solved self-consistently

Recycling is linear in the carrier density, so the iteration of SI Fig. S4.2 is one linear solve.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.__init__ import CONFIG
from src.forward_model.sele_simulator import (
    C0_CM_PER_S, HBAR_EV_S, KB_EV_PER_K, TK, B0, effective_lifetime, emission_band,
    intrinsic_concentration, simulate_sele)
from src.io import load_csv
from src.mesh import _compute_front_generation
from src.optical_constants import load_optical_constants

WAFER_CM = 350e-4
# SI Table S5.1 best fit; D from the 1500 cm^2/Vs minority mobility via Einstein, D = mu kT/q.
S51 = dict(p0=1e19, D=1500.0 * KB_EV_PER_K * TK, S=2e7, tau=0.4e-9, alpha_scale=1.0)


def fresnel_unpolarised(mu: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Reflectance inside GaAs (index ``n``) at a GaAs/air interface; ``mu`` = cos(incidence)."""
    sin_t = n * np.sqrt(np.clip(1.0 - mu ** 2, 0.0, 1.0))
    tir = sin_t >= 1.0
    cos_t = np.sqrt(np.clip(1.0 - sin_t ** 2, 0.0, 1.0))
    r_s = ((n * mu - cos_t) / (n * mu + cos_t)) ** 2
    r_p = ((mu - n * cos_t) / (mu + n * cos_t)) ** 2
    return np.where(tir, 1.0, 0.5 * (r_s + r_p))


def wafer_edges(n_side: int = 290, first: float = 1e-7) -> np.ndarray:
    """Geometric cells from both surfaces, 1 nm at the surface, meeting in the middle."""
    widths = first * 1.03 ** np.arange(n_side)
    widths *= (WAFER_CM / 2) / widths.sum()
    half = np.concatenate([[0.0], np.cumsum(widths)])
    return np.concatenate([half, WAFER_CM - half[-2::-1]])


def mu_nodes(n_ref: float, n_tir: int = 24, n_cone: int = 16):
    """Gauss-Legendre in cos(theta), split at the critical angle where R jumps to 1."""
    mu_c = np.sqrt(1.0 - 1.0 / n_ref ** 2)
    nodes, weights = [], []
    for lo, hi, count in ((0.0, mu_c, n_tir), (mu_c, 1.0, n_cone)):
        x, w = np.polynomial.legendre.leggauss(count)
        nodes.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
        weights.append(0.5 * (hi - lo) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def _cell_mean_exp(kappa, near, far):
    """Mean of exp(-kappa s) over s in [near, far]; kappa broadcasts against them."""
    width = far - near
    safe = np.where(kappa * width > 1e-12, kappa * width, 1.0)
    return np.where(kappa * width > 1e-12,
                    (np.exp(-kappa * near) - np.exp(-kappa * far)) / safe, np.exp(-kappa * near))


class Optics:
    """Escape probabilities and the reabsorption kernel on a mesh, energy-resolved."""

    def __init__(self, edges: np.ndarray, p0: float, alpha_scale: float = 1.0, stride: int = 2):
        band = emission_band()
        order = np.argsort(band.photon_energy_ev)[::stride]
        self.energy = band.photon_energy_ev[order]
        self.n = band.n[order]
        wavelength_cm = band.wavelength_nm[order] * 1e-7
        self.alpha = 4.0 * np.pi * band.k[order] / wavelength_cm * alpha_scale
        self.alpha_b = band.alpha_b[order]
        self.t_normal = band.transmission[order]
        self.cone = band.escape_cone[order]
        spacing = np.diff(self.energy)
        self.weight = np.zeros_like(self.energy)
        self.weight[:-1] += spacing / 2
        self.weight[1:] += spacing / 2

        ni = intrinsic_concentration(np.array([p0]))[0]
        self.rate = (1.0 / (np.pi ** 2 * HBAR_EV_S ** 3 * C0_CM_PER_S ** 2 * ni ** 2)
                     * self.alpha_b * self.n ** 2 * self.energy ** 2
                     * np.exp(-self.energy / (KB_EV_PER_K * TK)))
        self.p0 = p0
        self.b_spectral = float(np.sum(self.rate * self.weight))

        self.a, self.b = edges[:-1], edges[1:]
        self.dz = self.b - self.a
        self.z0 = 0.5 * (self.a + self.b)

    def _angles(self, e):
        mu, w = mu_nodes(self.n[e])
        return mu, w, fresnel_unpolarised(mu, self.n[e]), self.alpha[e] / mu

    def escape_front(self, mode: str) -> np.ndarray:
        """``(E, Nz)`` probability that a photon emitted in each cell leaves by the front."""
        out = np.zeros((self.energy.size, self.a.size))
        for e in range(self.energy.size):
            if mode == "simple":
                out[e] = (0.5 * self.t_normal[e] * self.cone[e]
                          * _cell_mean_exp(self.alpha[e], self.a, self.b))
                continue
            mu, w, r, kappa = self._angles(e)
            kappa, r, w = kappa[:, None], r[:, None], w[:, None]
            direct = _cell_mean_exp(kappa, self.a, self.b)
            if mode == "direct":
                out[e] = 0.5 * np.sum(w * (1 - r) * direct, axis=0)
                continue
            back = _cell_mean_exp(kappa, WAFER_CM - self.b, WAFER_CM - self.a) * np.exp(-kappa * WAFER_CM)
            q = r * r * np.exp(-2 * kappa * WAFER_CM)
            out[e] = 0.5 * np.sum(w * (1 - r) * (direct + r * back) / (1 - q), axis=0)
        return out

    def escape_back(self) -> np.ndarray:
        out = np.zeros((self.energy.size, self.a.size))
        for e in range(self.energy.size):
            mu, w, r, kappa = self._angles(e)
            kappa, r, w = kappa[:, None], r[:, None], w[:, None]
            near = _cell_mean_exp(kappa, WAFER_CM - self.b, WAFER_CM - self.a)
            far = _cell_mean_exp(kappa, self.a, self.b) * np.exp(-kappa * WAFER_CM)
            q = r * r * np.exp(-2 * kappa * WAFER_CM)
            out[e] = 0.5 * np.sum(w * (1 - r) * (near + r * far) / (1 - q), axis=0)
        return out

    def absorbed(self, e: int, reflections: bool) -> np.ndarray:
        """``(Nz, Nz)`` total absorbed fraction (carriers *and* FCA) in cell i per photon from cell j."""
        mu, w, r, kappa = self._angles(e)
        d, a, b, z0 = WAFER_CM, self.a[None, :, None], self.b[None, :, None], self.z0[None, None, :]
        kappa, r, w = kappa[:, None, None], r[:, None, None], w[:, None, None]

        def cell(near, far):
            return np.exp(-kappa * near) - np.exp(-kappa * far)

        plus = cell(np.maximum(0, a - z0), np.maximum(0, b - z0))
        minus = cell(np.maximum(0, z0 - b), np.maximum(0, z0 - a))
        if reflections:
            q = r * r * np.exp(-2 * kappa * d)
            plus = plus + (r * cell(2 * d - z0 - b, 2 * d - z0 - a)
                           + r * r * cell(2 * d + a - z0, 2 * d + b - z0)) / (1 - q)
            minus = minus + (r * cell(z0 + a, z0 + b)
                             + r * r * cell(2 * d + z0 - b, 2 * d + z0 - a)) / (1 - q)
        return 0.5 * np.sum(w * (plus + minus), axis=0)

    def recycling_matrix(self, reflections: bool) -> np.ndarray:
        """``K[i, j]``: generation density in cell i per unit excess-carrier density in cell j."""
        k = np.zeros((self.a.size, self.a.size))
        for e in range(self.energy.size):
            gen = self.absorbed(e, reflections) * (self.alpha_b[e] / self.alpha[e])
            k += self.weight[e] * self.rate[e] * gen
        return self.p0 * k * self.dz[None, :] / self.dz[:, None]

    def conservation_residual(self, e: int, j: int, esc_front: np.ndarray, esc_back: np.ndarray) -> float:
        """1 - (absorbed + front escape + back escape) for photons emitted in cell j at energy e."""
        return 1.0 - float(self.absorbed(e, True)[:, j].sum() + esc_front[e, j] + esc_back[e, j])


def continuity_operator(edges, diffusivity, srv, tau_eff):
    dz = np.diff(edges)
    n = dz.size
    g = diffusivity / (0.5 * (dz[:-1] + dz[1:]))
    g_surface = 1.0 / (1.0 / srv + 0.5 * dz[[0, -1]] / diffusivity)
    a = np.zeros((n, n))
    idx = np.arange(n - 1)
    a[idx, idx + 1] = -g / dz[:-1]
    a[idx + 1, idx] = -g / dz[1:]
    a[idx, idx] += g / dz[:-1]
    a[idx + 1, idx + 1] += g / dz[1:]
    a[0, 0] += g_surface[0] / dz[0]
    a[-1, -1] += g_surface[1] / dz[-1]
    a[np.arange(n), np.arange(n)] += 1.0 / tau_eff
    return a


def sele(operator, recycling, escape_weight_per_volume, dz):
    """SELE(z) for unit areal generation in each cell, with ``recycling`` folded into the operator."""
    m = operator if recycling is None else operator - recycling
    return np.linalg.solve(m.T, escape_weight_per_volume * dz) / dz


def run(params: dict, optics: Optics, edges: np.ndarray, kernels: dict, b_scale: float = 1.0):
    dz = np.diff(edges)
    tau_eff = float(effective_lifetime(np.array([params["p0"]]), np.array([params["tau"]]))[0])
    a_op = continuity_operator(edges, params["D"], params["S"], tau_eff)

    def weight(escape):
        return b_scale * optics.p0 * np.einsum("e,e,ej->j", optics.weight, optics.rate, escape)

    out = {
        "V0": sele(a_op, None, weight(kernels["esc_simple"]), dz),
        "V1": sele(a_op, None, weight(kernels["esc_direct"]), dz),
        "V2": sele(a_op, None, weight(kernels["esc_full"]), dz),
        "V3 (recycling, no reflection)": sele(a_op, b_scale * kernels["K_direct"],
                                              weight(kernels["esc_direct"]), dz),
        "V3": sele(a_op, b_scale * kernels["K_full"], weight(kernels["esc_full"]), dz),
    }
    gain = np.max(np.abs(np.linalg.eigvals(np.linalg.solve(a_op, b_scale * kernels["K_full"]))))
    return out, tau_eff, float(gain)


def ele_from(edges, sele_profile):
    constants = load_optical_constants()
    wavelengths = load_csv(CONFIG.data_paths.wavelengths).ravel()
    g = _compute_front_generation(
        k=np.interp(wavelengths, constants.wavelength_nm, constants.k), wavelength_nm=wavelengths,
        z_cm=edges, k_bulk=np.interp(wavelengths, constants.wavelength_nm, constants.k_bulk))
    return g @ sele_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=2, help="use every n-th emission wavelength")
    parser.add_argument("--cache", type=Path, default=None, help=".npz to store/reuse the kernels")
    parser.add_argument("--scan-doping", action="store_true")
    args = parser.parse_args()

    edges = wafer_edges()
    z_um = 0.5 * (edges[:-1] + edges[1:]) * 1e4
    optics = Optics(edges, S51["p0"], S51["alpha_scale"], args.stride)
    print(f"mesh: {z_um.size} cells; emission points: {optics.energy.size}; "
          f"B_spectral = {optics.b_spectral:.3e} cm^3/s (simulator tau_rad uses B0 = {B0:.1e})")

    if args.cache is not None and args.cache.exists():
        kernels = dict(np.load(args.cache))
    else:
        kernels = {"esc_simple": optics.escape_front("simple"), "esc_direct": optics.escape_front("direct"),
                   "esc_full": optics.escape_front("full"),
                   "K_direct": optics.recycling_matrix(False), "K_full": optics.recycling_matrix(True)}
        if args.cache is not None:
            np.savez(args.cache, **kernels)

    # conservation law: every emitted photon is absorbed (carrier or FCA), or escapes front or back
    probe = [(e, j) for e in (0, optics.energy.size // 2, optics.energy.size - 1) for j in (0, 150, 290, 430)]
    back = optics.escape_back()
    worst = max(abs(optics.conservation_residual(e, j, kernels["esc_full"], back)) for e, j in probe)
    print(f"photon conservation, worst |1 - sum| over {len(probe)} (E, z0) probes: {worst:.2e}")

    depths = [0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 30]
    picks = [int(np.argmin(abs(z_um - d))) for d in depths]
    configs = [("S5.1 best fit", S51)]
    configs += [(f"tau_SRH={t:g} ns", {**S51, "tau": t * 1e-9}) for t in (0.1, 1.0, 4.0, 60.0)]
    configs += [(f"S={s:g} cm/s", {**S51, "S": s}) for s in (1e3, 1e5, 1e6)]

    for scale_name, b_scale in (("spectrum as emitted by the simulator", 1.0),
                                ("spectrum scaled up to B0 (upper bound)", B0 / optics.b_spectral)):
        print(f"\n######## {scale_name} (x{b_scale:.2f}) ########")
        for name, params in configs:
            profiles, tau_eff, gain = run(params, optics, edges, kernels, b_scale)
            print(f"\n--- {name}: tau_eff = {tau_eff*1e9:.3f} ns, recycling loop gain (spectral radius) = {gain:.3f}")
            print(f"{'z [um]':>30s}" + "".join(f"{d:>9g}" for d in depths))
            for key, s in profiles.items():
                print(f"{key + ' SELE [%]':>30s}" + "".join(f"{100*s[i]:9.4f}" for i in picks))
            for num, den in (("V1", "V0"), ("V2", "V1"), ("V3", "V2"), ("V3", "V0")):
                print(f"{num + '/' + den:>30s}" + "".join(f"{profiles[num][i]/profiles[den][i]:9.3f}" for i in picks))
            ele = {k: ele_from(edges, v) for k, v in profiles.items()}
            print(f"{'ELE ratio V3/V0 (28 wl): min / median / max':>30s}"
                  f"  {np.min(ele['V3']/ele['V0']):.3f} / {np.median(ele['V3']/ele['V0']):.3f} / {np.max(ele['V3']/ele['V0']):.3f}")

    # check against a known result: V0 on this mesh against the repo simulator itself
    p = S51
    sim = simulate_sele(np.array([[p["p0"], p["D"], p["S"], p["tau"], p["alpha_scale"]]]),
                        0.5 * (edges[:-1] + edges[1:]))[0]
    mine = run(S51, optics, edges, kernels)[0]["V0"]
    keep = (z_um > 0.05) & (z_um < 30)
    print(f"\nV0 vs repo simulate_sele (S5.1 params), 0.05 < z < 30 um: max rel diff "
          f"{np.max(abs(mine[keep]/sim[keep]-1)):.2e}, at z=1/3/10 um: "
          + ", ".join(f"{mine[i]/sim[i]:.4f}" for i in (picks[3], picks[5], picks[7])))

    if args.scan_doping:
        scan_doping(edges, z_um, args.stride)


def scan_doping(edges, z_um, stride):
    """Recycling gain against doping, everything else at the S5.1 values (D held fixed)."""
    print("\n######## doping scan (D, S fixed at S5.1; recycling and reflection both on) ########")
    depths = [0.1, 1, 3, 10, 30]
    picks = [int(np.argmin(abs(z_um - d))) for d in depths]
    print(f"{'p0 [cm^-3]':>11s}{'tau_SRH':>9s}{'spectrum':>10s}{'tau_eff':>9s}{'loop gain':>10s}"
          f"{'V0 peak %':>10s}{'at um':>7s}" + "".join(f"  V3/V0@{d:g}um" for d in depths))
    for p0 in (1e17, 3e17, 1e18, 3e18, 1e19):
        optics = Optics(edges, p0, 1.0, stride)
        kernels = {"esc_simple": optics.escape_front("simple"), "esc_direct": optics.escape_front("direct"),
                   "esc_full": optics.escape_front("full"),
                   "K_direct": optics.recycling_matrix(False), "K_full": optics.recycling_matrix(True)}
        for tau in (0.4e-9, 60e-9):
            for label, scale in (("as-is", 1.0), ("to B0", B0 / optics.b_spectral)):
                profiles, tau_eff, gain = run({**S51, "p0": p0, "tau": tau}, optics, edges, kernels, scale)
                v0, v3 = profiles["V0"], profiles["V3"]
                print(f"{p0:11.0e}{tau*1e9:9.1f}{label:>10s}{tau_eff*1e9:9.3f}{gain:10.3f}"
                      f"{100*v0.max():10.4f}{z_um[v0.argmax()]:7.2f}"
                      + "".join(f"{v3[i]/v0[i]:13.3f}" for i in picks), flush=True)


if __name__ == "__main__":
    main()
