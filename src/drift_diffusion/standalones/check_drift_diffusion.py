"""Physics and numerics checks for the drift-diffusion solver. Prints evidence, asserts tolerances."""

from dataclasses import replace

import numpy as np

from src.drift_diffusion import generation, solver
from src.drift_diffusion.device import PNJunction
from src.drift_diffusion.material import Q

DEVICE = PNJunction()
LIGHT = generation.beer_lambert(wavelength_nm=532.0, photon_flux_cm2_s=1e17)


def check(name, value, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value}")
    assert ok, name


def equilibrium():
    eq = solver.solve_equilibrium(DEVICE)
    m = DEVICE.material
    vbi = DEVICE.built_in_voltage()
    check("psi(0) - psi(L) = Vbi", f"{eq.psi[0] - eq.psi[-1]:.6f} vs {vbi:.6f} V",
          abs(eq.psi[0] - eq.psi[-1] - vbi) < 1e-9)

    # The fully coupled system must leave an equilibrium state where it is.
    coupled = solver.solve(DEVICE, 0.0, initial=eq)
    drift = max(np.abs(coupled.phi_n).max(), np.abs(coupled.phi_p).max())
    check("quasi-Fermi levels stay flat at equilibrium", f"{drift:.2e} V", drift < 1e-9)
    check("terminal current at equilibrium", f"{coupled.current_density:.2e} A/cm^2",
          abs(coupled.current_density) < 1e-12)

    reduced = DEVICE.donor_cm3 * DEVICE.acceptor_cm3 / (DEVICE.donor_cm3 + DEVICE.acceptor_cm3)
    v_eff = vbi - 2 * m.thermal_voltage   # majority-carrier tails at the depletion edges
    e_max = np.sqrt(2 * Q * v_eff * reduced / m.permittivity)
    peak = np.abs(eq.field).max()
    check("peak field vs depletion approximation", f"{peak:.4g} vs {e_max:.4g} V/cm",
          abs(peak / e_max - 1) < 0.03)


def jacobian():
    state = solver.solve(DEVICE, 0.7, LIGHT)
    disc = solver.discretise(DEVICE, state.x_cm)
    z = solver._unknowns(state)
    rng = np.random.default_rng(0)
    z = z + 0.1 * rng.standard_normal(z.size)
    v = rng.standard_normal(z.size)
    F, J = solver.residual_and_jacobian(disc, DEVICE, z, state.generation, 0.7)
    eps = 1e-6
    Fp, _ = solver.residual_and_jacobian(disc, DEVICE, z + eps * v, state.generation, 0.7)
    Fm, _ = solver.residual_and_jacobian(disc, DEVICE, z - eps * v, state.generation, 0.7)
    fd = (Fp - Fm) / (2 * eps)
    analytic = J @ v
    scale = np.abs(J).max(axis=1).toarray().ravel() * np.abs(v).max()
    err = np.max(np.abs(fd - analytic) / scale)
    check("analytic vs finite-difference Jacobian", f"{err:.2e} (row-relative)", err < 1e-6)


def continuity():
    for label, state in [("dark, 1.0 V", solver.solve(DEVICE, 1.0)),
                         ("illuminated, 0 V", solver.solve(DEVICE, 0.0, LIGHT))]:
        spread = np.ptp(state.j_total) / abs(state.current_density)
        check(f"J(x) constant ({label})", f"spread/|J| = {spread:.2e}", spread < 1e-2)


def dark_ideality():
    voltages = np.arange(0.6, 1.05, 0.05)
    j = np.array([s.current_density for s in solver.sweep_voltage(DEVICE, voltages)])
    ideality = np.diff(voltages) / (DEVICE.material.thermal_voltage * np.diff(np.log(j)))
    print("       ideality factor, 0.6 -> 1.0 V:", np.round(ideality, 2))
    check("ideality between 1 and 2, falling with bias", f"{ideality[0]:.2f} -> {ideality[-1]:.2f}",
          np.all((ideality > 0.95) & (ideality < 2.05)) and ideality[-1] < ideality[0])


def collection():
    lit = solver.solve(DEVICE, 0.0, LIGHT)
    disc = solver.discretise(DEVICE, lit.x_cm)
    generated = Q * np.sum(disc.volume[1:-1] * lit.generation[1:-1])
    check("|Jsc| <= q * integral(G)", f"{-lit.current_density * 1e3:.3f} <= {generated * 1e3:.3f} mA/cm^2",
          0 < -lit.current_density <= generated)

    # Carriers born inside the junction field are swept out before they can recombine.
    xj = DEVICE.junction_cm
    box = lambda x: np.where((x > xj - 0.02e-4) & (x < xj + 0.08e-4), 1e21, 0.0)
    swept = solver.solve(DEVICE, 0.0, box)
    generated = Q * np.sum(disc.volume[1:-1] * swept.generation[1:-1])
    efficiency = -swept.current_density / generated
    check("collection of generation inside the depletion region", f"{efficiency:.4f}",
          0.99 < efficiency <= 1.0 + 1e-6)

    no_loss = replace(DEVICE, material=replace(DEVICE.material, tau_n=1.0, tau_p=1.0, b_rad=0.0))
    lossless = solver.solve(no_loss, 0.0, box)
    # With no bulk recombination, the only carriers not collected are minority carriers that
    # diffuse into the wrong ohmic contact: Jp at x=0 and Jn at x=L.
    contact_loss = lossless.jp[0] + lossless.jn[-1]
    balance = (-lossless.current_density - contact_loss) / generated
    check("global balance: J = q*integral(G) - minority contact currents",
          f"{balance:.9f} (collection {-lossless.current_density / generated:.5f})",
          abs(balance - 1) < 1e-6)


def mesh_convergence():
    coarse = [solver.solve(DEVICE, 0.0, LIGHT).current_density, solver.solve(DEVICE, 1.0).current_density]
    fine_mesh = DEVICE.build_mesh(refine=2.0)
    fine = [solver.solve(DEVICE, 0.0, LIGHT, mesh=fine_mesh).current_density,
            solver.solve(DEVICE, 1.0, mesh=fine_mesh).current_density]
    for label, c, f in zip(["Jsc (illuminated)", "J (dark, 1.0 V)"], coarse, fine):
        rel = abs(c / f - 1)
        check(f"mesh convergence, {label}", f"{rel:.2e} change on refining 2x", rel < 1e-2)


def main():
    equilibrium()
    jacobian()
    continuity()
    dark_ideality()
    collection()
    mesh_convergence()
    print("all checks passed")


if __name__ == "__main__":
    main()
