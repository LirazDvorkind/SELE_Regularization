"""Band diagrams of the default GaAs junction: equilibrium, forward bias, and under light.

Edit the device, wavelength and flux below; figures go to ``results/drift_diffusion/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.drift_diffusion import generation, plotting, solver
from src.drift_diffusion.device import PNJunction

DEVICE = PNJunction()
LIGHT = generation.beer_lambert(wavelength_nm=532.0, photon_flux_cm2_s=1e17)
FORWARD_BIAS = 0.8
OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "drift_diffusion"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Vbi = {DEVICE.built_in_voltage():.3f} V, ni = {DEVICE.material.ni:.3g} cm^-3")

    equilibrium = solver.solve_equilibrium(DEVICE)
    dark_forward = solver.solve(DEVICE, FORWARD_BIAS, initial=equilibrium)
    light_sc = solver.solve(DEVICE, 0.0, LIGHT, initial=equilibrium)
    light_forward = solver.solve(DEVICE, FORWARD_BIAS, LIGHT, initial=light_sc)

    states = [equilibrium, dark_forward, light_sc, light_forward]
    titles = ["Equilibrium", f"Dark, $V_a$ = {FORWARD_BIAS} V",
              "Illuminated, short circuit", f"Illuminated, $V_a$ = {FORWARD_BIAS} V"]
    plotting.plot_band_diagrams(states, titles).savefig(OUT_DIR / "band_diagrams.png", dpi=150)
    plotting.plot_overview(light_sc).savefig(OUT_DIR / "overview_illuminated_sc.png", dpi=150)

    voltages = np.round(np.arange(0.0, 1.05, 0.025), 3)
    dark = solver.sweep_voltage(DEVICE, voltages)
    lit = solver.sweep_voltage(DEVICE, voltages, LIGHT)
    j_dark = np.array([s.current_density for s in dark])
    j_lit = np.array([s.current_density for s in lit])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(voltages, j_lit * 1e3, color=plotting.VERMILION, ls="-", marker="o", ms=3,
            label="illuminated")
    ax.plot(voltages, j_dark * 1e3, color=plotting.BLUE, ls="--", marker="s", ms=3, label="dark")
    ax.axhline(0, color=plotting.GREY, lw=0.8)
    ax.set_ylim(1.5 * j_lit.min() * 1e3, -1.5 * j_lit.min() * 1e3)
    ax.set_xlabel(r"$V_a$ (V)")
    ax.set_ylabel(r"$J$ (mA/cm$^2$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "jv.png", dpi=150)

    voc = np.interp(0.0, j_lit, voltages)
    print(f"Jsc = {light_sc.current_density * 1e3:.3f} mA/cm^2, Voc = {voc:.3f} V")
    print(f"figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
