"""GaAs material parameters for the drift-diffusion solver. Units: cm, s, V, eV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Q = 1.602176634e-19          # C
KB = 1.380649e-23            # J/K
EPS0 = 8.8541878128e-14      # F/cm


@dataclass(frozen=True)
class CaugheyThomas:
    """``mu(N) = mu_min + (mu_max - mu_min) / (1 + (N / n_ref)^alpha)``, N the total doping.

    Set ``mu_min == mu_max`` for a constant mobility.
    """

    mu_max: float
    mu_min: float
    n_ref: float
    alpha: float

    def __call__(self, total_doping: ArrayLike) -> NDArray[np.float64]:
        total_doping = np.asarray(total_doping, dtype=np.float64)
        return self.mu_min + (self.mu_max - self.mu_min) / (
            1.0 + (total_doping / self.n_ref) ** self.alpha)


@dataclass(frozen=True)
class GaAsMaterial:
    """Room-temperature GaAs. Band parameters from the Ioffe database, mobility fits from
    Sotoodeh et al., J. Appl. Phys. 87, 2890 (2000)."""

    temperature_k: float = 300.0
    eps_r: float = 12.9
    electron_affinity_ev: float = 4.07
    bandgap_ev: float = 1.424
    nc: float = 4.7e17
    nv: float = 9.0e18
    tau_n: float = 1e-8
    tau_p: float = 1e-8
    trap_level_ev: float = 0.0          # SRH trap energy relative to the intrinsic level
    b_rad: float = 7.2e-10              # cm^3/s
    electron_mobility: CaugheyThomas = CaugheyThomas(9400.0, 500.0, 6.0e16, 0.394)
    hole_mobility: CaugheyThomas = CaugheyThomas(491.5, 20.0, 1.48e17, 0.38)

    @property
    def thermal_voltage(self) -> float:
        return KB * self.temperature_k / Q

    @property
    def permittivity(self) -> float:
        return self.eps_r * EPS0

    @property
    def ni(self) -> float:
        return float(np.sqrt(self.nc * self.nv)
                     * np.exp(-self.bandgap_ev / (2.0 * self.thermal_voltage)))

    @property
    def ec_minus_ei_ev(self) -> float:
        """The intrinsic level sits off midgap because Nc != Nv."""
        return self.bandgap_ev / 2.0 + 0.5 * self.thermal_voltage * np.log(self.nc / self.nv)

    @property
    def n1(self) -> float:
        return self.ni * float(np.exp(self.trap_level_ev / self.thermal_voltage))

    @property
    def p1(self) -> float:
        return self.ni * float(np.exp(-self.trap_level_ev / self.thermal_voltage))
