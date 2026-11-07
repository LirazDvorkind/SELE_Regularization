"""The p-n junction geometry and its simulation mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.pn_junction.drift_diffusion.material import GaAsMaterial


@dataclass(frozen=True)
class PNJunction:
    """Abrupt GaAs homojunction: n-type on ``0 <= x < xj``, p-type on ``xj <= x <= L``.

    Light enters at x = 0, so the n side is the emitter. Both ends are ohmic for the majority
    carrier. The minority carrier at each end (holes at the front, electrons at the back) is
    either also pinned to equilibrium -- an ideal ohmic contact, infinite recombination
    velocity, when the SRV is ``None`` -- or lost at a finite surface recombination velocity,
    which stands in for a passivating window or back-surface-field layer.
    """

    length_cm: float = 3.0e-4
    junction_cm: float = 0.5e-4
    donor_cm3: float = 1e18
    acceptor_cm3: float = 1e17
    material: GaAsMaterial = field(default_factory=GaAsMaterial)
    front_srv_cm_s: Optional[float] = None
    back_srv_cm_s: Optional[float] = None

    def __post_init__(self):
        if not 0.0 < self.junction_cm < self.length_cm:
            raise ValueError("the junction must lie strictly inside the device")
        for srv in (self.front_srv_cm_s, self.back_srv_cm_s):
            if srv is not None and srv < 0.0:
                raise ValueError("surface recombination velocities must be non-negative")

    def donors(self, x_cm: ArrayLike) -> NDArray[np.float64]:
        return np.where(np.asarray(x_cm) < self.junction_cm, self.donor_cm3, 0.0)

    def acceptors(self, x_cm: ArrayLike) -> NDArray[np.float64]:
        return np.where(np.asarray(x_cm) < self.junction_cm, 0.0, self.acceptor_cm3)

    def net_doping(self, x_cm: ArrayLike) -> NDArray[np.float64]:
        return self.donors(x_cm) - self.acceptors(x_cm)

    def built_in_voltage(self) -> float:
        m = self.material
        return m.thermal_voltage * float(np.log(self.donor_cm3 * self.acceptor_cm3 / m.ni ** 2))

    def build_mesh(self, h_min_cm: float = 5e-8, h_max_cm: float = 1e-6,
                   growth: float = 1.1, refine: float = 1.0) -> NDArray[np.float64]:
        """Graded mesh, finest at both contacts and at the junction.

        The junction needs resolving on the Debye-length scale; the front contact because
        short-wavelength generation decays within tens of nm there. ``refine > 1`` shrinks
        every spacing by that factor, for convergence studies.
        """
        h_min, h_max = h_min_cm / refine, h_max_cm / refine
        growth = growth ** (1.0 / refine)
        emitter = _graded_segment(self.junction_cm, h_min, h_max, growth)
        base = _graded_segment(self.length_cm - self.junction_cm, h_min, h_max, growth)
        spacing = np.concatenate([emitter, base])
        # The junction lands exactly on a node, so every edge lies in a single doping region.
        return np.concatenate([[0.0], np.cumsum(spacing)])


def _graded_segment(length: float, h_min: float, h_max: float,
                    growth: float) -> NDArray[np.float64]:
    """Spacings that grow geometrically from both ends toward the middle."""
    steps = []
    total, h = 0.0, h_min
    while total + h < length / 2.0:
        steps.append(h)
        total += h
        h = min(h * growth, h_max)
    if not steps:
        return np.array([length])
    half = np.asarray(steps)
    spacing = np.concatenate([half, half[::-1]])
    return spacing * (length / spacing.sum())
