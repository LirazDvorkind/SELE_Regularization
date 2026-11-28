"""The five free parameters of the SELE forward model.

Transcribed from the randomisation block of ``MATLAB SELE Simulation/create_training_set.m``,
with one deliberate departure: ``D`` reaches down to 5 rather than 50 cm^2/s, and is sampled
log-uniformly over that wider span. See the note below.

Everything else in the simulator (bandgap narrowing, ``ni``, Auger and radiative lifetimes,
the free-carrier blend of ``k``) is derived from ``p0`` rather than sampled independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# Order is fixed: it defines the column order of every parameter array in this project.
PARAMETER_NAMES: Tuple[str, ...] = ("p0", "D", "S", "tau", "alpha_scale")
PARAMETER_UNITS: Tuple[str, ...] = ("cm^-3", "cm^2/s", "cm/s", "s", "-")
# ``D``'s floor is 5, not the 50 of create_training_set.m, and it is sampled log-uniformly.
# D and tau reach the curve *shape* only through Ln = sqrt(D tau_eff), so they are
# interchangeable there, but tau also sets brightness on its own. A floor on D is therefore a
# ceiling on brightness at fixed shape, and at 50 that ceiling sat about an order of magnitude
# below what the lifetime sweep of the test set needs: those curves bottomed out at 6-9%
# residual while every shape in the set is reproducible to a tenth of a percent. Lowering the
# floor fits all 17 to under 0.7%, and the fits ask for 6-40 cm^2/s.
# ``standalones/probe_parameter_bounds.py`` re-derives this one bound at a time.
PARAMETER_LOWER: Tuple[float, ...] = (1e16, 5.0, 200.0, 5e-9, 0.1)
PARAMETER_UPPER: Tuple[float, ...] = (1e19, 200.0, 1e7, 2.5e-7, 10.0)
PARAMETER_IS_LOG: Tuple[bool, ...] = (True, True, True, True, True)


@dataclass(frozen=True)
class ParameterSpec:
    """Bounds and sampling law for the parameter vector, and the map to normalised space.

    Normalised space is ``[-1, 1]`` per parameter, taking ``log10`` first where the MATLAB
    code samples log-uniformly. Because the bounds are exact rather than estimated from data,
    the same transform applies to any dataset without carrying statistics around.
    """

    names: Tuple[str, ...] = PARAMETER_NAMES
    units: Tuple[str, ...] = PARAMETER_UNITS
    lower: Tuple[float, ...] = PARAMETER_LOWER
    upper: Tuple[float, ...] = PARAMETER_UPPER
    is_log: Tuple[bool, ...] = PARAMETER_IS_LOG

    @property
    def size(self) -> int:
        return len(self.names)

    def _bounds_in_transform_space(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        lo = np.array(self.lower, dtype=np.float64)
        hi = np.array(self.upper, dtype=np.float64)
        log = np.array(self.is_log, dtype=bool)
        lo = np.where(log, np.log10(lo), lo)
        hi = np.where(log, np.log10(hi), hi)
        return lo, hi

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """``(n, 5)`` parameter vectors drawn from the same law the MATLAB script uses."""
        lo, hi = self._bounds_in_transform_space()
        drawn = lo + (hi - lo) * rng.random((n, self.size))
        return self._invert_transform(drawn)

    def to_normalized(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        params = np.asarray(params, dtype=np.float64)
        lo, hi = self._bounds_in_transform_space()
        log = np.array(self.is_log, dtype=bool)
        transformed = np.where(log, np.log10(np.maximum(params, np.finfo(float).tiny)), params)
        return 2.0 * (transformed - lo) / (hi - lo) - 1.0

    def from_normalized(self, normalized: NDArray[np.float64]) -> NDArray[np.float64]:
        normalized = np.asarray(normalized, dtype=np.float64)
        lo, hi = self._bounds_in_transform_space()
        return self._invert_transform(lo + (normalized + 1.0) * (hi - lo) / 2.0)

    def clip_normalized(self, normalized: NDArray[np.float64]) -> NDArray[np.float64]:
        """Keep samples inside the box the simulator was ever exercised on."""
        return np.clip(np.asarray(normalized, dtype=np.float64), -1.0, 1.0)

    def _invert_transform(self, transformed: NDArray[np.float64]) -> NDArray[np.float64]:
        log = np.array(self.is_log, dtype=bool)
        return np.where(log, np.power(10.0, transformed), transformed)

    def as_dict(self) -> dict:
        """Serialisable form, stored in checkpoints so inference never re-derives bounds."""
        return {
            "names": list(self.names),
            "units": list(self.units),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "is_log": list(self.is_log),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ParameterSpec":
        return cls(
            names=tuple(payload["names"]),
            # Units are only ever printed, so an older payload without them still loads.
            units=tuple(payload.get("units", PARAMETER_UNITS)),
            lower=tuple(float(v) for v in payload["lower"]),
            upper=tuple(float(v) for v in payload["upper"]),
            is_log=tuple(bool(v) for v in payload["is_log"]),
        )

    def describe(self, params: Sequence[float]) -> str:
        return ", ".join(
            f"{name}={value:.4g} {unit}".rstrip(" -")
            for name, unit, value in zip(self.names, self.units, params)
        )


PARAMETER_SPEC = ParameterSpec()
