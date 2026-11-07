"""Which parameter bound is keeping the lifetime sweep from being fitted?

The best-fitting draws for those curves sit hard against the lower bounds of ``D`` and
``alpha_scale``, which makes the box itself the suspect rather than the physics. This relaxes
one bound at a time and reports the best residual reachable for a few representative curves,
so the responsible bound is identified rather than guessed at.

The bounds in ``parameters.py`` are transcribed from the randomisation block of
``create_training_set.m``. That is a choice about what to put in a training set, not a
statement about what a wafer can do, so it is fair game to question.

    python -m src.forward_model.standalones.probe_parameter_bounds
"""

from __future__ import annotations

import argparse

import numpy as np

from src.forward_model.analytic_ele import simulate_ele
from src.forward_model.parameters import ParameterSpec
from src.test_set.loader import load_curve

_DRAWS = 300_000
_SEED = 11
_CURVES = ("srv_1e5", "srv_1e7", "tau_1ns", "tau_8ns", "tau_60ns")

# Variants are expressed against the ORIGINAL create_training_set.m box, not against the
# current spec. The current spec already carries the widened D, so relaxing from it would
# compare nothing -- and a variant that pinned the other bounds to their old values would
# silently put D back at 50 and misattribute the result.
_V1_BOX = {
    "lower": (1e16, 50.0, 200.0, 5e-9, 0.1),
    "upper": (1e19, 200.0, 1e7, 2.5e-7, 10.0),
    "is_log": (True, False, True, True, True),
}

# Each entry relaxes one bound of _V1_BOX. D switches to log sampling when its floor drops, so
# the draws stay spread over the decades rather than piling up at the top.
_VARIANTS = {
    "v1 box (D floor 50)": {},
    "D down to 5": {"lower": (1e16, 5.0, 200.0, 5e-9, 0.1),
                    "is_log": (True, True, True, True, True)},
    "alpha down to 3e-3": {"lower": (1e16, 50.0, 200.0, 5e-9, 3e-3)},
    "p0 up to 1e20": {"upper": (1e20, 200.0, 1e7, 2.5e-7, 10.0)},
    "tau up to 1e-6": {"upper": (1e19, 200.0, 1e7, 1e-6, 10.0)},
    "all four": {"lower": (1e16, 5.0, 200.0, 5e-9, 3e-3),
                 "upper": (1e20, 200.0, 1e7, 1e-6, 10.0),
                 "is_log": (True, True, True, True, True)},
    "current spec": None,
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    measured = {c: np.asarray(load_curve(c).ele, dtype=np.float64).ravel() for c in _CURVES}
    current = ParameterSpec()
    v1 = {**current.as_dict(), **_V1_BOX}

    print("  best residual reachable by random search, per variant of the box")
    print(f"  {'variant':<20}" + "".join(f"{c:>12}" for c in _CURVES))

    for label, overrides in _VARIANTS.items():
        spec = current if overrides is None else ParameterSpec.from_dict({**v1, **overrides})

        predicted = simulate_ele(spec.sample(_DRAWS, np.random.default_rng(_SEED)))
        row = ""
        for curve_id in _CURVES:
            m = measured[curve_id]
            residual = np.linalg.norm(predicted - m, axis=1) / np.linalg.norm(m)
            row += f"{np.nanmin(residual) * 100:11.3f}%"
        print(f"  {label:<20}{row}")


if __name__ == "__main__":
    main()
