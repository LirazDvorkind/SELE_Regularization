"""Assert the closed-form ELE equals ``G @ simulate_sele`` on the same mesh.

    python -m src.forward_model.standalones.check_analytic_ele

There are two routes from parameters to ELE: the explicit one, which samples SELE on a mesh
and multiplies by G, and the analytic one, which sums the same series in closed form. The
second is what makes dataset generation fast, and it is worth having only as long as it
provably computes the same thing. Re-run this after any change to either path.

The comparison is made on meshes of at most 20000 elements. Beyond that the *explicit* side
is the one that degrades: it adds up that many terms in floating point and its rounding grows
linearly with the count, while the closed form does a fixed amount of arithmetic whatever the
mesh. Holding it to a tight tolerance at a million elements would be testing the reference,
not the thing under test.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from src.forward_model.analytic_ele import DEFAULT_ELEMENTS, simulate_ele
from src.forward_model.ele import mesh_and_operator, sele_to_ele
from src.forward_model.parameters import PARAMETER_SPEC
from src.forward_model.sele_simulator import simulate_sele

_TRUSTED_MESHES = (100, 500, 2000, 20000)
# Loose enough to absorb the reference's own accumulated rounding at 20000 elements, which is
# around 1e-12, and still ten orders of magnitude tighter than any real disagreement.
_TOLERANCE = 1e-11


def main() -> int:
    params = PARAMETER_SPEC.sample(64, np.random.default_rng(11))
    worst = 0.0

    print("Agreement with the explicit mesh computation")
    for n_elements in _TRUSTED_MESHES:
        centres, operator = mesh_and_operator(n_elements)
        explicit = sele_to_ele(simulate_sele(params, centres), operator)
        analytic = simulate_ele(params, n_elements=n_elements)

        error = float(np.max(np.abs(analytic - explicit) / np.abs(explicit)))
        worst = max(worst, error)
        print(f"  {n_elements:6d} elements: max relative difference {error:.3e}")

    print("\nConvergence of the analytic ELE toward the continuum")
    reference = simulate_ele(params, n_elements=16_000_000)
    for n_elements in (500, 8000, DEFAULT_ELEMENTS):
        value = simulate_ele(params, n_elements=n_elements)
        gap = float(np.max(np.abs(value - reference) / np.abs(reference)))
        print(f"  {n_elements:9d} elements: {gap:.3e} from the limit")

    print("\nThroughput")
    for n_curves in (2000, 20000):
        started = time.time()
        simulate_ele(PARAMETER_SPEC.sample(n_curves, np.random.default_rng(1)))
        elapsed = time.time() - started
        print(f"  {n_curves} curves in {elapsed:.2f}s ({n_curves / elapsed:.0f}/s)")

    print()
    if worst <= _TOLERANCE:
        print(f"PASS: worst relative difference {worst:.3e} <= {_TOLERANCE:.0e}")
        return 0
    print(f"FAIL: worst relative difference {worst:.3e} > {_TOLERANCE:.0e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
