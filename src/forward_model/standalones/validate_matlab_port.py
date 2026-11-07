"""Check the Python simulator against MATLAB.

Run ``matlab -batch export_validation_curves`` in ``MATLAB SELE Simulation/`` first, then::

    python -m src.forward_model.standalones.validate_matlab_port

Compares two things, because the port rests on both:
  * the SELE curves themselves, against ``compute_sele_curve.m``;
  * the optical constants ``src/optical_constants.py`` reads from the paper figure, against
    the ``.mat`` the MATLAB simulator loads. A silent disagreement there would shift every
    curve without breaking anything visibly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io

from src.forward_model.sele_simulator import simulate_sele
from src.optical_constants import load_optical_constants

_REFERENCE = (Path(__file__).resolve().parents[3] / "MATLAB SELE Simulation"
              / "validation_reference.mat")
_TOLERANCE = 1e-9


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Error relative to each value, floored by the array's own scale so near-zero entries
    do not dominate."""
    scale = np.maximum(np.abs(expected), np.abs(expected).max() * 1e-12)
    return float(np.max(np.abs(actual - expected) / scale))


def compare_optical_constants(reference: np.ndarray) -> float:
    constants = load_optical_constants()
    ours = np.column_stack([constants.wavelength_nm, constants.n, constants.n_bulk,
                            constants.k, constants.k_bulk])
    if ours.shape != reference.shape:
        raise SystemExit(f"optical constants differ in shape: {ours.shape} vs {reference.shape}")
    return _relative_error(ours, reference)


def compare_curves(params: np.ndarray, x_cm: np.ndarray, reference: np.ndarray) -> float:
    ours = simulate_sele(params, x_cm)
    print(f"  {params.shape[0]} curves x {x_cm.size} depths")
    print(f"  SELE range: MATLAB [{reference.min():.4e}, {reference.max():.4e}]")
    print(f"              Python [{ours.min():.4e}, {ours.max():.4e}]")
    return _relative_error(ours, reference)


def main() -> int:
    if not _REFERENCE.exists():
        raise SystemExit(
            f"{_REFERENCE.name} is missing -- run `matlab -batch export_validation_curves` "
            f"in {_REFERENCE.parent} first.")

    reference = scipy.io.loadmat(str(_REFERENCE))

    print("Optical constants (.fig vs .mat)")
    optics_error = compare_optical_constants(np.asarray(reference["optical"], dtype=np.float64))
    print(f"  max relative error: {optics_error:.3e}\n")

    print("SELE curves (Python vs compute_sele_curve.m)")
    curve_error = compare_curves(
        np.asarray(reference["params"], dtype=np.float64),
        np.asarray(reference["x"], dtype=np.float64).ravel(),
        np.asarray(reference["SELE_all"], dtype=np.float64),
    )
    print(f"  max relative error: {curve_error:.3e}\n")

    worst = max(optics_error, curve_error)
    if worst <= _TOLERANCE:
        print(f"PASS: worst relative error {worst:.3e} <= {_TOLERANCE:.0e}")
        return 0
    print(f"FAIL: worst relative error {worst:.3e} > {_TOLERANCE:.0e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
