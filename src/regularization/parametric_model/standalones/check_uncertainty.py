"""How the close-enough envelope behaves as the ELE tolerance changes.

``uncertainty.py`` reports the best-fitting draw and the envelope of every draw within an
absolute ELE misfit of the measurement. This sweeps that tolerance over the ground-truth test
set and prints, per curve, how many draws pass, how often the truth sits inside the envelope
over 0-3 um, and how wide the envelope is relative to the best draw.

Read coverage as a diagnostic, not as a calibration target. Every draw in the envelope
explains the measurement, so when the truth falls outside it the error is shared by all of
them -- a bias of the simulator or of the measurement -- and a looser tolerance that "fixes"
coverage does so only by admitting draws that no longer fit.

    python -m src.regularization.parametric_model.standalones.check_uncertainty
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from src.regularization.parametric_model import uncertainty as unc
from src.regularization.parametric_model.inference import load_model, sample_sele
from src.regularization.parametric_model.standalones.run_inference import _DEFAULT_CHECKPOINT
from src.test_set.loader import load_test_set

_SAMPLES = 4000
_TOLERANCES = (0.005, 0.01, 0.02, 0.05)
_SURFACE_UM = 3.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--samples", type=int, default=_SAMPLES)
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    print(f"  per tolerance: draws kept, coverage of the truth over 0-{_SURFACE_UM:g} um, "
          f"envelope width relative to the best draw")
    print(f"  {'curve':<12}" + "".join(f"{f'tol {t:.1%}':>22}" for t in _TOLERANCES))
    print(f"  {'':<12}" + "".join(f"{'kept  cover  width':>22}" for _ in _TOLERANCES))

    rows = []
    for curve in load_test_set(include_reference=False):
        ensemble = sample_sele(model, curve.ele, n_samples=args.samples)
        truth = np.interp(ensemble.z_cm, curve.z_cm, curve.sele)
        near = ensemble.z_cm / 1e-4 < _SURFACE_UM

        row, record = "", []
        for tolerance in _TOLERANCES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                selection = unc.select(ensemble, curve.ele, tolerance)
            kept = selection.n_kept if selection.met else 0
            covered = ((truth >= selection.low) & (truth <= selection.high))[near].mean()
            width = np.nanmedian(
                ((selection.high - selection.low) / np.abs(selection.best_sele))[near])
            row += f"{kept:8d}{covered * 100:7.1f}%{width:7.3f}"
            record += [kept, covered, width]

        print(f"  {curve.curve_id:<12}{row}")
        rows.append(record)

    summary = np.median(np.asarray(rows), axis=0)
    print(f"  {'median':<12}" + "".join(
        f"{summary[3 * i]:8.0f}{summary[3 * i + 1] * 100:7.1f}%{summary[3 * i + 2]:7.3f}"
        for i in range(len(_TOLERANCES))))


if __name__ == "__main__":
    main()
