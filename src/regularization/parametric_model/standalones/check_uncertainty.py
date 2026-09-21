"""Does a misfit-ranked band contain the truth, and how much tighter is it?

``uncertainty.py`` builds a band from the best-fitting fraction of draws rather than from
percentiles of the posterior as drawn. That is only worth reporting if it still contains the
answer, and nothing about a single measurement can establish it -- so it is measured here
against the ground-truth test set and the coverage carries over to real measurements by
resemblance, not by proof.

Two numbers per curve and level: what fraction of depths hold the truth, and how wide the
band is relative to the plain percentile band it replaces. Tighter *and* still covering is
the win; tighter at the cost of coverage is not.

    python -m src.regularization.parametric_model.standalones.check_uncertainty
"""

from __future__ import annotations

import argparse

import numpy as np

from src.regularization.parametric_model import uncertainty as unc
from src.regularization.parametric_model.inference import load_model, sample_sele
from src.regularization.parametric_model.standalones.run_inference import _DEFAULT_CHECKPOINT
from src.test_set.loader import load_test_set

_SAMPLES = 4000
_LEVELS = (0.10, 0.25, 0.68, 0.95)
_SURFACE_UM = 3.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--samples", type=int, default=_SAMPLES)
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    print("  coverage of the truth over 0-3 um, and band width relative to the median")
    print(f"  {'curve':<12}" + "".join(
        f"{f'misfit {l:.2f}':>16}" for l in _LEVELS) + f"{'percentile 95':>16}")
    print(f"  {'':<12}" + "".join(f"{'cover  width':>16}" for _ in _LEVELS) + f"{'cover  width':>16}")

    rows = []
    for curve in load_test_set(include_reference=False):
        ensemble = sample_sele(model, curve.ele, n_samples=args.samples)
        truth = np.interp(ensemble.z_cm, curve.z_cm, curve.sele)
        z_um = ensemble.z_cm / 1e-4
        near = z_um < _SURFACE_UM

        row, record = "", []
        for level in _LEVELS:
            band = unc.misfit_band(ensemble, curve.ele, level)
            covered = ((truth >= band.low) & (truth <= band.high))[near].mean()
            width = np.nanmedian(((band.high - band.low) / np.abs(band.median))[near])
            row += f"{covered * 100:9.1f}%{width:7.3f}"
            record += [covered, width]

        # The band it is meant to replace: plain percentiles of every draw, 95% level.
        low, median, high = (np.percentile(ensemble.sele, q, axis=0)
                             for q in (2.275, 50.0, 97.725))
        covered = ((truth >= low) & (truth <= high))[near].mean()
        width = np.nanmedian(((high - low) / np.abs(median))[near])
        row += f"{covered * 100:9.1f}%{width:7.3f}"
        record += [covered, width]

        print(f"  {curve.curve_id:<12}{row}")
        rows.append(record)

    summary = np.asarray(rows)
    print(f"  {'median':<12}" + "".join(
        f"{np.median(summary[:, 2 * i]) * 100:9.1f}%{np.median(summary[:, 2 * i + 1]):7.3f}"
        for i in range(summary.shape[1] // 2)))


if __name__ == "__main__":
    main()
