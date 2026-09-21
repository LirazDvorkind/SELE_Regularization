"""Score the posterior against the ground-truth test set, in both domains.

``run_inference`` reports how well the reconstruction explains the *measurement*. That is the
forward equation and it is the primary bar, but it cannot fail in the one way that matters
most: a degenerate problem admits profiles that reproduce the ELE and are still wrong in
depth. This compares the predicted band against the known SELE.

Everything is reported **by depth zone**, never as one number over the whole range. The
measurement constrains the first few micrometres and says almost nothing past them, so a
single L2 over 0-30 um is dominated by the part nobody can see: it will rank a reconstruction
that nails the surface below one that happens to guess the deep tail, and it makes a
reweighting that sharpens the constrained region look like a regression. Read each zone
against how much the measurement can say about it.

Coverage is the fraction of depths where the truth lies inside the **reported** band -- the
misfit-ranked one from ``uncertainty.py``, the same band the figures draw -- not percentiles
of the raw posterior, which are not what this method ships. ``check_uncertainty.py`` compares
the two and is where the level was chosen. A band that misses the truth is wrong however
tight it is, and one that contains it everywhere while being enormous is not an answer
either, so read coverage alongside width.

Comparison runs on the solver mesh over the first 30 um. The profiles span the full 350 um
wafer, but incident light is absorbed within a few um, so the deep tail is not constrained by
any measurement. Resampling is by depth, never by index.

    python -m src.regularization.parametric_model.standalones.score_test_set
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from src.regularization.parametric_model import uncertainty as unc
from src.regularization.parametric_model.inference import load_model, sample_sele
from src.regularization.parametric_model.standalones.run_inference import _DEFAULT_CHECKPOINT
from src.test_set.loader import load_test_set

_SAMPLES = 4000
_ZONES = ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 30.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--samples", type=int, default=_SAMPLES)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    zone_header = "".join(f"{f'{lo:g}-{hi:g}um':>10}" for lo, hi in _ZONES)

    print(f"  median-curve relative error by depth zone, then coverage of the reported "
          f"band (best-fitting {unc.REPORTED_LEVEL:.0%} of draws)")
    print(f"  {'curve':<12}{zone_header}{'0-3um':>9}{'0-30um':>9}")

    for curve in load_test_set(include_reference=False):
        ensemble = sample_sele(model, curve.ele, n_samples=args.samples)

        # The ensemble lives on the solver mesh; the truth is on its own, finer and deeper.
        truth = np.interp(ensemble.z_cm, curve.z_cm, curve.sele)
        band = unc.misfit_band(ensemble, curve.ele, unc.REPORTED_LEVEL)
        z_um = ensemble.z_cm / 1e-4

        row = ""
        for low, high in _ZONES:
            zone = (z_um >= low) & (z_um < high)
            error = (np.linalg.norm(band.median[zone] - truth[zone])
                     / np.linalg.norm(truth[zone]))
            row += f"{error * 100:9.1f}%"

        inside = (truth >= band.low) & (truth <= band.high)
        near = z_um < 3.0
        print(f"  {curve.curve_id:<12}{row}"
              f"{inside[near].mean() * 100:8.1f}%{inside.mean() * 100:8.1f}%")


if __name__ == "__main__":
    main()
