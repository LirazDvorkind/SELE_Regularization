"""Score the posterior against the ground-truth test set, in both domains.

``run_inference`` reports how well the reconstruction explains the *measurement*. That is the
forward equation and it is the primary bar, but it cannot fail in the one way that matters
most: a degenerate problem admits profiles that reproduce the ELE and are still wrong in
depth. This compares the reported curve -- the best-fitting draw -- and the envelope of the
draws within ``uncertainty.TOLERANCE`` against the known SELE.

Everything is reported **by depth zone**, never as one number over the whole range. The
measurement constrains the first few micrometres and says almost nothing past them, so a
single L2 over 0-30 um is dominated by the part nobody can see: it will rank a reconstruction
that nails the surface below one that happens to guess the deep tail, and it makes a
reweighting that sharpens the constrained region look like a regression. Read each zone
against how much the measurement can say about it.

Coverage is the fraction of depths where the truth lies inside the envelope of the
close-enough draws, the same envelope the figures draw. An envelope that misses the truth is
not an error bar on it: every draw in it explains the measurement, so a miss means the error
is shared by all of them -- a bias of the model or of the measurement, not ambiguity.
``check_uncertainty.py`` shows how coverage and width move with the tolerance.

Comparison runs on the solver mesh over the first 30 um. The profiles span the full 350 um
wafer, but incident light is absorbed within a few um, so the deep tail is not constrained by
any measurement. Resampling is by depth, never by index.

    python -m src.regularization.parametric_model.standalones.score_test_set
    python -m src.regularization.parametric_model.standalones.score_test_set --set forward_model

``--set forward_model`` scores ``Data/test_set_forward_model/`` instead: curves the simulator
itself made, so model mismatch is absent and any error is the method's own. Those curves
carry their true parameters, which are read only after inference, to report how far the best
draw's parameters land from them.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.regularization.parametric_model import uncertainty as unc
from src.regularization.parametric_model.inference import load_model, sample_sele
from src.regularization.parametric_model.standalones.run_inference import _DEFAULT_CHECKPOINT
from src.test_set.loader import load_forward_model_set, load_test_set

_SAMPLES = 4000
_ZONES = ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 30.0))


def _score(model, curve_id, ele, truth_z, truth_sele, n_samples):
    ensemble = sample_sele(model, ele, n_samples=n_samples)

    # The ensemble lives on the solver mesh; the truth may be on its own, finer and deeper.
    truth = np.interp(ensemble.z_cm, truth_z, truth_sele)
    selection = unc.select(ensemble, ele)
    z_um = ensemble.z_cm / 1e-4

    row = ""
    for low, high in _ZONES:
        zone = (z_um >= low) & (z_um < high)
        error = (np.linalg.norm(selection.best_sele[zone] - truth[zone])
                 / np.linalg.norm(truth[zone]))
        row += f"{error * 100:9.1f}%"

    inside = (truth >= selection.low) & (truth <= selection.high)
    near = z_um < 3.0
    print(f"  {curve_id:<14}{row}"
          f"{inside[near].mean() * 100:8.1f}%{inside.mean() * 100:8.1f}%"
          f"{selection.n_kept:7d}{selection.best_misfit * 100:9.3f}%")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--samples", type=int, default=_SAMPLES)
    parser.add_argument("--set", choices=("paper", "forward_model"), default="paper")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    zone_header = "".join(f"{f'{lo:g}-{hi:g}um':>10}" for lo, hi in _ZONES)

    print(f"  best-draw relative error by depth zone, then coverage of the envelope of draws "
          f"within {unc.TOLERANCE:.0%} in ELE, then the best draw's ELE misfit")
    print(f"  {'curve':<14}{zone_header}{'0-3um':>9}{'0-30um':>9}{'kept':>7}{'ELE':>10}")

    if args.set == "paper":
        for curve in load_test_set(include_reference=False):
            _score(model, curve.curve_id, curve.ele, curve.z_cm, curve.sele, args.samples)
        return

    recovered = []
    curves = load_forward_model_set()
    for curve in curves:
        selection = _score(model, curve.curve_id, curve.ele, curve.z_cm, curve.sele,
                           args.samples)
        recovered.append(selection.best_params)

    # Reported in decades: every parameter is sampled log-uniformly, and alpha_scale and p0
    # span orders of magnitude, so a ratio is the comparable unit across all five.
    print("\n  best draw's parameters against the truth, log10(best / true) in decades")
    print(f"  {'curve':<14}" + "".join(f"{name:>12}" for name in model.spec.names))
    for curve, best in zip(curves, recovered):
        decades = np.log10(best / curve.true_params)
        print(f"  {curve.curve_id:<14}" + "".join(f"{d:12.3f}" for d in decades))


if __name__ == "__main__":
    main()
