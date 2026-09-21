"""Reconstruct SELE with uncertainty bands from an ELE measurement.

    # every ground-truth test-set curve
    python -m src.regularization.parametric_model.standalones.run_inference --test-set

    # one curve, or the paper's simulated measurement
    python -m src.regularization.parametric_model.standalones.run_inference --curve srv_1e5
    python -m src.regularization.parametric_model.standalones.run_inference --ele-sim

Test-set curves are fed their native-mesh ``ele.csv`` rather than ``ele_500.csv``. The
latter was produced by the same 500-element operator the reconstruction uses, which would
let the method invert its own discretisation and flatter the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.forward_model.ele import measurement_wavelengths
from src.regularization.parametric_model.inference import (
    band_report,
    load_model,
    mean_parameters,
    residual_report,
    sample_sele,
)
from src.regularization.parametric_model.plot_parametric_result import save_figures
from src.test_set.loader import load_curve, load_test_set

_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CHECKPOINT = _ROOT / "Data" / "parametric_model" / "models" / "parametric_posterior.pt"
_OUTPUT_DIR = _ROOT / "results" / "parametric_model"


def _report(model, stem, ele, wavelengths, truth_z=None, truth_sele=None,
            n_samples=4000, output_dir=_OUTPUT_DIR):
    print(f"\n{stem}")
    ensemble = sample_sele(model, ele, n_samples=n_samples)

    print(f"  posterior mean: {model.spec.describe(mean_parameters(model, ensemble))}")
    print("  ELE residual against the measurement:")
    print(residual_report(model, ensemble, ele))
    print(band_report(ensemble))

    save_figures(ensemble, wavelengths, ele, output_dir, stem,
                 ground_truth_z_cm=truth_z, ground_truth_sele=truth_sele)
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--curve", type=str, default=None, help="a test-set curve id")
    parser.add_argument("--test-set", action="store_true", help="every test-set curve")
    parser.add_argument("--ele-sim", action="store_true", help="Data/ELE_sim.csv")
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"no checkpoint at {args.checkpoint} -- train one with the Colab notebook in "
            f"{Path(__file__).parent} and copy the .pt there.")

    model = load_model(args.checkpoint)
    wavelengths = measurement_wavelengths()

    if args.ele_sim:
        ele = np.loadtxt(_ROOT / "Data" / "ELE_sim.csv", delimiter=",").ravel()
        # The measurement is ELE_sim.csv, but its depth grid is not Data/z.csv -- that file
        # holds 171 points against the profile's 100000. The test set's paper_gt entry
        # carries the profile together with the mesh it actually lives on.
        reference = load_curve("paper_gt")
        print("\nNote: ELE_sim.csv and the same profile pushed through the current G differ "
              "by up to 20%.\nThat file predates the optics unification, so part of any "
              "residual here is that gap,\nnot the reconstruction. The test-set curves are "
              "self-consistent and are the better score.")
        _report(model, "ele_sim", ele, wavelengths,
                truth_z=reference.z_cm, truth_sele=reference.sele, n_samples=args.samples)

    curves = []
    if args.test_set:
        curves = load_test_set(include_reference=False)
    elif args.curve:
        curves = [load_curve(args.curve)]

    for curve in curves:
        _report(model, curve.curve_id, curve.ele, wavelengths,
                truth_z=curve.z_cm, truth_sele=curve.sele, n_samples=args.samples)

    if not curves and not args.ele_sim:
        parser.error("pick one of --test-set, --curve or --ele-sim")


if __name__ == "__main__":
    main()
