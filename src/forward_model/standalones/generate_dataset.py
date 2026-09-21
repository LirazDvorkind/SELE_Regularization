"""Generate the (ELE, parameters) training set for the parametric posterior network.

    python -m src.forward_model.standalones.generate_dataset [--curves 100000] [--seed 7]

Replaces ``MATLAB SELE Simulation/create_training_set.m`` for this purpose. That script
saves only the curves and discards the parameter values that produced them, which is exactly
what a parameter-inference network needs. Generating here keeps parameters and curves
together by construction, and the port is verified against MATLAB by
``validate_matlab_port.py``.

ELE comes from the closed form in ``analytic_ele``, so there is no depth mesh and no
resolution tradeoff. Two files come out:
  * ``*_train.npz``  -- what Colab needs: ELE and parameters only, a few tens of MB.
  * ``*_sele.npz``   -- a SELE sample on the solver mesh, for sanity plots only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.forward_model.analytic_ele import DEFAULT_ELEMENTS, simulate_ele
from src.forward_model.ele import mesh_and_operator, measurement_wavelengths
from src.forward_model.parameters import PARAMETER_SPEC
from src.forward_model.sele_simulator import simulate_sele

_OUTPUT_DIR = (Path(__file__).resolve().parents[3] / "Data" / "parametric_model" / "datasets")
_SOLVER_RESOLUTION = 500
_SELE_SAMPLE_CURVES = 2000


def generate(n_curves: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    params = PARAMETER_SPEC.sample(n_curves, rng)

    started = time.time()
    ele = simulate_ele(params)
    print(f"  ELE: {n_curves} curves in {time.time() - started:.1f}s")

    n_sample = min(_SELE_SAMPLE_CURVES, n_curves)
    solver_centres, _ = mesh_and_operator(_SOLVER_RESOLUTION)
    started = time.time()
    sele_sample = simulate_sele(params[:n_sample], solver_centres).astype(np.float32)
    print(f"  SELE sample: {n_sample} curves in {time.time() - started:.1f}s")

    # calc_Sp2's 1/(1 - (alpha*Ln)^2) blows up where the absorption length meets the
    # diffusion length, so a draw can land on a singularity. Report rather than hide it.
    finite = np.isfinite(ele).all(axis=1) & (ele > 0).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        print(f"  dropped {dropped} curve(s) with non-finite or non-positive ELE")

    return {
        "ele": ele[finite],
        "params": params[finite],
        "params_normalized": PARAMETER_SPEC.to_normalized(params[finite]),
        "wavelength_nm": measurement_wavelengths(),
        "sele_sample": sele_sample,
        "sele_sample_params": params[:n_sample],
        "sele_sample_z_cm": solver_centres,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    name = args.name or f"parametric_{args.curves // 1000}k"
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.curves} curves "
          f"(ELE in closed form, {DEFAULT_ELEMENTS} effective elements)")
    result = generate(args.curves, args.seed)

    train_path = _OUTPUT_DIR / f"{name}_train.npz"
    np.savez_compressed(
        train_path,
        ele=result["ele"],
        params=result["params"],
        params_normalized=result["params_normalized"],
        wavelength_nm=result["wavelength_nm"],
        param_names=np.array(PARAMETER_SPEC.names),
        param_lower=np.array(PARAMETER_SPEC.lower),
        param_upper=np.array(PARAMETER_SPEC.upper),
        param_is_log=np.array(PARAMETER_SPEC.is_log),
        ele_mesh_elements=DEFAULT_ELEMENTS,
        seed=args.seed,
    )

    sele_path = _OUTPUT_DIR / f"{name}_sele.npz"
    np.savez_compressed(
        sele_path,
        sele=result["sele_sample"],
        z_cm=result["sele_sample_z_cm"],
        params=result["sele_sample_params"],
    )

    print(f"\nWrote {train_path.name} "
          f"({train_path.stat().st_size / 1e6:.1f} MB) -- upload this one to Drive")
    print(f"Wrote {sele_path.name} ({sele_path.stat().st_size / 1e6:.1f} MB) -- local only")
    print(f"ELE range: [{result['ele'].min():.4e}, {result['ele'].max():.4e}]")


if __name__ == "__main__":
    main()
