"""Sanity-check a generated dataset against the MATLAB-made curves already in the repo.

    python -m src.forward_model.standalones.check_dataset [--name parametric_100k]

``Data/score_model/datasets/sele_simulated_1000_curves_500_long.csv`` came out of
``create_training_set.m`` under the same physics, so the two sets are different draws from
one distribution and their summary statistics should overlap. They will not match curve for
curve, and should not.

They no longer share a sampling law exactly: ``D`` now reaches below the MATLAB floor of 50.
That widens the population rather than shifting it, since D reaches SELE mostly through
``Ln = sqrt(D tau_eff)`` and the other four parameters already spread that quantity over the
same range. A drift in the tails is expected here; a drift in the median is not.

This catches the failure the per-curve MATLAB comparison cannot: a correct simulator driven
by a mis-transcribed sampling range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_DATASET_DIR = _ROOT / "Data" / "parametric_model" / "datasets"
_DEFAULT_NAME = "parametric_100k"
_MATLAB = (_ROOT / "Data" / "score_model" / "datasets"
           / "sele_simulated_1000_curves_500_long.csv")

_QUANTILES = (1, 5, 25, 50, 75, 95, 99)


def _summarise(sele: np.ndarray) -> dict:
    peak_index = np.argmax(sele, axis=1)
    return {
        "surface": sele[:, 0],
        "peak": sele[np.arange(sele.shape[0]), peak_index],
        "peak position [element]": peak_index.astype(float),
        "depth-integrated": sele.mean(axis=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default=_DEFAULT_NAME)
    generated_path = _DATASET_DIR / f"{parser.parse_args().name}_sele.npz"

    if not generated_path.exists():
        raise SystemExit(f"{generated_path.name} is missing -- run generate_dataset first.")
    if not _MATLAB.exists():
        raise SystemExit(f"{_MATLAB} is missing -- pull the score-model datasets with DVC.")

    generated = np.load(generated_path)["sele"].astype(np.float64)
    matlab = np.loadtxt(_MATLAB, delimiter=",")
    print(f"Python  : {generated.shape[0]} curves x {generated.shape[1]} depths")
    print(f"MATLAB  : {matlab.shape[0]} curves x {matlab.shape[1]} depths\n")

    python_stats, matlab_stats = _summarise(generated), _summarise(matlab)
    header = "  ".join(f"{q:>9d}%" for q in _QUANTILES)
    for name in python_stats:
        print(f"{name}")
        print(f"  quantile {header}")
        for label, stats in (("python", python_stats), ("matlab", matlab_stats)):
            values = np.percentile(stats[name], _QUANTILES)
            print(f"  {label}   " + "  ".join(f"{v:10.3e}" for v in values))
        print()


if __name__ == "__main__":
    main()
