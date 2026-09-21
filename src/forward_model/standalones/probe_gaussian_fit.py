"""Can a single Gaussian stand in for the set of parameters that fit a measurement?

The network's output head is a Gaussian, so it can only ever describe an ellipsoid. This
takes the parameter vectors that actually reproduce a measurement, fits the best Gaussian to
them, draws from it, and pushes those draws back through the physics. If the set really is
ellipsoidal the draws fit about as well as the members did. If the draws are markedly worse,
the set is curved or hollow, the ellipsoid is covering space the physics rejects, and no
amount of training fixes it -- the head is the wrong shape.

    python -m src.forward_model.standalones.probe_gaussian_fit --curve srv_1e5
"""

from __future__ import annotations

import argparse

import numpy as np

from src.forward_model.analytic_ele import simulate_ele
from src.forward_model.parameters import ParameterSpec
from src.test_set.loader import load_curve

_DRAWS = 400_000
_SLACK = 2.0
_MIN_MEMBERS = 200
_GAUSSIAN_DRAWS = 4000


def _residuals(params, measured):
    return np.linalg.norm(simulate_ele(params) - measured, axis=1) / np.linalg.norm(measured)


def probe(curve_id: str, spec: ParameterSpec, seed: int = 11) -> None:
    measured = np.asarray(load_curve(curve_id).ele, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)

    params = spec.sample(_DRAWS, rng)
    residuals = _residuals(params, measured)

    members = residuals <= residuals.min() * _SLACK
    if members.sum() < _MIN_MEMBERS:
        members = residuals <= np.partition(residuals, _MIN_MEMBERS)[_MIN_MEMBERS]
    normalized = spec.to_normalized(params[members])

    mean = normalized.mean(axis=0)
    covariance = np.cov(normalized, rowvar=False)
    draws = rng.multivariate_normal(mean, covariance, size=_GAUSSIAN_DRAWS)
    drawn = _residuals(spec.from_normalized(spec.clip_normalized(draws)), measured)

    kept = residuals[members]
    print(f"\n{curve_id}   {int(members.sum())} members, "
          f"residual {kept.min() * 100:.3f}-{kept.max() * 100:.3f}%")
    print(f"  samples from the fitted Gaussian:")
    for label, value in (("best", drawn.min()), ("median", np.median(drawn)),
                         ("90th pct", np.percentile(drawn, 90))):
        print(f"    {label:<9} {value * 100:7.3f}%")
    inside = float((drawn <= kept.max()).mean())
    print(f"    fraction as good as the worst member   {inside * 100:5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=str, required=True)
    args = parser.parse_args()
    probe(args.curve, ParameterSpec())


if __name__ == "__main__":
    main()
