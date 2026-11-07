"""How many Gaussians would it take to describe the set of fitting parameters?

``probe_gaussian_fit`` showed one Gaussian cannot: fitted to the correct answer it still
draws mostly rubbish, because the set is a thin curved sheet and an ellipsoid around it is
mostly empty. A mixture can follow a bend, so the question is not whether but how many
components -- a handful makes a mixture head worth retraining, dozens makes it not.

This fits mixtures of increasing size to the set, samples each, and pushes the samples back
through the physics. The number that matters is the hit rate: what fraction of draws fit the
measurement as well as the set members do. One Gaussian sits near a percent. A mixture that
has actually captured the shape should approach a large fraction of that.

Mixtures here are fitted by k-means plus a Gaussian per cluster rather than by EM. The
question is how finely the sheet has to be tiled for flat tiles to follow it, and hard
assignment answers that directly without EM's sensitivity to initialisation.

    python -m src.forward_model.standalones.probe_mixture_size --curve srv_1e5
"""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from src.forward_model.analytic_ele import simulate_ele
from src.forward_model.parameters import ParameterSpec
from src.test_set.loader import load_curve

_DRAWS = 2_000_000
_MEMBERS = 5_000
_COMPONENT_COUNTS = (1, 2, 3, 5, 8, 15, 30, 60, 120)
_SAMPLES_PER_FIT = 4_000
# Without this a cluster smaller than the dimension gets a singular covariance, and the
# mixture scores well by collapsing onto its own training points rather than by fitting.
_COVARIANCE_FLOOR = 1e-6
_MIN_CLUSTER = 10


def _residuals(params, measured):
    return np.linalg.norm(simulate_ele(params) - measured, axis=1) / np.linalg.norm(measured)


def _kmeans(points: NDArray[np.float64], k: int, rng, iterations: int = 60):
    centres = points[rng.choice(len(points), size=1)]
    while len(centres) < k:
        distance = np.min(((points[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        total = distance.sum()
        probability = distance / total if total > 0 else None
        centres = np.vstack([centres, points[rng.choice(len(points), p=probability)]])

    labels = np.zeros(len(points), dtype=int)
    for _ in range(iterations):
        new = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                centres[j] = points[labels == j].mean(axis=0)
    return labels


def _fit_mixture(points: NDArray[np.float64], k: int, rng):
    labels = _kmeans(points, k, rng) if k > 1 else np.zeros(len(points), dtype=int)
    weights, means, covariances = [], [], []
    for j in range(k):
        cluster = points[labels == j]
        if len(cluster) < _MIN_CLUSTER:
            continue
        weights.append(len(cluster) / len(points))
        means.append(cluster.mean(axis=0))
        covariances.append(np.cov(cluster, rowvar=False)
                           + _COVARIANCE_FLOOR * np.eye(points.shape[1]))
    weights = np.asarray(weights)
    return weights / weights.sum(), np.asarray(means), np.asarray(covariances)


def _sample_mixture(weights, means, covariances, n, rng):
    counts = rng.multinomial(n, weights)
    return np.vstack([rng.multivariate_normal(means[j], covariances[j], size=c)
                      for j, c in enumerate(counts) if c > 0])


def probe(curve_id: str, spec: ParameterSpec, seed: int = 11) -> None:
    measured = np.asarray(load_curve(curve_id).ele, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)

    params = spec.sample(_DRAWS, rng)
    residuals = _residuals(params, measured)
    order = np.argsort(residuals)[:_MEMBERS]
    members = spec.to_normalized(params[order])
    threshold = residuals[order].max()

    prior_rate = float((residuals <= threshold).mean())
    print(f"\n{curve_id}")
    print(f"  {_MEMBERS} best of {_DRAWS} draws: residual "
          f"{residuals[order].min() * 100:.3f}-{threshold * 100:.3f}%")
    print(f"  blind prior hit rate {prior_rate * 100:.3f}%")
    print(f"  components   effective   median draw   hit rate   gain over prior")

    for k in _COMPONENT_COUNTS:
        weights, means, covariances = _fit_mixture(members, k, rng)
        draws = _sample_mixture(weights, means, covariances, _SAMPLES_PER_FIT, rng)
        drawn = _residuals(spec.from_normalized(spec.clip_normalized(draws)), measured)
        rate = float((drawn <= threshold).mean())
        print(f"  {k:10d} {len(weights):11d} {np.median(drawn) * 100:12.3f}% "
              f"{rate * 100:9.1f}% {rate / prior_rate:15.0f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=str, default="srv_1e5")
    args = parser.parse_args()
    probe(args.curve, ParameterSpec())


if __name__ == "__main__":
    main()
