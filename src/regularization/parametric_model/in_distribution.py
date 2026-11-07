"""Is this measurement the kind of thing the network was trained on?

Every other check asks whether the answer explains the data. This one asks whether the
question was fair. A network given an ELE curve unlike anything in training will still return
a confident mean and a finite band, and the residual can look fine while the numbers mean
nothing -- there is no signal in the output that says "extrapolating".

Two tests, because they fail differently:

* **per-channel range** catches gross extrapolation: some input channel is beyond anything
  seen in training. Cheap and interpretable, but weak on its own -- a point can sit inside
  every channel's range and still be nowhere near the training data.
* **off-manifold distance** catches that second case. The 29 input channels are not
  independent; the training curves lie close to a subspace of about four dimensions, because
  five physical parameters generated them. Distance from that subspace says whether a
  measurement is the sort of curve the simulator can make at all, which per-channel ranges
  cannot see.

Both reference the training set rather than a chosen threshold, so the verdict is "unlike the
training data" rather than "far from the mean in some units".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from src.regularization.parametric_model import features as feat
from src.regularization.parametric_model.inference import PosteriorModel

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = _ROOT / "Data" / "parametric_model" / "datasets" / "parametric_100k_train.npz"

# The subspace is cut where adding components stops describing the data and starts describing
# floating-point noise; 99.99% lands at four for this training set.
_VARIANCE_KEPT = 0.9999
# A measurement further off the subspace than all but this fraction of training curves is
# reported as out of distribution.
_RESIDUAL_PERCENTILE = 99.9


class TrainingEnvelope(NamedTuple):
    mean: NDArray[np.float64]              # (n_features,)
    basis: NDArray[np.float64]             # (k, n_features), rows span the training subspace
    channel_low: NDArray[np.float64]       # (n_features,)
    channel_high: NDArray[np.float64]
    residuals: NDArray[np.float64]         # sorted off-subspace distances of the training set


@dataclass(frozen=True)
class DistributionCheck:
    channels_outside: int
    worst_excess: float          # how far past a channel's training range, in units of that range
    residual: float
    residual_percentile: float
    n_features: int
    subspace_dim: int

    @property
    def in_distribution(self) -> bool:
        return self.channels_outside == 0 and self.residual_percentile <= _RESIDUAL_PERCENTILE


@lru_cache(maxsize=4)
def _envelope(dataset_path: str, input_mode: str, stats_key: bytes) -> TrainingEnvelope:
    payload = np.load(dataset_path, allow_pickle=False)
    stats = {"mean": np.frombuffer(stats_key[:len(stats_key) // 2], dtype=np.float64),
             "std": np.frombuffer(stats_key[len(stats_key) // 2:], dtype=np.float64)}
    features = feat.transform(payload["ele"], input_mode, stats)

    mean = features.mean(axis=0)
    centred = features - mean
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    share = np.cumsum(singular ** 2 / (singular ** 2).sum())
    basis = right[:int(np.searchsorted(share, _VARIANCE_KEPT) + 1)]

    residuals = np.linalg.norm(centred - (centred @ basis.T) @ basis, axis=1)
    return TrainingEnvelope(
        mean=mean,
        basis=basis,
        channel_low=features.min(axis=0),
        channel_high=features.max(axis=0),
        residuals=np.sort(residuals),
    )


def training_envelope(
        model: PosteriorModel,
        dataset_path: Optional[Path] = None,
) -> TrainingEnvelope:
    path = Path(dataset_path or DEFAULT_DATASET)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing -- the in-distribution check is defined against the training "
            f"set, so pull it with DVC or regenerate it.")
    # lru_cache needs hashable arguments, and the stats are what decide the transform.
    stats_key = (np.ascontiguousarray(model.input_stats["mean"], dtype=np.float64).tobytes()
                 + np.ascontiguousarray(model.input_stats["std"], dtype=np.float64).tobytes())
    return _envelope(str(path), model.input_mode, stats_key)


def check(
        model: PosteriorModel,
        ele: NDArray[np.float64],
        dataset_path: Optional[Path] = None,
) -> DistributionCheck:
    envelope = training_envelope(model, dataset_path)
    x = feat.transform(np.atleast_2d(ele), model.input_mode, model.input_stats)[0]

    span = envelope.channel_high - envelope.channel_low
    excess = np.maximum(envelope.channel_low - x, x - envelope.channel_high)
    outside = int((excess > 0).sum())

    centred = x - envelope.mean
    residual = float(np.linalg.norm(
        centred - (centred @ envelope.basis.T) @ envelope.basis))
    percentile = float(np.searchsorted(envelope.residuals, residual)
                       / envelope.residuals.size * 100.0)

    return DistributionCheck(
        channels_outside=outside,
        worst_excess=float(np.max(excess / np.where(span > 0, span, np.nan))),
        residual=residual,
        residual_percentile=percentile,
        n_features=x.size,
        subspace_dim=envelope.basis.shape[0],
    )


def report(
        model: PosteriorModel,
        ele: NDArray[np.float64],
        dataset_path: Optional[Path] = None,
) -> str:
    result = check(model, ele, dataset_path)
    verdict = ("in distribution" if result.in_distribution
               else "OUT OF DISTRIBUTION -- the network is extrapolating and neither the "
                    "median nor the band is meaningful")
    return "\n".join([
        f"  input check: {verdict}",
        f"    channels beyond the training range   {result.channels_outside}"
        f"/{result.n_features}   (worst {result.worst_excess:+.3f} of the channel's range)",
        f"    distance off the {result.subspace_dim}-D training subspace"
        f"   {result.residual:.4f}, at the {result.residual_percentile:.2f}th percentile",
    ])


__all__ = ["TrainingEnvelope", "DistributionCheck", "training_envelope", "check", "report",
           "DEFAULT_DATASET"]
