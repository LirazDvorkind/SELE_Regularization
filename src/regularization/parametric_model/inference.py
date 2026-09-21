"""From an ELE measurement to a family of plausible SELE curves.

The network gives a Gaussian over normalised parameters. Everything after that is physics:
draw parameter vectors, run the simulator, and the result is an ensemble of real profiles, so
whatever is read off it stays physically admissible no matter how wide it gets.

What this module reports are percentiles of that ensemble as drawn, with every sample counting
equally. That is the network's own spread and it is *not* what the figures or the headline
numbers use -- ``uncertainty.py`` ranks the draws by how well each reproduces the measurement
and reports the envelope of the best-fitting fraction, which is the calibrated statement. The
functions here remain the raw view, useful for seeing what the network alone believes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from src.forward_model.analytic_ele import simulate_ele
from src.forward_model.ele import mesh_and_operator
from src.forward_model.parameters import ParameterSpec
from src.forward_model.sele_simulator import simulate_sele
from src.regularization.parametric_model import features as feat
from src.regularization.parametric_model.model_definition import build_parametric_network


@dataclass(frozen=True)
class PosteriorModel:
    network: torch.nn.Module
    input_mode: str
    input_stats: Dict[str, NDArray[np.float64]]
    spec: ParameterSpec


@dataclass(frozen=True)
class SeleEnsemble:
    """Simulated curves for parameter draws, plus the measurement they were inferred from."""

    z_cm: NDArray[np.float64]
    sele: NDArray[np.float64]              # (n_samples, n_depths)
    ele: NDArray[np.float64]               # (n_samples, n_wavelengths), refit through G
    params: NDArray[np.float64]            # (n_samples, 5), physical units
    mean_normalized: NDArray[np.float64]   # (5,)
    covariance_normalized: NDArray[np.float64]  # (5, 5)

    def percentile(self, q) -> NDArray[np.float64]:
        return np.percentile(self.sele, q, axis=0)

    def ele_percentile(self, q) -> NDArray[np.float64]:
        return np.percentile(self.ele, q, axis=0)


def load_model(checkpoint_path: str, device: Optional[torch.device] = None) -> PosteriorModel:
    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    network = build_parametric_network(checkpoint["config"])
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(device).eval()

    stats = {k: np.asarray(v, dtype=np.float64)
             for k, v in checkpoint["input_stats"].items()}
    return PosteriorModel(
        network=network,
        input_mode=checkpoint["config"]["input_mode"],
        input_stats=stats,
        spec=ParameterSpec.from_dict(checkpoint["param_spec"]),
    )


def predict_posterior(
        model: PosteriorModel,
        ele: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(mean, covariance)`` over *normalised* parameters for one measurement."""
    x = feat.transform(np.atleast_2d(ele), model.input_mode, model.input_stats)
    with torch.no_grad():
        mean, scale_tril = model.network(torch.tensor(x, dtype=torch.float32))
    covariance = (scale_tril @ scale_tril.transpose(-1, -2))[0].numpy().astype(np.float64)
    return mean[0].numpy().astype(np.float64), covariance


def sample_sele(
        model: PosteriorModel,
        ele: NDArray[np.float64],
        n_samples: int = 4000,
        resolution: int = 500,
        seed: int = 0,
) -> SeleEnsemble:
    mean, covariance = predict_posterior(model, ele)

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean, covariance, size=n_samples, method="cholesky")
    # The simulator was only ever exercised inside the sampled box, and a Gaussian has
    # unbounded tails, so clip rather than extrapolate the physics.
    params = model.spec.from_normalized(model.spec.clip_normalized(draws))

    z_cm, _ = mesh_and_operator(resolution)
    sele = simulate_sele(params, z_cm)

    # The refit is the closed-form ELE, not G applied to the display mesh. Folding the
    # coarse-mesh discretisation error (around 50% at 500 elements) into the refit would put a
    # floor under the residual that has nothing to do with how good the posterior is.
    ele_refit = simulate_ele(params)

    finite = np.isfinite(sele).all(axis=1) & np.isfinite(ele_refit).all(axis=1)
    if not finite.all():
        print(f"  dropped {int((~finite).sum())} non-finite sample(s)")

    return SeleEnsemble(
        z_cm=z_cm,
        sele=sele[finite],
        ele=ele_refit[finite],
        params=params[finite],
        mean_normalized=mean,
        covariance_normalized=covariance,
    )


def mean_parameters(model: PosteriorModel, ensemble: SeleEnsemble) -> NDArray[np.float64]:
    """The posterior mean in physical units, clipped into the sampled box.

    The network can place its mean slightly outside ``[-1, 1]`` when a measurement sits at the
    edge of, or outside, the training distribution. Un-clipped, that prints impossible
    parameters such as a negative diffusion coefficient.
    """
    return model.spec.from_normalized(model.spec.clip_normalized(ensemble.mean_normalized))


def sample_residuals(
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Relative residual of every posterior draw against the measurement."""
    measured = np.asarray(measured_ele, dtype=np.float64).ravel()
    return (np.linalg.norm(ensemble.ele - measured, axis=1) / np.linalg.norm(measured))


def ele_residual(ensemble: SeleEnsemble, measured_ele: NDArray[np.float64]) -> float:
    """Residual of the depth-wise median reconstruction.

    What you get by taking the median curve at face value. It is not the best fit available:
    the forward model is nonlinear, so the curve from the median parameters and the median of
    the curves are different objects, and neither minimises the residual.
    """
    measured = np.asarray(measured_ele, dtype=np.float64).ravel()
    predicted = np.median(ensemble.ele, axis=0)
    return float(np.linalg.norm(predicted - measured) / np.linalg.norm(measured))


def mean_parameter_residual(
        model: PosteriorModel,
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
) -> float:
    """Residual of the single parameter vector at the posterior mean."""
    measured = np.asarray(measured_ele, dtype=np.float64).ravel()
    predicted = simulate_ele(mean_parameters(model, ensemble)[np.newaxis, :])[0]
    return float(np.linalg.norm(predicted - measured) / np.linalg.norm(measured))


def residual_report(
        model: PosteriorModel,
        ensemble: SeleEnsemble,
        measured_ele: NDArray[np.float64],
) -> str:
    """How well the posterior explains the measurement, from three angles.

    The best draw answers whether the posterior contains parameters that reproduce the data
    at all; if it is small, the method has found the answer. The fraction of draws fitting
    well answers whether the posterior is as tight as the data allows; if it is small, the
    bands are wider than the measurement warrants and the spread is the network's resolution
    rather than the physics.
    """
    residuals = sample_residuals(ensemble, measured_ele)
    return "\n".join([
        f"  best draw            {residuals.min() * 100:7.3f}%",
        f"  typical draw         {np.median(residuals) * 100:7.3f}%",
        f"  median curve         {ele_residual(ensemble, measured_ele) * 100:7.3f}%",
        f"  at posterior mean    {mean_parameter_residual(model, ensemble, measured_ele) * 100:7.3f}%",
        f"  draws fitting <1%    {float((residuals < 0.01).mean()) * 100:7.1f}%",
    ])


def band_report(ensemble: SeleEnsemble) -> str:
    """Width of the 68% SELE band relative to the median, by depth zone.

    A posterior can fit the measurement perfectly and still be useless if the bands are wide
    everywhere; the feasibility work predicted exactly that beyond a few micrometres.
    """
    low, median, high = (ensemble.percentile(q) for q in (15.865, 50.0, 84.135))
    relative = np.divide(high - low, np.abs(median),
                         out=np.full_like(median, np.nan), where=median != 0)
    z_um = ensemble.z_cm / 1e-4

    lines = ["  depth zone      relative 68% band width"]
    for lo, hi in ((0.0, 3.0), (3.0, 8.0), (8.0, 30.0)):
        zone = (z_um >= lo) & (z_um < hi)
        lines.append(f"  {lo:5.1f}-{hi:4.1f} um    {np.nanmedian(relative[zone]):.3f}")
    return "\n".join(lines)


__all__ = [
    "PosteriorModel", "SeleEnsemble", "load_model", "predict_posterior", "sample_sele",
    "mean_parameters", "sample_residuals", "ele_residual", "mean_parameter_residual",
    "residual_report", "band_report",
]
