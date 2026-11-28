"""How often the predicted Gaussian misses the true parameters, and why.

Training data is noiseless, so the ideal posterior for a measurement is the set of parameter
vectors whose ELE equals it, weighted by the prior. Two questions follow, and they are separate:

- **width** is the ill-posedness. A parameter the data cannot pin down *should* get a wide
  spread; ``sigma / sigma_prior`` near 1 says the measurement told the network nothing about it.
- **calibration** is the network. Even a perfect posterior leaves the truth outside its 2-sigma
  region 4.55% of the time, so only the excess over that is a failure of the model.

A miss is then triaged by pushing draws from the predicted Gaussian back through the forward
model. If some draw reproduces the measured ELE within ``uncertainty.TOLERANCE``, the network
found a different member of the equivalence set and its ellipsoid did not reach the truth's --
ambiguity the data leaves open, compounded by a Gaussian that cannot follow a curved ridge. If
no draw fits, the network found nothing that explains the data: a hard failure.

Scored on the training run's own validation split, rebuilt with the checkpoint's seed, and on
``Data/test_set_forward_model/`` curve by curve.

    python -m src.regularization.parametric_model.standalones.check_calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chi2, norm

from src.forward_model.analytic_ele import simulate_ele
from src.regularization.parametric_model import features as feat
from src.regularization.parametric_model import uncertainty as unc
from src.regularization.parametric_model.inference import PosteriorModel, load_model
from src.regularization.parametric_model.standalones.run_inference import (
    _DEFAULT_CHECKPOINT,
    _OUTPUT_DIR,
)
from src.regularization.parametric_model.standalones.train_parametric_model import (
    TrainingConfig,
    split_indices,
)
from src.test_set.loader import load_forward_model_set

_DRAWS = 1000
_SIGMA_PRIOR = 1.0 / np.sqrt(3.0)  # uniform on [-1, 1] in normalised space
_N_SIGMA = (1.0, 2.0)
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#000000")
_STYLES = ("-", "--", "-.", ":", (0, (5, 1)), "-")


def _nominal(n_sigma: float) -> float:
    """Probability mass inside +-n_sigma of a 1-D Gaussian; the target for every miss rate."""
    return float(2.0 * norm.cdf(n_sigma) - 1.0)


def _validation_split(checkpoint_path: str, data_path: str):
    config = torch.load(checkpoint_path, weights_only=False, map_location="cpu")["config"]
    payload = np.load(data_path, allow_pickle=False)
    validation_idx, _ = split_indices(payload["ele"].shape[0], config["validation_fraction"],
                                      config["seed"])

    # The triage refits with today's simulator. If the dataset was made by an older one, a
    # miss can no longer be told apart from the physics having moved under the network.
    probe = validation_idx[:200]
    resimulated = simulate_ele(payload["params"][probe],
                               n_elements=int(payload["ele_mesh_elements"]))
    drift = np.median(np.linalg.norm(resimulated - payload["ele"][probe], axis=1)
                      / np.linalg.norm(payload["ele"][probe], axis=1))
    if drift > 1e-6:
        print(f"  WARNING: the dataset's ELE differs from the current simulator by {drift:.1%} "
              f"(median). The network learned older physics; the triage and the forward-model "
              f"set below mix the two. Regenerate the dataset and retrain to remove this.")
    return payload["ele"][validation_idx], payload["params_normalized"][validation_idx]


@torch.no_grad()
def _posterior(model: PosteriorModel, ele: np.ndarray):
    """``(mean, scale_tril)`` for a batch, the batched form of ``predict_posterior``."""
    x = torch.tensor(feat.transform(np.atleast_2d(ele), model.input_mode, model.input_stats),
                     dtype=torch.float32)
    mean, scale_tril = model.network(x)
    return mean.numpy().astype(np.float64), scale_tril.numpy().astype(np.float64)


def _statistics(mean, scale_tril, truth):
    residual = truth - mean
    covariance = scale_tril @ np.swapaxes(scale_tril, -1, -2)
    sigma = np.sqrt(np.einsum("nii->ni", covariance))
    whitened = np.linalg.solve(scale_tril, residual[..., None])[..., 0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # ascending: tightest axis first
    along_axes = np.einsum("nij,ni->nj", eigenvectors, residual) / np.sqrt(eigenvalues)
    return {
        "z": residual / sigma,
        "r2": (whitened ** 2).sum(axis=1),
        "z_axes": along_axes,
        "axis_sigma": np.sqrt(eigenvalues),
        "contraction": sigma / _SIGMA_PRIOR,
    }


def _best_misfit(model, mean, scale_tril, ele, n_draws, rng):
    """Lowest relative ELE misfit among ``n_draws`` from each case's predicted Gaussian."""
    n_cases, n_params = mean.shape
    eps = rng.standard_normal((n_cases, n_draws, n_params))
    draws = mean[:, None, :] + np.einsum("nij,ndj->ndi", scale_tril, eps)
    params = model.spec.from_normalized(model.spec.clip_normalized(draws.reshape(-1, n_params)))
    refit = simulate_ele(params).reshape(n_cases, n_draws, -1)
    misfit = (np.linalg.norm(refit - ele[:, None, :], axis=2)
              / np.linalg.norm(ele, axis=1)[:, None])
    return np.nanmin(misfit, axis=1)


def _print_calibration(stats, names):
    n_params = len(names)
    print("\n  miss rate: truth outside the predicted region (a perfect posterior scores the "
          "reference)")
    print(f"  {'':<16}" + "".join(f"{f'>{k:g}sigma':>11}" for k in _N_SIGMA))
    print(f"  {'reference':<16}" + "".join(f"{(1 - _nominal(k)) * 100:10.2f}%" for k in _N_SIGMA))
    for j, name in enumerate(names):
        print(f"  {name:<16}" + "".join(
            f"{(np.abs(stats['z'][:, j]) > k).mean() * 100:10.2f}%" for k in _N_SIGMA))
    print(f"  {'joint (5-D)':<16}" + "".join(
        f"{(stats['r2'] > chi2.ppf(_nominal(k), df=n_params)).mean() * 100:10.2f}%"
        for k in _N_SIGMA))

    print("\n  per parameter: spread of z (1 if calibrated), mean z (bias), "
          "sigma / sigma_prior (1 = data says nothing)")
    print(f"  {'':<16}{'std(z)':>10}{'mean(z)':>10}{'contract':>10}")
    for j, name in enumerate(names):
        print(f"  {name:<16}{stats['z'][:, j].std():10.3f}{stats['z'][:, j].mean():10.3f}"
              f"{np.median(stats['contraction'][:, j]):10.3f}")

    print("\n  along the covariance's own axes, tightest first: the ridge is the loose end")
    print(f"  {'axis':<16}{'sigma':>10}{'std(z)':>10}{'>2sigma':>10}")
    for j in range(n_params):
        z = stats["z_axes"][:, j]
        print(f"  {j + 1:<16}{np.median(stats['axis_sigma'][:, j]):10.4f}{z.std():10.3f}"
              f"{(np.abs(z) > 2).mean() * 100:9.2f}%")


def _print_location(stats, truth, names, threshold):
    """Joint miss rate by where the truth sits in each parameter's range."""
    edges = np.linspace(-1.0, 1.0, 6)
    miss = stats["r2"] > threshold
    print("\n  joint 2-sigma miss rate by where the true value sits in its range "
          "(normalised, low to high)")
    print(f"  {'':<16}" + "".join(f"{f'{lo:+.1f}..{hi:+.1f}':>12}"
                                  for lo, hi in zip(edges[:-1], edges[1:])))
    for j, name in enumerate(names):
        bins = np.clip(np.digitize(truth[:, j], edges) - 1, 0, len(edges) - 2)
        print(f"  {name:<16}" + "".join(f"{miss[bins == b].mean() * 100:11.1f}%"
                                        for b in range(len(edges) - 1)))


def _print_triage(label, best_misfit):
    fits = best_misfit <= unc.TOLERANCE
    print(f"  {label:<26}{best_misfit.size:7d}{fits.mean() * 100:11.1f}%"
          f"{(~fits).mean() * 100:11.1f}%{np.median(best_misfit) * 100:12.3f}%")


def _figure(stats, names, output_dir: Path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.linspace(-5, 5, 400)
    bins = np.linspace(-5, 5, 61)
    for j, name in enumerate(names):
        z = np.clip(stats["z"][:, j], -5, 5)
        counts, _ = np.histogram(z, bins=bins, density=True)
        centres = 0.5 * (bins[1:] + bins[:-1])
        left.plot(centres, counts, color=_COLORS[j], linestyle=_STYLES[j], label=name)
    left.plot(grid, norm.pdf(grid), color="#999999", linewidth=2.5, alpha=0.6,
              label="calibrated N(0,1)", zorder=0)
    left.set_xlabel("z = (true - predicted mean) / predicted sigma")
    left.set_ylabel("density")
    left.set_title("Per-parameter error in units of the predicted sigma")
    left.legend(frameon=False, fontsize=8)

    nominal = np.linspace(0.01, 0.99, 99)
    n_params = len(names)
    for j, name in enumerate(names):
        empirical = [(np.abs(stats["z"][:, j]) <= norm.ppf(0.5 + p / 2)).mean() for p in nominal]
        right.plot(nominal, empirical, color=_COLORS[j], linestyle=_STYLES[j], label=name)
    joint = [(stats["r2"] <= chi2.ppf(p, df=n_params)).mean() for p in nominal]
    right.plot(nominal, joint, color=_COLORS[-1], linewidth=2.2, label="joint (5-D)")
    right.plot([0, 1], [0, 1], color="#999999", linewidth=2.5, alpha=0.6, zorder=0,
               label="calibrated")
    right.set_xlabel("nominal coverage")
    right.set_ylabel("fraction of truths inside")
    right.set_title("Coverage: above the diagonal is too loose, below too tight")
    right.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  figure: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--data", type=str, default=TrainingConfig().data_path)
    parser.add_argument("--draws", type=int, default=_DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    names = model.spec.names
    n_params = len(names)
    threshold = chi2.ppf(_nominal(2.0), df=n_params)
    rng = np.random.default_rng(args.seed)

    ele, truth = _validation_split(args.checkpoint, args.data)
    print(f"  validation split: {ele.shape[0]} curves")
    mean, scale_tril = _posterior(model, ele)
    stats = _statistics(mean, scale_tril, truth)
    _print_calibration(stats, names)
    _print_location(stats, truth, names, threshold)

    # Hits are triaged too, on an equal-sized sample: that is the baseline for how often the
    # draws reproduce the ELE when nothing has gone wrong.
    miss = np.flatnonzero(stats["r2"] > threshold)
    hit = rng.choice(np.flatnonzero(stats["r2"] <= threshold), size=miss.size, replace=False)
    print(f"\n  triage: {args.draws} draws per case, does any reproduce the ELE within "
          f"{unc.TOLERANCE:.0%}?")
    print(f"  {'':<26}{'cases':>7}{'some fit':>11}{'none fit':>11}{'median best':>12}")
    _print_triage("joint 2-sigma misses", _best_misfit(model, mean[miss], scale_tril[miss],
                                                       ele[miss], args.draws, rng))
    _print_triage("hits (equal sample)", _best_misfit(model, mean[hit], scale_tril[hit],
                                                      ele[hit], args.draws, rng))

    _figure(stats, names, _OUTPUT_DIR)

    curves = load_forward_model_set()
    fm_ele = np.stack([c.ele for c in curves])
    fm_truth = model.spec.to_normalized(np.stack([c.true_params for c in curves]))
    fm_mean, fm_tril = _posterior(model, fm_ele)
    fm = _statistics(fm_mean, fm_tril, fm_truth)
    fm_best = _best_misfit(model, fm_mean, fm_tril, fm_ele, args.draws, rng)
    print("\n  forward-model test set: z per parameter, joint Mahalanobis radius, "
          "best draw's ELE misfit")
    print(f"  {'curve':<14}" + "".join(f"{n:>9}" for n in names)
          + f"{'radius':>9}{'2sig?':>7}{'best ELE':>10}")
    for i, curve in enumerate(curves):
        outside = "miss" if fm["r2"][i] > threshold else ""
        print(f"  {curve.curve_id:<14}" + "".join(f"{v:9.2f}" for v in fm["z"][i])
              + f"{np.sqrt(fm['r2'][i]):9.2f}{outside:>7}{fm_best[i] * 100:9.3f}%")
    print(f"  joint 2-sigma misses: {(fm['r2'] > threshold).sum()} of {len(curves)} "
          f"(radius beyond {np.sqrt(threshold):.2f})")


if __name__ == "__main__":
    main()
